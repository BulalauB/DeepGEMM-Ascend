#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/comm/mega_moe.hpp>
#include <deep_gemm/layout/mega_moe.hpp>
#include <deep_gemm/scheduler/mega_moe.hpp>
#include <deep_gemm/simd/mega_moe.hpp>
#include <deep_gemm/utils/math.hpp>

#include <c_api/asc_simd.h>
#include <c_api/cache_ctrl/cache_ctrl.h>
#include <simt_api/device_functions.h>
#include <simt_api/common_functions.h>

namespace deep_gemm::mega_moe {

// 引入布局层的逻辑M块容量，后续用它计算行地址、块数和进度槽。
using layout::BLOCK_M;
// 引入布局层的物理N块宽度，后续用它计算列任务数和权重tile地址。
using layout::BLOCK_N;
// BLOCK_M=256：单expert逻辑路由块的行容量/GM步长；BLOCK_N=256：一个GEMM task的物理输出列数。
// 两者均是tile维度，不是32核的调度wave宽度；实际M由valid_m决定，GMM1的N256为gate128+up128。

// Pipeline choices: double-buffer L0A/B and keep three A / four routed B stages in L1
// 整体生产消费链：AIV0 的 MTE2/V/MTE3 负责 FP4->FP8 NZ 权重；AIC 的 MTE1/M 消费 L1A/L1B；
// AIC FIX 交结果给 AIV1，AIV1 激活/量化写 Linear2 输入或将 Linear2 输出按行回传，空档做输出归并。
// 这里 stage 数控制不同层次的在途容量：L0 两槽、L1A 三槽、routed L1B 四槽、scale 两槽、FIX/epilogue 两槽。
// 各层 empty credit 只能归还自己负责的空间，不能因下一级已提交就提前覆写上一级仍被读取的槽。
// L0A、L0B各自的轮转槽数；各64KiB分两槽，每槽32KiB，MTE1装入与M管线消费交错。
constexpr uint32_t kNumL0Stages = 2;
// kNumL1AStages：L1激活A的3个64KiB槽，共192KiB；MTE2生产，MTE1读入L0后回收。
// kNumL1BStages：routed权重B的4个64KiB槽，共256KiB；AIV0生产，AIC的MTE1消费。
// kNumSFStages：A/B scale各2槽；每槽各16KiB，缓存32个int16 pair即2048个K元素，A/B共用同步节拍。
constexpr uint32_t kNumL1AStages = 3, kNumL1BStages = 4, kNumSFStages = 2;

// Split each logical M-block between two FIX/epilogue buffers
// AIC FIX向AIV1交付GMM1投影的UB槽数，不是权重槽或GMM task数；两槽各承载一个M128结果片段。
constexpr uint32_t kNumEpilogueStages = 2;
// 单个FIX/epilogue槽的最大行数=256/2=128；每槽BF16[128,256]占64KiB，V仍按最多32行小步处理。
constexpr uint32_t STORE_BLOCK_M = BLOCK_M / kNumEpilogueStages;

// FP8 compute shapes
// BLOCK_M  BLOCK_N  BLOCK_K  MAD_M  MAD_N  MAD_K
//     256      256      512    128    128    256
//     256      256      256    256    256    128
// 逻辑M256/N256保持调度与GM布局稳定；实际MAD按有效M选择，权重解包端必须作相同选择。
// M128: K块512、MAD K256/N128，一个逻辑N块做两组N；M256: K块256、MAD K128/N256。
// 两种形状的L1A单槽都为64KiB、routed L1B单槽也都为64KiB，允许跨task沿用槽游标。
// 编译期分别实例化kMadM=128和256；运行时依据valid_m选择对应实例，不在每次MAD时重新求形状。
template <uint32_t kMadM>
// 将逻辑task尺寸、L1搬运尺寸和单条MAD尺寸放在同一配置中，保证AIC与AIV0使用一致的形状。
struct MMAConfig {
    // 逻辑task的M行容量始终256；kMadM是128/256的计算/缓冲配置，实际指令M为Align16(valid_m)。
    static constexpr uint32_t BLOCK_M = layout::BLOCK_M;
    // 逻辑task物理N宽度始终256；M128配置在task内部拆成两次N128，GMM1两半对应gate/up。
    static constexpr uint32_t BLOCK_N = layout::BLOCK_N;
    // L1一次装载的K跨度，单位元素；FP8为1B，使kMadM*BLOCK_K=64KiB，得到M128/K512或M256/K256。
    static constexpr uint32_t BLOCK_K = L0ASizeBytes / kMadM;
    // 配置所能容纳的M上限128/256；不表示有效行少时仍固定计算128/256行。
    static constexpr uint32_t MAD_M = kMadM;
    // 单条MAD的K归约长度；L0A双缓冲每槽32KiB，所以是BLOCK_K/2，分别得到256/128。
    static constexpr uint32_t MAD_K = BLOCK_K / kNumL0Stages;
    // 单条MAD的N宽度；L0B每槽32KiB/FP8的MAD_K，分别得到N128/N256，单位列而非字节。
    static constexpr uint32_t MAD_N = L0BSizeBytes / kNumL0Stages / MAD_K;
    // 编译期检查：逻辑N256必须能由整数个MAD_N覆盖，后面的N循环没有另写残列分支。
    static_assert(BLOCK_N % MAD_N == 0);
    // 编译期检查：该配置完整N256的FP32累加结果必须放得进L0C；M256时恰好256KiB。
    static_assert(MAD_M * BLOCK_N * sizeof(float) <= L0CSizeBytes);

    // Shared B reuses routed B's region; events 5-7 permit at most three stages
    // shared权重由AIC自己MTE2搬入，单槽存完整N256；容量给M128配置2槽、M256配置4槽，再受事件预算限为3槽。
    static constexpr __aicore__ uint32_t get_num_shared_l1b_stages() {
        // B区总256KiB / shared单槽字节数；FP8为1B，故BLOCK_N*BLOCK_K即单槽容量。
        constexpr uint32_t num_slots = kNumL1BStages * L0BSizeBytes / (BLOCK_N * BLOCK_K);
        // 容量够也最多返回3：本地shared B事件只保留5、6、7三个编号，避免占用其他事件族。
        return num_slots < 3 ? num_slots : 3;
    }
};

// AIC layout (KiB); shared and routed B occupy the same L1 region
// Memory  Region       Offset  Stages x size
// L1      A                 0      3 x 64
//         routed B        192      4 x 64
//         shared B        192      2 x 128 (M=128), 3 x 64 (M=256)
//         A SF            448      2 x 16
//         B SF            480      2 x 16
// L0A     A                 0      2 x 32
// L0B     B                 0      2 x 32
// L0C     FP32 accum        0      2 x 64 (M=128), 1 x 256 (M=256)

// AIC: L1A 0–2, SF 3–4, shared L1B 5–6 or 5–7
// 下列PipeEvent<源PIPE,目的PIPE>{base}只保存本核事件基号：实际ID=base+stage_idx。
// Full通知数据可读，Empty通知源缓冲可覆盖；编号不携带UB/L1地址，调用处还要用同一stage_idx选数据槽。
// 同一数字可出现在不同PIPE方向或不同核的事件中；不能仅按数字判断同步的是哪一块空间。
// L1 的字节预算：A 三槽共 192KiB，B 区 [192,448)KiB，A scale [448,480)KiB，B scale [480,512)KiB。
// B 同一物理区域由 shared/routed 时分复用；shared M128 的每槽 128KiB 只能放两槽，M256 的 64KiB 槽最多三槽。
// routed B 的四个 64KiB 槽由 AIV0 送入，不是再额外追加 256KiB；跨阶段须先完成原 B 的消费。
// L0A/B 各两槽 32KiB，M128 与 M256 使用不同 MAD_K/MAD_N，M128 分两次 N128、M256 一次 N256。
// L0A 装载一次后复用于该 K 段所有 N 子乘，最后一个 N 才归还 A credit；B credit 单独推进以避免形状切换串槽。
// kL1AFull：ID0~2，激活GM->L1完成；kSFFull：ID3~4，A/B scale装载完成；均放行MTE1。
// kSharedL1BFull：ID5~6/7，shared权重GM->L1完成；有效槽数由get_num_shared_l1b_stages决定。
constexpr PipeEvent<PIPE_MTE2, PIPE_MTE1> kL1AFull{0}, kSFFull{3}, kSharedL1BFull{5};
// kL1AEmpty：ID0~2，A的L1->L0读取完成；kSFEmpty：ID3~4，当前scale chunk最后读取完成。
// kSharedL1BEmpty：ID5~6/7，shared B最后读取完成；这三类均允许MTE2覆盖对应L1槽。
constexpr PipeEvent<PIPE_MTE1, PIPE_MTE2> kL1AEmpty{0}, kSFEmpty{3}, kSharedL1BEmpty{5};

// AIC: A uses 0–1; split-N B uses 2–3, otherwise readiness shares A's events
// kL0AFull：基号0、两槽ID0~1；kL0BFull：基号2、两槽ID2~3，保证MTE1装载后M才读取。
// N拆两次时用B Full覆盖同队列此前的A/scale装载；N不拆时，B装载就绪也复用A Full事件族。
constexpr PipeEvent<PIPE_MTE1, PIPE_M> kL0AFull{0}, kL0BFull{2};
// A Empty ID0~1在最后一次N子乘读完后归还；B Empty ID2~3在各自MAD读取后归还，二者游标独立。
constexpr PipeEvent<PIPE_M, PIPE_MTE1> kL0AEmpty{0}, kL0BEmpty{2};

// AIV0: FP4 has 3 stages at 0; NZ has 2 stages at 0
// ID0~2：packed FP4的GM->UB完成，允许V执行FP4->FP8 NZ展开；3槽数在aiv0_weight_pipeline中定义。
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kFP4Full{0};
// ID0~2：V不再读取该packed FP4槽，允许MTE2覆盖；位置由fp4_stage_idx取模3，事件本身不维护轮转。
constexpr PipeEvent<PIPE_V, PIPE_MTE2> kFP4Empty{0};
// ID0~1：V已写完对应FP8 NZ槽，允许MTE3将其搬入AIC的L1B。
constexpr PipeEvent<PIPE_V, PIPE_MTE3> kNZFull{0};
// ID0~1：MTE3已读完NZ槽，允许V覆盖；不代表L1B也可覆盖，后者另等跨核kL1BEmpty。
constexpr PipeEvent<PIPE_MTE3, PIPE_V> kNZEmpty{0};
// ID0：metadata SIMT VF结束的V->Scalar握手；AIV0等到后才能把metadata所占UB复用于权重流水。
constexpr PipeEvent<PIPE_V, PIPE_S> kMetadataDone{0};

// AIV1: pull 0–1 and push 2–5, full forward and empty backward
// UB槽数不是M-block数：pull两槽各装一行token/scale；push四槽各装一行BF16输出。
// pull的推进循环上限使用kNumPullStages=2；push推进上限是独立的常量2，并不因有四槽就一次发四行。
// kNumPullStages=2：两份FP8行+scale槽；kNumPushStages=4：四份BF16输出行槽，均为每AIV1的本地UB容量。
constexpr uint32_t kNumPullStages = 2, kNumPushStages = 4;
// kPullFull ID0~1：远端输入->UB完成，放行写本地输入环；kPushFull ID2~5：本地输出->UB完成，放行远端回传。
constexpr PipeEvent<PIPE_MTE2, PIPE_MTE3> kPullFull{0}, kPushFull{2};
// 对应MTE3已读完UB行，允许MTE2覆盖：pull用ID0~1，push用ID2~5；不等同于整块完成计数。
constexpr PipeEvent<PIPE_MTE3, PIPE_MTE2> kPullEmpty{0}, kPushEmpty{2};

// AIV1: top-k weights, epilogue output, and store completion
// ID1：一个epilogue M128 stage的FP32逐行top-k权重已从GM搬到UB，放行V乘权。
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kTopkFull{1};
// ID0：V已消费本stage的权重，允许MTE2复用唯一的权重UB槽；不是两个轮转权重槽。
constexpr PipeEvent<PIPE_V, PIPE_MTE2> kTopkEmpty{0};
// ID0：本stage的SwiGLU/乘权/FP8量化及scale写完，允许MTE3读取并写入GMM2输入GM。
constexpr PipeEvent<PIPE_V, PIPE_MTE3> kQuantFull{0};
// kPullDone ID0：本核一块的输入stores完成后Scalar才发布Dispatched。
// kStoresDone ID2：Linear1Ready或批量CombineSent发布前排空MTE3，保证先有数据、后有完成证书。
constexpr PipeEvent<PIPE_MTE3, PIPE_S> kPullDone{0}, kStoresDone{2};

// AIV0/AIV1: reduce stages 6–7 and the final communication drain
// 每个AIV的归并UB槽数=2；每槽容纳一个token通道chunk的所有有效top-k/shared输入，宽度按剩余UB计算。
constexpr uint32_t kNumReduceStages = 2;
// ID6~7：MTE2装齐当前归并chunk的输入，允许V执行FP32求和。
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kReduceFull{6};
// ID6~7：V完成求和和BF16转换，允许MTE3写最终output。
constexpr PipeEvent<PIPE_V, PIPE_MTE3> kReduceOutputFull{6};
// ID6~7：MTE3已读完本槽，允许MTE2复用；尾部改变chunk宽度前也等待两槽全部归还。
constexpr PipeEvent<PIPE_MTE3, PIPE_MTE2> kReduceEmpty{6};
// ID3：overlap模式进入尾部reduce前，Scalar确认此前MTE3操作完成，随后才可将整块UB重用于归并。
constexpr PipeEvent<PIPE_MTE3, PIPE_S> kCommDone{3};

// AIC/AIV0: MTE3 -> MTE1 full, MTE1 -> MTE3 empty
// Metadata uses flag 2 before routed L1B stages 0–3
// C/V 跨角色同步与本地 HardEvent 区分：
//   AIV0 MTE3 -> AIC MTE1 : kL1BFull，L1B 权重写完放行 Cube 装载；
//   AIC MTE1 -> AIV0 MTE3 : kL1BEmpty，最后一次 L1B 读取结束，允许 Vector 覆盖这个 B 槽；
//   AIC FIX -> AIV1 V/S   : kEpilogueFull，GMM1 UB 投影或 GMM2 GM 输出已写完；
//   AIV1 MTE3/S -> AIC FIX: kEpilogueEmpty，激活写回/对应消费完成，允许下一次 FIX 复用结果槽。
// full 表示数据可读，empty 表示空间可覆盖；通知的 PIPE 决定之前哪个执行队列的操作已到完成边界。
// metadata 的 flag 2 与 routed B 事件同属 AIV0 链路，必须先完成元数据阶段再把这些 flag 用于 B 环。
// 共享 B 先由 AIC 自己搬运，转 routed 阶段时再一次性给 AIV0 初始 L1B credit，避免两个生产者并发写同一 L1。
// IntraBlockEvent<0/1>选择AIC与配对AIV0/AIV1的连接；{base}仍是flag基号，和本核PipeEvent不是同一同步接口。
// AIV0链路flag2：本轮源metadata到齐且全32个AIV0完成块数汇合；AIC消费后该flag才可复用于routed B槽2。
constexpr IntraBlockEvent<0> kMetadataFull{2};
// AIV0链路flag0~3：Full通知B槽写完，Empty通知AIC最后读取完；同基号、相反方向，槽由l1b_stage_idx选择。
constexpr IntraBlockEvent<0> kL1BFull{0}, kL1BEmpty{0};

// AIC/AIV1: FIX -> V/S full, MTE3/S -> FIX empty
// Linear1 and Linear2 share the same epilogue credits
// AIV1链路flag0~1：GMM1分别保护两个M128 UB槽；GMM2复用stage0握手，Full表示FIX写GM完成。
// Full由AIC FIX通知；Empty由AIV1消费完成后通知，允许AIC继续FIX，独立于GM输出环的CombineSent约束。
constexpr IntraBlockEvent<1> kEpilogueFull{0}, kEpilogueEmpty{0};

// Block sync: both AIVs' MTE3 -> AIC Scalar after final output stores
// 最终block同步的flag ID=4，不是等待4个任务；配对AIV0/AIV1各在最终output stores后arrive，AIC等两侧完成再退出。
constexpr uint32_t kCombineDoneID = 4;

// CANN 9.2 forwards mode 12 but does not name write-through-share in its C API enum
// GM store的缓存策略枚举值12（write-through-share）；用于跨角色/共享GM写回，不是同步ID，也不替代ready事件。
constexpr auto kWriteThroughShare = static_cast<asc_store_l2_cache_mode>(12);

// 可暂停的输出归并游标：保存 token/channel 进度及 peer 完成前缀，使等待期间也能做部分归并。
// reduced_token_mask 防止环形遍历重复写已完成 token，completed_blocks 缓存单调前进的 peer 前缀。
// workspace以编译期引用传入；P、Tmax和top-k固定后，下面数组容量也随该kernel实例固定。
template <const auto& workspace>
// token_idx/channel_idx是下次归并的原token行/通道偏移；stage_idx是两槽UB索引，~0u表示尚未初始化。
// num_remaining_tokens统计当前AIV尚未归并的token；valid_topk_mask的bit对应原始top-k槽。
// completed_blocks[p]缓存生产rank p完成的逻辑块前缀；reduced_token_mask只记录本AIV分到的token。
struct CombineReduceState {
    // 一个uint32位图word可记录32个本AIV负责的token；数组长度再按64个AIV的token步长折算。
    static constexpr uint32_t kNumBitsPerWord = sizeof(uint32_t) * 8;
    // 0xffffffff表示reduce槽尚未初始化，首次推进发放empty credit；有效轮转stage只有0/1。
    static constexpr uint32_t kUninitializedStage = ~0u;

    // Resume at a channel boundary
    // token_idx是原输入行号；channel_idx记录下一段通道；stage_idx首次为哨兵，初始化后按0/1轮转。
    uint32_t token_idx, channel_idx = 0, stage_idx = kUninitializedStage;
    // 只统计这个AIV尚未完成的token，不是整个rank剩余路由行数；整行所有通道提交后才减1。
    uint32_t num_remaining_tokens;

    // 当前token各top-k槽的有效位，bit k对应第k条路由；初值0，未置位的槽不参与求和，允许全部路由无效。
    uint32_t valid_topk_mask = 0;
    // shared输出对本AIV是否已证明就绪；就绪后缓存为true，避免每个token反复读取同一完成计数。
    bool shared_output_ready = false;
    // 最近阻塞归并的生产rank，仅用于等待诊断；不是当前AIV所属rank或核号。
    uint32_t blocking_rank_idx = 0;
    // 每个生产rank缓存一个已完成块前缀；初值0，读到更大前缀后可放行落在此前缀内的路由。
    uint64_t completed_blocks[workspace.config.num_ranks]{};

    // One bit per owned token, strided by the AIV count
    // 每word覆盖本AIV轮转分到的32个token；一组64个AIV合计覆盖64*32个原token，{}将位图清零。
    uint32_t reduced_token_mask[ceil_div(workspace.config.num_max_tokens_per_rank, kNumAIVs * kNumBitsPerWord)]{};
};

// Describe staged UB scratch without allocating memory
template <const auto& workspace>
// @param addr reduce区在本AIV UB中的起始字节地址；重叠期避开其他分区，尾处理传0复用整块UB。
// @return 两槽布局中一个槽的视图规格；shape_m=最大求和输入槽数，shape_n=本次通道chunk宽度。
__aicore__ constexpr auto make_reduce_buffer(uint32_t addr = 0) {
    // 引用当前kernel实例的静态布局参数，避免把top-k、H等作为独立运行时参数传来传去。
    constexpr auto& config = workspace.config;
    // 归并最多输入路数=top-k槽数+一个合并后的shared槽；S个shared expert在此仍只占一槽，不是加S。
    constexpr uint32_t kNumInputs = config.num_topk + (config.num_shared_experts > 0);
    // 一条BF16向量的通道数，用作reduce chunk向下取整单位；由编译器向量类型字节宽度/2B得到。
    constexpr uint32_t kNumChannelsPerVector = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);
    // 剩余UB字节数 / 两个stage / 最大输入路数 / BF16每元素2B，先得到每路可容纳的通道数。
    const uint32_t num_channels = (UBSizeBytes - addr) / (kNumReduceStages * kNumInputs * sizeof(bfloat16_t)) /
        // 再除向量通道数并乘回，向下取整到完整向量，保证所有stage都在UB边界内。
        kNumChannelsPerVector * kNumChannelsPerVector;
    // 返回UB地址视图而非分配内存；行数为输入路数，列数不超过H，addr为两stage区域的起点。
    return ub_ptr<bfloat16_t>(kNumInputs, config.hidden < num_channels ? config.hidden : num_channels, addr);
}

// True when all tokens are processed; each step issues one output chunk
// 最多推进 num_steps 个通道 chunk；返回 true 仅表示本 AIV 的全部 token 已处理。
// 返回 false 可能是步数预算耗尽，也可能是某条路由的生产块未完成，调用方可继续其他流水工作。
// 重叠模式在 token 首个 chunk 验证各路由的实际生产块，避免只按 token 顺序猜测远端就绪。
template <const auto& workspace, bool kOverlapReduce>
// @param buffer 本rank对称GM基址；output是最终BF16[T,H]输出，不是某个专家的输出槽。
// @param num_tokens 本rank有效输入行数T，用于原token索引边界。
// @param[in,out] state 当前AIV的可恢复归并游标，保存下次继续所需的token/channel位置。
// @param ub_reduce_inputs 当前阶段可用的两槽UB描述，槽内压紧装入有效top-k和shared输入。
// @param num_steps 本次最多处理的通道chunk数，默认~0u近似不限制；不是token数或wave数。
// @return true表示本AIV全部token已归并；false表示预算耗尽或遇到未就绪路由。
__aicore__ inline bool advance_combine_reduce(__gm__ uint8_t* buffer, __gm__ bfloat16_t* output,
                                      uint32_t num_tokens, CombineReduceState<workspace>& state,
                                      const ub_ptr<bfloat16_t>& ub_reduce_inputs, uint32_t num_steps = ~0u) {
    // 取得当前实例的模型/EP参数；后续地址步长使用注册容量，不用本轮T替代Tmax。
    constexpr auto& config = workspace.config;
    // 一个shared M-block的GMM2列任务数，用来推导有多少AIC会贡献shared完成证书。
    constexpr uint32_t kNumLinear2NBlocks = config.hidden / BLOCK_N;
    // 每个rank持有连续的E/P个routed专家，通过全局expert ID除这个数得到生产rank。
    constexpr uint32_t kNumExpertsPerRank = config.num_experts / config.num_ranks;
    // 编译期决定是否需要读取shared接收槽；多个shared专家在此合并为一份输出。
    constexpr bool kHasSharedExperts = config.num_shared_experts > 0;

    // 本次使用的UB视图一行可装多少通道；重叠期和尾处理期可能使用不同chunk宽度。
    const uint32_t num_chunk_channels = ub_reduce_inputs.shape_n;

    // 接收各路由BF16结果的本地GM区域，物理顺序为[top-k/shared slot,Tmax,H]。
    const auto combine_input = workspace.combine_input.get_ptr(buffer);
    // 按uint64读原始expert ID，使-1等无效值也会在后面被判为>=E并跳过。
    const auto topk_idx = workspace.input.topk_idx.template get_ptr<uint64_t>(buffer);
    // 引用持久游标中的stage字段；下面改变stage_idx时，state里也同步更新。
    auto& stage_idx = state.stage_idx;

    // Wait for all active shared Linear2 cores
    // 仅重叠归并且确有待处理token时提前检查shared；无shared或已经确认就绪则跳过。
    if (kOverlapReduce and kHasSharedExperts and state.num_remaining_tokens > 0 and not state.shared_output_ready) {
        // shared证书累计每个活跃AIC的最后一次贡献，目标为min(shared总task数,32)，不是每个M块都等32。
        state.shared_output_ready = comm::is_block_ready<workspace>(buffer, layout::kLinear2Ready, 0,
            min(ceil_div(num_tokens, BLOCK_M) * kNumLinear2NBlocks, kNumAICores));
        // 当前没有进展条件，返回外层调度器，让权重、Dispatch或epilogue继续推进。
        if (not state.shared_output_ready)
            return false;
    }

    // 每次循环处理当前token，或者在其未ready时将游标移到本AIV负责的下一个token。
    while (state.num_remaining_tokens > 0) {
        // 保存当前token号；下面可能更新state.token_idx，但本次通道循环仍使用这个局部行号。
        const uint32_t token_idx = state.token_idx;
        // 非overlap走连续行；overlap按64行步长访问本AIV的token，抵达尾部后回到自己的起始余数。
        const uint32_t next_token_idx = not kOverlapReduce ? token_idx + 1 :
            token_idx + kNumAIVs < num_tokens ? token_idx + kNumAIVs : token_idx % kNumAIVs;
        // token/64是本AIV序列中的逻辑位置，再除32得到它所在的位图word。
        const uint32_t word_idx = token_idx / (kNumAIVs * state.kNumBitsPerWord);
        // 在word内选择对应bit；使用无符号1u，确保最高位也能正常表示。
        const uint32_t token_bit = 1u << (token_idx / kNumAIVs % state.kNumBitsPerWord);

        // Skip completed tokens
        if (kOverlapReduce and (state.reduced_token_mask[word_idx] & token_bit)) {
            // 轮转访问可能再次遇到已完成行，只移动游标，不重复求和、不重复减少remaining计数。
            state.token_idx = next_token_idx;
            continue;
        }

        // Check the actual producer block for each route before the first chunk
        if (kOverlapReduce and state.channel_idx == 0) {
            // 只有新token的首个chunk重建有效槽位；中途暂停恢复同一token时沿用已经检查过的结果。
            state.valid_topk_mask = 0;
            // 按输入的top-k槽枚举路由，不按到达顺序枚举，也不对重复expert ID去重。
            for (uint32_t topk_slot_idx = 0; topk_slot_idx < config.num_topk; ++ topk_slot_idx) {
                // 将原token行号和top-k槽展平，用于同时索引expert ID与实际生产块映射。
                const uint32_t token_topk_idx = token_idx * config.num_topk + topk_slot_idx;
                // 读取这一条路由选择的全局专家，随后确定哪个rank负责生产结果。
                const uint64_t expert_idx = asc_load_dev(topk_idx + token_topk_idx);
                // 越界或无效路由没有回传结果，也不进入最终求和。
                if (expert_idx >= config.num_experts)
                    continue;
                // 专家连续分布在rank上，商为生产rank；不是当前token最初所在rank。
                const uint32_t peer_idx = expert_idx / kNumExpertsPerRank;
                // Dispatch写下该路由的真实逻辑M-block编号；不能从token顺序猜其生产位置。
                const uint32_t pool_block_idx = asc_load_dev(workspace.route_m_blocks.get_ptr(buffer, token_topk_idx));
                // Completed prefixes only advance; refresh a peer when its cached value is insufficient
                // 引用该peer的缓存前缀，使后续读到的新值可以供本AIV其他token复用。
                auto& completed_blocks = state.completed_blocks[peer_idx];
                // 前缀是右开边界，只有pool_block_idx<completed_blocks才能证明对应结果已写回。
                if (pool_block_idx >= completed_blocks) {
                    // 当前rank的这个槽由peer发布，记录peer完成发送的连续块前缀，不是任意最大块号。
                    completed_blocks = asc_load_dev(workspace.metadata.combine_ready_prefix.get_ptr(buffer, peer_idx));
                    // 刷新后仍不够则本路由未ready；保留未完成token，先让外层流水做其他工作。
                    if (pool_block_idx >= completed_blocks) {
                        // 留下最近阻塞的peer，便于超时诊断打印真正等待的生产方。
                        state.blocking_rank_idx = peer_idx;
                        // 首chunk尚未发起，可安全跳到下个token探测；下次轮转还会重新检查当前token。
                        state.token_idx = next_token_idx;
                        return false;
                    }
                }
                // 此槽有效且已就绪，设置bit供下面的数据装载循环筛选；不在这里执行数值相加。
                state.valid_topk_mask |= 1u << topk_slot_idx;
            }
        }

        // Initialize UB credits on first use
        if (stage_idx == state.kUninitializedStage) {
            // 初次使用这个UB视图时两槽都为空，一次性发放MTE3->MTE2的两个empty credit。
            notify<PIPE_MTE3, PIPE_MTE2, kNumReduceStages>(kReduceEmpty);
            // 从第0槽开始；后续每提交一个chunk按模2切换。
            stage_idx = 0;
        }

        // 从上次保存的通道位置继续，直到当前token的整个H维都处理完；一次迭代对应一个chunk。
        for (uint32_t channel_base = state.channel_idx; channel_base < config.hidden; channel_base += num_chunk_channels) {
            // 最后一个chunk可能不足UB视图宽度，只读取、计算、写回剩余有效通道。
            const uint32_t num_channels = min(config.hidden - channel_base, num_chunk_channels);
            // 等上一轮MTE3不再读取这个UB槽，防止新的MTE2输入覆盖仍在写回的结果。
            wait<PIPE_MTE3, PIPE_MTE2>(kReduceEmpty, stage_idx);

            // Pack valid routes, then shared output; keep the reserved stage stride
            // 实际求和路数从0开始；无效top-k槽被跳过，不需要向UB写一份零向量占位。
            uint32_t num_inputs = 0;
            // 最后可选的一槽是shared输出；它在top-k槽之后，但不读取topk_idx。
            for (uint32_t topk_slot_idx = 0; topk_slot_idx < config.num_topk + kHasSharedExperts; ++ topk_slot_idx) {
                // overlap用已检查的有效位；非overlap已过rank barrier，只需判断原始expert ID有效性。
                if (topk_slot_idx < config.num_topk and (kOverlapReduce ? (state.valid_topk_mask & (1u << topk_slot_idx)) == 0 :
                        asc_load_dev(topk_idx + static_cast<uint64_t>(token_idx) * config.num_topk + topk_slot_idx) >= config.num_experts))
                    continue;
                // get_combine_m_idx按slot*Tmax+token定位行，再乘H并加channel_base定位本次通道段。
                const auto src = combine_input + workspace.get_combine_m_idx(topk_slot_idx, token_idx) * config.hidden + channel_base;
                // 每个有效输入紧凑排num_channels个BF16；num_inputs++为下一路保留位置，stage间仍使用最大槽跨度。
                copy_gm_to_ub(ub_reduce_inputs[stage_idx].ptr() + num_inputs++ * num_channels,
                              src, 1, num_channels, num_channels * sizeof(bfloat16_t), asc_load_l2_cache_mode::NOTALLOC_KEEP);
            }
            // 通知V当前chunk所有有效输入都已加载，不能只等其中一条路由的DMA。
            notify<PIPE_MTE2, PIPE_V>(kReduceFull, stage_idx);

            // Accumulate in FP32, then round once to BF16
            // 在V端消费同一个stage的full，确保下面求和读取的输入数据完整。
            wait<PIPE_MTE2, PIPE_V>(kReduceFull, stage_idx);
            // VF将num_inputs路BF16转FP32累加，最终转回BF16并把结果放到该UB槽前部；这里不再乘top-k权重。
            vf_combine_reduce(ub_reduce_inputs[stage_idx].ptr(), num_inputs, num_channels);
            // 求和完成后才放行MTE3读取结果；这个事件保护输出数据，不是释放整个UB槽。
            notify<PIPE_V, PIPE_MTE3>(kReduceOutputFull, stage_idx);

            // Store BF16 output
            // 在MTE3端等待V完成，再启动最终输出写回，避免读取旧的输入行内容。
            wait<PIPE_V, PIPE_MTE3>(kReduceOutputFull, stage_idx);
            // output按实际原token的[T,H]布局写入；本次只覆盖该token的有效通道段。
            copy_ub_to_gm(output + static_cast<uint64_t>(token_idx) * config.hidden + channel_base,
                          ub_reduce_inputs[stage_idx].ptr(), 1, num_channels,
                          num_channels * sizeof(bfloat16_t), kWriteThroughShare);
            // 这次写出读完UB后释放槽；MTE2下一次复用该槽时会消费此empty。
            notify<PIPE_MTE3, PIPE_MTE2>(kReduceEmpty, stage_idx);
            // 切换另一份UB槽，允许输入装载与前一份结果写出交错；并非切换token或生产rank。
            stage_idx = (stage_idx + 1) % kNumReduceStages;

            // Save the resume position
            // 先记录已提交通道段的末尾；若本次步数耗尽，下一次可从这里继续而不重复写前面的通道。
            state.channel_idx = channel_base + num_channels;
            // 整个H维都提交完成才将token标为完成；一个chunk完成不能提前减少token计数。
            if (state.channel_idx == config.hidden) {
                // 为下一token恢复首chunk位置，从而触发其路由ready检查。
                state.channel_idx = 0;
                // 游标移到当前AIV分到的下一token，连续分摊与轮转分摊由上面的next_token_idx统一处理。
                state.token_idx = next_token_idx;
                // 仅overlap需要位图，防止后续轮转再次遇到这行时重复归并。
                if constexpr (kOverlapReduce)
                    state.reduced_token_mask[word_idx] |= token_bit;
                // 这个AIV的未完成token数减少1；输出真正写完仍由MTE3事件及尾部排空保证。
                -- state.num_remaining_tokens;
            }

            // 每完成一个chunk消耗一步；预算耗尽就返回外层，避免一次归并长期占住调度循环。
            if (-- num_steps == 0)
                // true只说明所有token的写出已提交；不能据此省略最终MTE3 drain/跨rank结束检查。
                return state.num_remaining_tokens == 0;
        }
    }
    // 所有本AIV负责的token都已提交归并结果，交由调用方完成必要的写出排空。
    return true;
}

template <const auto& workspace, bool kOverlapReduce>
// @param buffer/output 本rank对称工作区/最终BF16[T,H]输出；num_tokens是有效T。
// @param rank_idx 本rank号；aiv_idx=2*core_idx+subblock，决定64个AIV间的token和peer检查分工。
// @param[in,out] reduce 先前重叠归并留下的游标，尾处理从该位置继续；sym_buffer提供跨rank地址映射。
__aicore__ __forceinline__ void combine_reduce(__gm__ uint8_t* buffer, __gm__ bfloat16_t* output,
                                              // 本轮有效T、当前rank与全局AIV号；aiv_idx决定本核的token/peer分工。
                                              uint32_t num_tokens, uint32_t rank_idx, uint32_t aiv_idx,
                                              CombineReduceState<workspace>& reduce, const layout::SymBuffer<>& sym_buffer) {
    // 引用本实例的top-k/shared/EP参数；最终归并仍复用前面同一份静态工作区布局。
    constexpr auto& config = workspace.config;

    // 尾处理先停止权重/dispatch/epilogue对UB的使用，再排空MTE3及已启动的reduce槽，
    // 随后从UB地址0重新分配更宽的reduce chunk。重叠期的小chunk与尾部大chunk不能共用在途槽。
    // 非overlap以rank barrier证明所有combine写可读；overlap逐token查生产块前缀，最后仍等所有peer完成标记。
    // Drain communication before reusing UB
    // overlap路径只需先排空本核通信，远端数据完整性稍后按生产块前缀检查。
    if constexpr (kOverlapReduce) {
        // 在MTE3队尾发出完成事件，覆盖此前通信及重叠归并已提交的写出。
        notify<PIPE_MTE3, PIPE_S>(kCommDone);
        // Scalar等待事件到达，确认后续重用UB不会覆盖仍被MTE3读取的数据。
        wait<PIPE_MTE3, PIPE_S>(kCommDone);
    } else {
        // 非overlap通过编号1的跨rank屏障确认全部生产方发送完成，再读取combine_input。
        comm::rank_barrier<workspace, 1>(buffer, sym_buffer, aiv_idx);
    }

    // Drain reduce stages before changing chunk width
    // 只有重叠期真的初始化过reduce槽，才需要消费已发出的empty credit。
    if (reduce.stage_idx != reduce.kUninitializedStage)
        // 两槽都等到可覆盖，防止chunk宽度变化后新旧stage地址发生重叠。
        wait<PIPE_MTE3, PIPE_MTE2, kNumReduceStages>(kReduceEmpty);
    // 恢复未初始化哨兵，下一次advance会按新的UB视图重新发放两槽empty。
    reduce.stage_idx = reduce.kUninitializedStage;

    // Reduce remaining tokens
    // After draining, both AIVs reuse UB from offset 0 for two reduce stages
    // 从UB地址0建立两份最大可用归并槽；权重/epilogue/通信阶段已经结束，可以复用它们的空间。
    constexpr auto ub_reduce_inputs = make_reduce_buffer<workspace>();
    // 循环推进剩余归并，true表示本AIV所有token的结果均已提交；false会继续探测并带超时保护。
    comm::wait_until([&]() __aicore__ {
        // 沿用重叠期保存的token/channel游标，不重新归并之前已完成的通道。
        return advance_combine_reduce<workspace, kOverlapReduce>(buffer, output, num_tokens, reduce, ub_reduce_inputs);
    }, [&]() __aicore__ {
        // 超时若仍在等shared输出，打印本核等待信息，不把它误报成某个peer路由未到达。
        if (config.num_shared_experts > 0 and not reduce.shared_output_ready) {
            // 报告当前rank与核号，帮助定位shared生产/消费链的阻塞点。
            comm::print_local_timeout(rank_idx);
            // 此诊断分支到此结束，外层wait_until会负责超时trap。
            return;
        }
        // This is the last observed blocker; token_idx already points to the next probe
        // 取得最近一次阻塞路由的生产rank；token游标可能已经切到下一待探测行。
        const uint32_t peer_idx = reduce.blocking_rank_idx;
        // 打印等待的peer前缀，false表示尚在等某条路由，而非最终结束标记。
        comm::print_peer_timeout(rank_idx, aiv_idx, peer_idx,
            // 读取接收方缓存区内该peer当前发布的前缀，诊断值使用右开边界语义。
            asc_load_dev(workspace.metadata.combine_ready_prefix.get_ptr(buffer, peer_idx)), false);
    });

    // Empty ranks also wait for peer writes before the next invocation reuses scratch
    // overlap模式还要确认所有peer已结束发送，即使本rank没有输入token也要执行。
    if constexpr (kOverlapReduce) {
        // 64个AIV分摊peer检查，避免每个AIV都重复读取全部rank的结束状态。
        for (uint32_t peer_idx = aiv_idx; peer_idx < config.num_ranks; peer_idx += kNumAIVs) {
            // 定位本rank内由此peer更新的完成前缀；所有位为1表示生产方已完成本轮全部发送。
            const auto prefix_ptr = workspace.metadata.combine_ready_prefix.get_ptr(buffer, peer_idx);
            // 轮询结束哨兵，防止下一次launch复用scratch时仍有上一轮远端写入。
            comm::wait_until([&]() __aicore__ { return asc_load_dev(prefix_ptr) == ~uint64_t{0}; }, [&]() __aicore__ {
                // 超时打印peer及当前前缀，true明确表示等待的是最终结束状态。
                comm::print_peer_timeout(rank_idx, aiv_idx, peer_idx, asc_load_dev(prefix_ptr), true);
            });
        }
    }
}

// AIV0: FP4 weights -> FP8 NZ -> L1B
// AIV0 重放路由 GEMM 的任务序列，将 FP4 权重解包为 FP8 NZ 后送入 AIC 的 L1B。
// FP4、NZ、L1B 各有独立的槽位信用，某一级可复用不代表下一级也已消费；三类游标分别推进。
template <const auto& workspace, bool kOverlapReduce>
// @param buffer 本rank对称GM，读取专家元数据；output为最终BF16[T,H]，仅插入归并时写入。
// @param num_tokens 本rank输入T，用于归并边界/尾部窗口；不是本rank收到的总路由数。
// @param[in,out] reduce 此AIV0拥有的归并进度；num_total_m_blocks为接收路由的逻辑块总数B。
// @param weights 四组预排布权重/scale，本函数只消费routed l1/l2的packed FP4 data。
// @param core_idx 配对AIC核号[0,32)，用来重放相同M/N任务分配。
__aicore__ inline void aiv0_weight_pipeline(__gm__ uint8_t* buffer, __gm__ bfloat16_t* output,
                                            // T用于本AIV归并范围，reduce保存游标，B为本rank已知的routed总块数。
                                            uint32_t num_tokens, CombineReduceState<workspace>& reduce, uint32_t num_total_m_blocks,
                                            // 权重视图按当前G1/G2任务选data，core_idx用于重放配对AIC的任务分配。
                                            const layout::Weights& weights, uint32_t core_idx) {
    // 引用H/I及top-k等固定配置；本函数只展开routed FP4权重，shared权重由AIC自行加载。
    constexpr auto& config = workspace.config;
    // GMM1每task物理N256含gate128/up128，因此中间I通道需要I/128个列任务。
    constexpr uint32_t kNumLinear1NBlocks = config.intermediate_hidden / (BLOCK_N / 2);
    // GMM2每task输出N256，H通道需要H/256个列任务。
    constexpr uint32_t kNumLinear2NBlocks = config.hidden / BLOCK_N;
    // 本实例GM环的物理块容量，用来判断是否需要在尾部窗口提前插入reduce。
    constexpr uint32_t kNumRingBlocks = workspace.get_num_ring_blocks();

    // AIV0 UB after metadata (KiB)
    // Region   Offset  Stages x size
    // FP4           0      3 x 32
    // FP8 NZ       96      2 x 64.5
    // Reduce      225      2 stages within the remaining 31 KiB
    // AIV0 UB: [0,96)KiB 是三个 packed FP4 槽，每槽32KiB；后接两个 FP8 NZ 槽，各按64KiB+BLOCK_K分配。
    // 额外 BLOCK_K 字节属于转换布局余量，真实末尾由 ub_nz.offset() 计算，不应仅按近似表格中的225KiB硬编码。
    // 剩余 UB 给两槽 reduce，通道 chunk 向整向量长度取整；与权重槽分区不重叠，允许 K-loop 空档推进归并。
    // FP4 credit 在展开完成后归还，NZ credit 在 UB->L1 完成后归还，L1B credit 等 AIC MTE1 最后读取后归还。
    // 三类 ring 索引独立，因为 stage 数为3/2/4；不能用同一 ping-pong 位控制全部缓冲。
    // kNumFP4Stages：packed FP4输入的3个32KiB UB槽；kNumNZStages：展开输出的2个64.5KiB NZ槽。
    // 两者独立轮转，回收点分别为V读完和MTE3读完；数量不同不要求同stage对应同一权重tile。
    constexpr uint32_t kNumFP4Stages = 3, kNumNZStages = 2;
    // FP4 UB从地址0开始，每槽按64KiB/2预留32KiB，以字节读取两个4bit值的打包表示。
    constexpr ub_ptr<uint8_t> ub_fp4(1, L0BSizeBytes / 2);
    // NZ区紧跟三份FP4槽；每槽按64KiB加512B布局余量分配，兼容两种MAD配置。
    constexpr ub_ptr<uint8_t> ub_nz(1, L0BSizeBytes + MMAConfig<128>::BLOCK_K, ub_fp4.offset(kNumFP4Stages));
    // 从两份NZ槽之后的剩余UB建立小chunk归并区，与权重三段流水的缓冲不重叠。
    constexpr auto ub_reduce_inputs = make_reduce_buffer<workspace>(ub_nz.offset(kNumNZStages));

    // 编译期同时检查权重区未越界且剩余UB至少能容纳一个有效向量宽度的归并chunk。
    static_assert(ub_nz.offset(kNumNZStages) <= UBSizeBytes and ub_reduce_inputs.shape_n > 0, "AIV0 UB overflow");

    // Initialize UB reuse credits; AIC releases the initial L1B credits
    // 开始时三个FP4槽都未被V读取，向MTE2发放三个可写credit。
    notify<PIPE_V, PIPE_MTE2, kNumFP4Stages>(kFP4Empty);
    // 开始时两个NZ槽都为空，向V发放两个可写credit；L1B初始credit仍须等AIC交接。
    notify<PIPE_MTE3, PIPE_V, kNumNZStages>(kNZEmpty);

    // Size the reduce window from local output tokens
    // 把本rank原输入T换算成M-block数量，作为尾部reduce窗口跨度，不是接收路由总块数。
    const uint32_t num_reduce_window_m_blocks = ceil_div(num_tokens, BLOCK_M);
    // 本AIV0私有调度游标，重放配对AIC的routed任务顺序，不从全局队列抢任务。
    sched::LinearTaskState schedule;
    // GMM1/GMM2各保留一个专家游标，两层交错时不互相破坏各自的expert/block位置。
    sched::ExpertCursor<workspace> expert_cursors[2]{};
    // FP4和NZ分别从槽0开始，后续独立按3和2取模；不存在共用的ping-pong位。
    uint32_t fp4_stage_idx = 0, nz_stage_idx = 0;
    // AIV0写入配对AIC的L1B槽号，后续按4取模并逐槽等待AIC归还空间。
    uint32_t l1b_stage_idx = 0;

    // 持续准备本核任务的权重，直到静态调度器报告全部routed任务耗尽。
    while (true) {
        // 接收本次任务属于GMM1还是GMM2；kNone表示没有下一任务。
        sched::BlockPhase block_phase;
        // AIC handles shared weights; zero shared tokens skips those tasks
        // 传shared token数0以跳过shared阶段，只获取与配对AIC相同的routed M/N任务。
        const auto task_idx = sched::take_linear_task<workspace>(0, num_total_m_blocks, schedule, core_idx, block_phase);
        // 本函数入口已知道metadata总块数，所以这里kNone是结束，不是尚未收到metadata。
        if (block_phase == sched::BlockPhase::kNone)
            // 退出权重流水，之后外层会进入最终combine_reduce。
            break;

        // Select task weights
        // 判断当前任务所属线性层，以选择权重、K长度和独立专家游标。
        const bool is_linear1 = block_phase == sched::BlockPhase::kLinear1;
        // 选择本层每个M-block对应的N任务数，用于展平task的商/余数解码。
        const uint32_t num_n_blocks = is_linear1 ? kNumLinear1NBlocks : kNumLinear2NBlocks;
        // GMM1沿H归约，GMM2沿I归约；完整K长度决定权重tile在GM中的行跨度。
        const uint32_t num_k = is_linear1 ? config.hidden : config.intermediate_hidden;
        // task除本层列任务数，得到不取模的逻辑M-block号。
        const uint32_t pool_block_idx = task_idx / num_n_blocks;
        // task取余得到本专家内的N256权重块号。
        const uint32_t n_block_idx = task_idx % num_n_blocks;
        // 把GMM1/GMM2映射到0/1，选择各自的专家游标。
        const uint32_t layer_idx = is_linear1 ? 0 : 1;
        // 此处重放配对AIC的task：全部valid_m行、一个物理N256列块以及完整num_k归约；K-loop/MAD是task内部拆分。
        // 定位当前逻辑块所属expert并取得有效行数，保证权重端与AIC选用同一套M128/M256配置。
        const uint32_t valid_m = expert_cursors[layer_idx].advance_to(buffer, pool_block_idx);
        // 专家跨多个M-block时给权重更有利于复用的L2提示；提示不保证命中或跨块驻留。
        const auto weight_l2_hint = expert_cursors[layer_idx].num_expert_tokens > BLOCK_M ?
            // 多块专家使用NORMAL_LAST_VICTIM，否则使用NOTALLOC_KEEP；此行只选择缓存策略。
            asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP;
        // 先选本地专家，再选该专家的N块，用uint64放大索引避免地址乘法窄位溢出。
        const uint64_t weight_block_idx =
            static_cast<uint64_t>(expert_cursors[layer_idx].local_expert_idx) * num_n_blocks + n_block_idx;
        // 选择GMM1或GMM2的packed权重基址，再跳过前面的完整N256块。
        const auto weight_data = (is_linear1 ? weights.l1.data : weights.l2.data) +
                                 // 一个权重块为256*num_k个FP4值，每字节两个，所以块跨度为256*(num_k/2)字节。
                                 weight_block_idx * BLOCK_N * (num_k / 2);

        // 按MAD_M编译两套装载/转换代码，lambda引用外部任务信息与持续轮转的三个槽游标。
        const auto load_weights = [&]<uint32_t MAD_M>() __aicore__ {
            // 当前配置一次L1权重tile的K宽度：M128取512，M256取256。
            constexpr uint32_t BLOCK_K = MMAConfig<MAD_M>::BLOCK_K;
            // 当前配置每次准备的N子块：M128取128，M256取256。
            constexpr uint32_t MAD_N = MMAConfig<MAD_M>::MAD_N;
            // UB内packed数据的行跨度；K个4bit值占K/2字节。
            constexpr uint32_t kNumPackedCols = BLOCK_K / 2;
            // NZ的K方向基础组数量，传给UB->L1复制使展开布局与硬件矩阵布局对应。
            constexpr uint32_t kNumColGroups = BLOCK_K / get_frac_k<uint8_t>();
            // 描述AIC中位于A三槽之后的B区，地址是目标L1视图，不是在AIV0另外分配一块L1。
            constexpr l1_ptr<float8_e4m3_t, Major::K> l1b(MAD_N, BLOCK_K, kNumL1AStages * L0ASizeBytes);

            // 沿整个权重K维分段，任务之间继续复用已有stage游标而非每任务清零。
            for (uint32_t k_idx = 0; k_idx < num_k; k_idx += BLOCK_K) {
                // valid_k remains a multiple of MAD_K, including the final K block
                // 尾段只搬剩余有效K；实际消费不会超过有效MAD_K段，UB预留仍按完整BLOCK_K计算。
                const uint32_t valid_k = min(BLOCK_K, num_k - k_idx);
                // M128配置将N256拆成两份N128准备，M256配置只准备一份N256。
                for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N) {
                    // Load packed FP4
                    // 等V不再读取这个FP4槽，再让MTE2装入下一块packed权重。
                    wait<PIPE_V, PIPE_MTE2>(kFP4Empty, fp4_stage_idx);
                    // GM->UB二维搬运，目的槽由fp4_stage_idx选择，与等待的事件使用同一个索引。
                    copy_gm_to_ub(ub_fp4[fp4_stage_idx],
                                  // 源地址先按N子块跳过整行，再按K偏移除2定位packed字节；const_cast仅适配搬运API类型。
                                  const_cast<__gm__ uint8_t*>(weight_data + static_cast<uint64_t>(n_mad_idx) * (num_k / 2) + k_idx / 2),
                                  // 复制MAD_N行、每行valid_k/2字节；GM行跨度num_k/2，UB行跨度BLOCK_K/2，不能混用字节与元素。
                                  MAD_N, valid_k / 2, num_k / 2, kNumPackedCols, weight_l2_hint);
                    // 装载完成通知V，只放行当前FP4槽对应的数据，不放行整个逻辑任务。
                    notify<PIPE_MTE2, PIPE_V>(kFP4Full, fp4_stage_idx);

                    // Dequantize to FP8 NZ
                    // V等待当前packed输入到齐，避免读取上一轮槽内容。
                    wait<PIPE_MTE2, PIPE_V>(kFP4Full, fp4_stage_idx);
                    // V等待MTE3已读完目标NZ槽，保护反量化输出空间。
                    wait<PIPE_MTE3, PIPE_V>(kNZEmpty, nz_stage_idx);
                    // SIMD把FP4打包值展开为FP8并排成NZ；缩放补偿由预处理后的SF承担，这里转换data。
                    vf_fp4_to_fp8_nz<MAD_N, BLOCK_K>(ub_fp4[fp4_stage_idx].ptr(), ub_nz[nz_stage_idx].ptr());
                    // 转换不再需要packed输入，可提前释放FP4槽给下一次MTE2装载。
                    notify<PIPE_V, PIPE_MTE2>(kFP4Empty, fp4_stage_idx);
                    // 转换结果写完后通知MTE3，NZ槽仍要保留到UB->L1搬运读完。
                    notify<PIPE_V, PIPE_MTE3>(kNZFull, nz_stage_idx);

                    // Publish L1B to AIC
                    // MTE3等待NZ就绪，保证写入L1的是本次转换后的权重。
                    wait<PIPE_V, PIPE_MTE3>(kNZFull, nz_stage_idx);
                    // 等待配对AIC最后一次读取旧B槽；UB空闲不能代替这层L1空间回收条件。
                    wait_intra_block<PIPE_MTE3, 0>(kL1BEmpty, l1b_stage_idx);
                    // 按NZ K组把有效MAD_N行送入对应L1B槽；跳过源NZ布局的padding，不复制成额外权重行。
                    asc_copy_ub2l1(l1b[l1b_stage_idx].template ptr<uint8_t>(), ub_nz[nz_stage_idx].ptr(), kNumColGroups, MAD_N, 1, 0);
                    // L1B写完后通知AIC的MTE1，AIC消费同一个l1b_stage_idx才能读取此tile。
                    notify_intra_block<PIPE_MTE3, 0>(kL1BFull, l1b_stage_idx);
                    // UB->L1读完后归还NZ槽，V可继续转换；此时L1B本身仍等AIC消费。
                    notify<PIPE_MTE3, PIPE_V>(kNZEmpty, nz_stage_idx);

                    // packed输入游标独立按3轮转，事件ID和UB地址都跟随这个新索引。
                    fp4_stage_idx = (fp4_stage_idx + 1) % kNumFP4Stages;
                    // NZ输出游标独立按2轮转，其生命周期比FP4输入更长。
                    nz_stage_idx = (nz_stage_idx + 1) % kNumNZStages;
                    // 目标L1B游标按4轮转，与AIC的B读取/归还顺序保持一致。
                    l1b_stage_idx = (l1b_stage_idx + 1) % kNumL1BStages;
                }

                // Reduce one chunk in unused UB
                // 仅overlap开启、路由块数超过环容量且当前任务进入尾部窗口时插入归并。
                if (kOverlapReduce and num_total_m_blocks > kNumRingBlocks and pool_block_idx + num_reduce_window_m_blocks >= num_total_m_blocks)
                    // 每个K段最多推进一个输出通道chunk，避免归并长时间挤占权重供给流水。
                    advance_combine_reduce<workspace, kOverlapReduce>(buffer, output, num_tokens, reduce, ub_reduce_inputs, 1);
            }
        };
        // 使用有效行数选择计算配置，必须与AIC的routed MAD选择条件完全相同。
        if (valid_m <= 128)
            // 小M走K512/N128权重tile，N256需要连续准备两份子块。
            load_weights.template operator()<128>();
        else
            // 大M走K256/N256权重tile，保持单槽64KiB且与对应MAD匹配。
            load_weights.template operator()<256>();
    }
}

// AIC: shared and routed GEMMs
// AIC 消费输入及权重槽并执行共享/路由 GEMM，结果经 FIX 交给 AIV1 执行 epilogue。
// 共享任务可在路由元数据到达前运行；路由任务必须匹配 AIV0 权重和 AIV1 激活消费顺序。
template <const auto& workspace>
// @param buffer 本rank对称GM基址，含输入、中间激活、输出环及progress。
// @param num_tokens 本rank有效T，决定shared任务；routed任务数从metadata证书读取。
// @param rank_idx 本rank号用于诊断；core_idx为AIC核号[0,32)。
// @param weights routed/shared两层的预排布权重及scale；routed数据由AIV0展开供给，scale由AIC读取。
__aicore__ inline void aic_gemm_pipeline(__gm__ uint8_t* buffer, uint32_t num_tokens, uint32_t rank_idx,
                                         // 选择shared/routed两层权重视图，逻辑core_idx同时决定任务N/M坐标。
                                         const layout::Weights& weights, uint32_t core_idx) {
    // 引用本kernel静态配置；AIC负责shared/routed两层矩阵计算及最终FIX写出。
    constexpr auto& config = workspace.config;
    // routed激活/输出GM环的块容量，跨圈写输出时需要检查旧块CombineSent。
    constexpr uint32_t kNumRingBlocks = workspace.get_num_ring_blocks();
    // 编译期选择shared路径，关闭时相关输入、权重和同步分支被移除。
    constexpr bool kHasSharedExperts = config.num_shared_experts > 0;
    // 一次scale装载缓存32个int16 pair，每pair覆盖64个K元素，即2048个K元素；不是32条token。
    constexpr uint32_t kNumSFPairsPerChunk = 32;
    // A scale区从A三槽和B四槽之后开始；此视图每槽为256行*32个int16 pair。
    constexpr l1_ptr<int16_t> l1_sfa(BLOCK_M, kNumSFPairsPerChunk,
                                   // 两项分别是A区192KiB和B区256KiB，总偏移448KiB，单位字节。
                                   kNumL1AStages * L0ASizeBytes + kNumL1BStages * L0BSizeBytes);
    // B scale紧跟两份A scale，起点480KiB；它也有两份16KiB槽。
    constexpr l1_ptr<int16_t> l1_sfb(BLOCK_N, kNumSFPairsPerChunk, l1_sfa.offset(kNumSFStages));
    // FIX目标是配对AIV1的UB视图：单槽BF16[128,256]，两槽物理地址与AIV1声明一致。
    constexpr ub_ptr<bfloat16_t> ub_accum(STORE_BLOCK_M, BLOCK_N);
    // 编译期证明两份B scale之后不超过512KiB L1，防止布局计算静默越界。
    static_assert(l1_sfb.offset(kNumSFStages) <= L1SizeBytes, "L1 overflow");

    // 指向本核独占cache sector的uint8提交计数，AIV1用它判断是否有可消费的GEMM。
    const auto num_issued_gemms_ptr = workspace.progress.num_issued_gemms.get_ptr(buffer, core_idx);
    // shared处理本rank原输入T，按256行划块；routed块数稍后由metadata证书取得。
    const uint32_t num_shared_m_blocks = kHasSharedExperts ? ceil_div(num_tokens, BLOCK_M) : 0;
    // shared有多个M块时会重复使用同一组权重，用此条件选择加载时的L2提示。
    const bool reuse_shared_weights = num_shared_m_blocks > 1;
    // routed GMM2结果先写本rank输出环，稍后由AIV1逐行发送给源rank。
    const auto routed_output = workspace.routed.output.get_ptr(buffer);
    // shared GMM2无需回传，直接写本rankcombine_input中的shared槽。
    const auto combine_input = workspace.combine_input.get_ptr(buffer);

    // Initialize reuse credits
    // 设置矩阵引擎的N方向模式；这条指令不负责把任务动态分配给各AIC。
    asc_set_mmad_direction_n();
    // L0A两槽初始为空，允许MTE1写入第一组激活。
    notify<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0AEmpty);
    // L0B两槽初始为空，允许MTE1写入第一组权重；A/B使用独立empty事件。
    notify<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0BEmpty);
    // L1A三槽初始为空，允许MTE2从GM装入激活。
    notify<PIPE_MTE1, PIPE_MTE2, kNumL1AStages>(kL1AEmpty);
    // A/B scale两槽初始为空；同一个SF事件族覆盖这两组scale的同步节拍。
    notify<PIPE_MTE1, PIPE_MTE2, kNumSFStages>(kSFEmpty);
    // 有shared阶段才发放本核MTE2可用的shared B初始credit。
    if constexpr (kHasSharedExperts)
        // 为shared B预留事件5~7发放空槽，具体配置只消费其中2个或3个；routed B credit暂不发。
        notify<PIPE_MTE1, PIPE_MTE2, 3>(kSharedL1BEmpty);

    // 本AIC私有任务游标，和配对AIV独立重放同一确定性顺序。
    sched::LinearTaskState schedule;
    // 两层分别保存routed专家游标，避免交错执行G1/G2时互相覆盖位置。
    sched::ExpertCursor<workspace> expert_cursors[2]{};
    // A输入L1槽从0开始，跨K段按3轮转。
    uint32_t l1a_stage_idx = 0;
    // 权重L1槽从0开始；shared和routed复用B区，但槽跨度与可用槽数不同。
    uint32_t l1b_stage_idx = 0;
    // scale chunk槽从0开始，按每2048个K元素的消费边界切换。
    uint32_t sf_stage_idx = 0;
    // 本核GEMM提交计数初始为0，uint8自然回绕；不是本rank全局完成任务总数。
    uint8_t num_issued_gemms = 0;
    // 记录是否已把B区从shared生产者交接给AIV0，防止重复发放routed初始credit。
    bool routed_l1b_initialized = false;
    // 暂未知routed逻辑块数，先置0；不影响shared任务由本地T直接调度。
    uint32_t num_total_m_blocks = 0;  // Routed pool only; shared blocks are separate
    // false表示routed调度仍待metadata，不意味着kernel任务已经耗尽。
    bool metadata_ready = false;
    // AIV0最终汇总的证书，高位为32个AIV0到达数，低24位为routed总块数。
    const auto num_m_blocks_certificate = workspace.metadata.num_m_blocks_certificate.get_ptr(buffer);

    // 持续获取本核下一个task，直到shared与routed全部提交。
    while (true) {
        // 当前task所属的shared/routed及G1/G2阶段，由调度器写入。
        sched::BlockPhase block_phase;
        // 按本核编号重放静态任务分配，不按哪个输入先ready动态抢任务。
        const auto task_idx = sched::take_linear_task<workspace>(
            // 传入本地T、已知routed B及metadata状态；shared可在metadata未到时先运行。
            num_tokens, num_total_m_blocks, schedule, core_idx, block_phase, metadata_ready);
        // kNone既可能是暂时无routed计数，也可能是所有任务已结束，需结合metadata_ready区分。
        if (block_phase == sched::BlockPhase::kNone) {
            // metadata已知时仍无task，说明整个任务序列耗尽。
            if (metadata_ready)
                // 停止提交矩阵任务，外层AIC分支随后等待两侧AIV最终输出完成。
                break;

            // Consume metadata before routed L1B reuses the same intra-block event
            // 在Scalar等待AIV0的metadata flag2；必须先消费它，才能将同flag复用于routed B槽2。
            wait_intra_block<PIPE_S, 0>(kMetadataFull);
            // 只取证书的低24位逻辑块总数，不把高位到达计数误当成块数。
            num_total_m_blocks = asc_load_dev(num_m_blocks_certificate) & layout::kCountValueMask;
            // 此后调度器可以枚举routed任务，不再把kNone解释成尚未收到metadata。
            metadata_ready = true;
            // 重新进入取task循环，按新B继续调度，而不是使用刚才无效的task_idx。
            continue;
        }

        // Decode the GEMM task
        // 识别shared两阶段，决定权重来源、输入区和输出目标。
        const bool is_shared = block_phase == sched::BlockPhase::kSharedLinear1 or block_phase == sched::BlockPhase::kSharedLinear2;
        // 识别G1两阶段，决定是否需要gate/up配对和AIV1的SwiGLU处理。
        const bool is_linear1 = block_phase == sched::BlockPhase::kLinear1 or block_phase == sched::BlockPhase::kSharedLinear1;

        // Shared experts widen Linear1 N and Linear2 K
        // AIC一次task是一个(M-block,N-block)的完整K归约；不是一条MAD，也不是只计算通信owner负责的8行。
        // shared把S个专家的中间通道合并为S*I，routed使用单expert的I。
        const uint32_t intermediate_hidden = is_shared ? config.shared_intermediate_hidden : config.intermediate_hidden;
        // 每个G1物理N256块含gate128/up128，因此列任务数为当前中间宽度/128。
        const uint32_t num_linear1_n_blocks = intermediate_hidden / (BLOCK_N / 2);
        // G1用配对列任务数，G2用H/256，作为task展平索引的行跨度。
        const uint32_t num_n_blocks = is_linear1 ? num_linear1_n_blocks : config.hidden / BLOCK_N;
        // G1归约H，G2归约当前中间宽度；shared的G2归约S*I。
        const uint32_t num_k = is_linear1 ? config.hidden : intermediate_hidden;
        // K方向每64个元素对应一个int16 scale pair，这里得到一整行所需pair数。
        const uint32_t num_sf_pairs = num_k / MX_SF_DIVISOR;
        // task商得到该层不取模的逻辑M块号。
        const uint32_t pool_block_idx = task_idx / num_n_blocks;
        // task余数得到当前专家/共享权重的N256列块号。
        const uint32_t n_block_idx = task_idx % num_n_blocks;
        // G1/G2分别选独立专家游标0/1；shared不读取routed专家游标。
        const uint32_t layer_idx = is_linear1 ? 0 : 1;
        // shared尾块按本地T截断；routed按专家接收数及16行组均分规则取得valid_m。
        const uint32_t valid_m = is_shared ? min(num_tokens - pool_block_idx * BLOCK_M, BLOCK_M) :
            expert_cursors[layer_idx].advance_to(buffer, pool_block_idx);
        // shared多M块或routed多M块专家会复用权重scale，给相应L2缓存提示；不保证物理命中。
        const auto weight_sf_l2_hint = (is_shared ? reuse_shared_weights : expert_cursors[layer_idx].num_expert_tokens > BLOCK_M) ?
            // SF与data使用不同缓存提示：可复用时NORMAL_FIRST_VICTIM，否则NOTALLOC_KEEP。
            asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP;
        // MAD实际M按16行向上对齐；GM读取和FIX输出仍用valid_m，避免写多余有效行。
        const uint32_t aligned_valid_m = aligned(valid_m, FRAC_MN);
        // shared无expert维，直接用N块号；routed先跨expert再跨N块。
        const uint64_t weight_block_idx = is_shared ? n_block_idx :
            static_cast<uint64_t>(expert_cursors[layer_idx].local_expert_idx) * num_n_blocks + n_block_idx;
        // scale以[K-pair,N256]存储，这个元素偏移跳过前面所有权重N块的scale。
        const uint64_t sf_offset = weight_block_idx * num_sf_pairs * BLOCK_N;
        // 在shared/routed与G1/G2四套data/SF指针中选当前task对应的一套。
        const auto task_weights = is_linear1 ? (is_shared ? weights.shared_l1 : weights.l1) : (is_shared ? weights.shared_l2 : weights.l2);

        // Hand L1B over to AIV0 after shared tasks
        // 只有第一次进入routed时交接B区；此前shared的MTE1消费顺序保护同一物理空间。
        if (not is_shared and not routed_l1b_initialized) {
            // 按routed的64KiB槽跨度从B槽0重新开始，不能沿用shared的128KiB/64KiB游标。
            l1b_stage_idx = 0;
            // 由AIC的MTE1边界向AIV0发放四个B空槽，AIV0此后才可写该L1区。
            notify_intra_block<PIPE_MTE1, 0, kNumL1BStages>(kL1BEmpty);
            // 标记已交接，后续routed任务直接沿用4槽轮转，不重复重置credit。
            routed_l1b_initialized = true;
        }

        // Select input data and scales
        // shared激活保持完整块地址，routed激活仅数据区按R取模复用。
        const uint32_t input_block_idx = is_shared ? pool_block_idx : layout::get_ring_block_idx<workspace>(pool_block_idx);
        // 根据当前层选择输入布局：G1来自原输入/Dispatch环，G2来自SwiGLU量化后的中间激活。
        const auto input_layout = is_linear1 ?
            // G1 shared读本rank原FP8输入；G1 routed读AIV1已整理的l1_acts。
            (is_shared ? workspace.input.acts : workspace.routed.l1_acts) :
            // G2分别读shared中间区或routed的l2_acts环，两者都已完成SwiGLU与FP8量化。
            (is_shared ? workspace.shared_l2_acts : workspace.routed.l2_acts);
        // 物理块号乘固定256得到输入GM首行，不能乘valid_m，否则尾块地址会错位。
        const uint64_t input_m_idx = static_cast<uint64_t>(input_block_idx) * BLOCK_M;
        // 每行有num_k个FP8元素，用首行索引乘K定位data基址。
        const auto input = input_layout.data.get_ptr(buffer, input_m_idx * num_k);
        // 每行逻辑scale pair数为num_sf_pairs，后续再根据G1/G2物理主序处理具体K偏移。
        const auto input_sf = input_layout.sf.get_ptr(buffer, input_m_idx * num_sf_pairs);

        // 按M128/M256编译两套计算流水，引用当前task与持续的L1/L0游标。
        const auto compute = [&]<uint32_t MAD_M>() __aicore__ {
            // 当前配置的L1 K段宽度：M128为512，M256为256。
            constexpr uint32_t BLOCK_K = MMAConfig<MAD_M>::BLOCK_K;
            // 当前配置的单次MAD K长度，分别为256或128。
            constexpr uint32_t MAD_K = MMAConfig<MAD_M>::MAD_K;
            // 当前配置的单次MAD N长度，分别为128或256。
            constexpr uint32_t MAD_N = MMAConfig<MAD_M>::MAD_N;
            // 不拆N时借用A Full事件族；拆两次N时用B Full，覆盖相同MTE1队列此前A/B/scale装载。
            constexpr auto l0_full = BLOCK_N == MAD_N ? kL0AFull : kL0BFull;
            // shared的B槽数由B区容量和事件5~7预算共同限制，为2或3。
            constexpr uint32_t kNumSharedL1BStages = MMAConfig<MAD_M>::get_num_shared_l1b_stages();
            // L1A从地址0开始，单槽64KiB，连续保留三份。
            constexpr l1_ptr<float8_e4m3_t, Major::K> l1a(MAD_M, BLOCK_K, 0);
            // routed B紧跟三份L1A，单槽MAD_N*BLOCK_K=64KiB，连续保留四份。
            constexpr l1_ptr<float8_e4m3_t, Major::K> l1b(MAD_N, BLOCK_K, l1a.offset(kNumL1AStages));
            // shared与routed B共用起点，但每槽装完整N256；槽跨度可能是128KiB。
            constexpr l1_ptr<float8_e4m3_t, Major::K> shared_l1b(BLOCK_N, BLOCK_K, l1b.addr);
            // L0A单槽存MAD_M*MAD_K个FP8值；两份32KiB槽由stage索引选择。
            constexpr l0a_ptr<float8_e4m3_t> l0a(MAD_M, MAD_K);
            // L0B单槽存MAD_N*MAD_K个FP8值，独立双缓冲。
            constexpr l0b_ptr<float8_e4m3_t> l0b(MAD_N, MAD_K);
            // L0C存FP32累加，N拆分时用不同l0c_idx保留两半结果，整个K-loop累加到同一N子块。
            constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);

            // 沿完整K依次装载L1 tile；双/三缓冲只交错执行，不改变归约顺序与范围。
            for (uint32_t k_idx = 0; k_idx < num_k; k_idx += BLOCK_K) {
                // valid_k remains a multiple of MAD_K, including the final K block
                // 最后一个L1 K段取剩余长度，受支持shape保证仍能按MAD_K整段消费。
                const uint32_t valid_k = min(BLOCK_K, num_k - k_idx);
                // 将当前K元素偏移换算成scale pair索引。
                const uint32_t sf_pair_idx = k_idx / MX_SF_DIVISOR;
                // 当前scale chunk最多32 pair；K末尾只读取实际剩余pair数。
                const uint32_t num_chunk_pairs = min(kNumSFPairsPerChunk, num_sf_pairs - sf_pair_idx);

                // Load shared FP8 weights
                // 只有shared权重由AIC自己的MTE2从GM加载，routed data等待AIV0生产。
                if (is_shared) {
                    // 先等旧shared B槽的MTE1读取完成，再允许GM->L1覆盖。
                    wait<PIPE_MTE1, PIPE_MTE2>(kSharedL1BEmpty, l1b_stage_idx);
                    // 当前权重N块在shared预排布data中的起始元素偏移。
                    const uint64_t weight_offset = weight_block_idx * BLOCK_N * num_k;
                    // 本次共享权重复制使用1*1循环设置，防止受其他GM->L1循环参数影响。
                    asc_set_gm2l1_loop_size(1, 1);
                    // 把预排好的shared FP8 NZ data搬入当前shared B槽，无需AIV0做FP4展开。
                    asc_copy_gm2l1_align(shared_l1b[l1b_stage_idx].template ptr<uint8_t>(),
                                         // 权重块偏移加k_idx*N256定位当前K段的NZ区域，按字节指针传给搬运API。
                                         const_cast<__gm__ uint8_t*>(task_weights.data + weight_offset + k_idx * BLOCK_N),
                                         // 本段有效数据为BLOCK_N*valid_k字节；数据已是硬件所需NZ布局。
                                         1, BLOCK_N * valid_k, 0, 0, false,
                                         // 多M块共享权重时使用复用提示，否则避免不必要的L2分配。
                                         reuse_shared_weights ? asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP,
                                         // 源/目标循环跨度按完整N256*K块容量设置，尾段不改变缓冲槽的预留跨度。
                                         BLOCK_N * BLOCK_K, BLOCK_N * BLOCK_K);
                    // 通知MTE1 shared B已到齐，后续才可从L1读入L0B。
                    notify<PIPE_MTE2, PIPE_MTE1>(kSharedL1BFull, l1b_stage_idx);
                }

                // scale一槽缓存32个int16 pair，即2048个K元素；M128可覆盖4个K512块，M256覆盖8个K256块。
                // 仅在chunk边界搬A/B scale，最后一次MTE1消费后才归还两组scale共用的SF credit。
                // Load weight scales once per SF chunk
                // 只有新2048-K scale chunk的起点才装载B scale，其他K段复用当前槽。
                if (sf_pair_idx % kNumSFPairsPerChunk == 0) {
                    // 等待前一chunk不再读取这份A/B scale槽，避免被新scale覆盖。
                    wait<PIPE_MTE1, PIPE_MTE2>(kSFEmpty, sf_stage_idx);
                    // 装载B scale到当前L1 SF槽；同一chunk覆盖后续多个BLOCK_K。
                    copy_gm_to_l1_mx(l1_sfb[sf_stage_idx],
                                     // B scale物理以N为主，K-pair偏移每次跨256个int16元素。
                                     gm_ptr<int16_t, Major::MN>(BLOCK_N, reinterpret_cast<uintptr_t>(task_weights.sf + sf_offset + sf_pair_idx * BLOCK_N)),
                                     // 复制本N块的256行与有效pair列，缓存提示独立于权重data的提示。
                                     BLOCK_N, num_chunk_pairs, weight_sf_l2_hint);
                }

                // Prefetch independent weights before waiting for activations
                // 等的是本core已分配task对应的块，不是等32核凑满：Flash块0 ready时core0~15可算，core16~31可同时等块1。
                // AIC在此阻塞，不会跳过未ready任务去偷取其他块；固定G1 lookahead也可能推迟前面已ready块的G2。
                // 首次K读取前等待输入块就绪；shared G1直接读本地输入，无需Dispatched证书。
                if (k_idx == 0 and (not is_shared or not is_linear1))
                    // 阻塞本核当前task，不跳去执行其他已ready任务。
                    comm::wait_for_block<workspace>(buffer, rank_idx,
                        // G1等待Dispatch整理完成，G2等待前一层全部激活列完成。
                        is_linear1 ? layout::kDispatched : layout::kLinear1Ready,
                        // shared用自身块号，routed进度区排在本轮shared块之后，进度编号不取模。
                        (is_shared ? 0 : num_shared_m_blocks) + pool_block_idx,
                        // Dispatch要求32个AIV1贡献；Linear1Ready要求当前中间宽度全部N列任务贡献。
                        is_linear1 ? kNumAICores : num_linear1_n_blocks);

                // Load activations and scales
                // 激活环槽等MTE1最后读取后才可让MTE2覆盖，独立于GM块ready条件。
                wait<PIPE_MTE1, PIPE_MTE2>(kL1AEmpty, l1a_stage_idx);
                // 装载当前task有效M行、当前K段的FP8激活到选定L1A槽。
                copy_gm_to_l1(l1a[l1a_stage_idx],
                              // GM输入为每行num_k个元素的K连续布局，当前指针只在列方向前进k_idx。
                              gm_ptr<float8_e4m3_t, Major::K>(num_k, reinterpret_cast<uintptr_t>(input + k_idx)),
                              // 只读取valid_m*valid_k，尾块不把逻辑256行padding当有效输入读取。
                              valid_m, valid_k, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                // 通知MTE1当前A tile可读，L0装载必须消费此事件。
                notify<PIPE_MTE2, PIPE_MTE1>(kL1AFull, l1a_stage_idx);

                // 激活scale同样只在2048-K chunk边界装载，与B scale共用SF槽号。
                if (sf_pair_idx % kNumSFPairsPerChunk == 0) {
                    // G1输入SF按原token行组织，与G2的K-pair主序不同。
                    if (is_linear1) {
                        // G1 scale的每行逻辑跨度为num_sf_pairs，当前K偏移为sf_pair_idx。
                        copy_gm_to_l1_mx(l1_sfa[sf_stage_idx],
                                         gm_ptr<int16_t, Major::K>(num_sf_pairs, reinterpret_cast<uintptr_t>(input_sf + sf_pair_idx)),
                                         valid_m, num_chunk_pairs, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    } else {
                        // G2 scale来自epilogue写出的[M-block,K-pair,M256]布局，按pair方向跨256行读取。
                        copy_gm_to_l1_mx(l1_sfa[sf_stage_idx],
                                         // 构造Major::MN视图并用sf_pair_idx*256定位scale列，不能套用G1的token-major索引。
                                         gm_ptr<int16_t, Major::MN>(BLOCK_M, reinterpret_cast<uintptr_t>(input_sf + sf_pair_idx * BLOCK_M)),
                                         valid_m, num_chunk_pairs, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    }

                    // One completion covers both scale loads on the same MTE2 queue
                    // A/B scale均排在同一MTE2队列，最后一个full覆盖两次装载的完成边界。
                    notify<PIPE_MTE2, PIPE_MTE1>(kSFFull, sf_stage_idx);
                }

                // Wait for L1 inputs
                // MTE1先等当前A已写入L1，再启动L1->L0，不把GM输入ready等同于本地L1到齐。
                wait<PIPE_MTE2, PIPE_MTE1>(kL1AFull, l1a_stage_idx);
                // shared B使用本核MTE2/MTE1事件，不经过AIV0跨核链路。
                if (is_shared) {
                    // 等待当前shared B tile的GM->L1完成。
                    wait<PIPE_MTE2, PIPE_MTE1>(kSharedL1BFull, l1b_stage_idx);
                } else {
                    // routed按N子块枚举其所占B槽，M128一次L1 K段需要相邻两份N128槽。
                    for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N)
                        // 每个子槽分别等配对AIV0的L1BFull，槽号以4取模，保证C/V消费顺序一致。
                        wait_intra_block<PIPE_MTE1, 0>(kL1BFull, (l1b_stage_idx + n_mad_idx / MAD_N) % kNumL1BStages);
                }

                // Reuse L0A across the BLOCK_N / MAD_N multiplies
                // 一个L1 K段拆成两次MAD_K；每份A在所有N子乘之间复用。
                for (uint32_t k_mad_idx = 0; k_mad_idx < valid_k; k_mad_idx += MAD_K) {
                    // 当前K段内的L0A槽号0/1，和L1A的三槽游标不是同一个索引。
                    const uint32_t l0a_stage_idx = k_mad_idx / MAD_K;
                    // 定位当前MAD所需scale在2048-K chunk内的pair偏移，按32取模。
                    const uint32_t sf_pair_idx_in_chunk = (sf_pair_idx + k_mad_idx / MX_SF_DIVISOR) % kNumSFPairsPerChunk;
                    // 创建按16行对齐的实际A矩阵视图；只改变shape，不改变该L0槽地址。
                    const auto l0a_mad = l0a[l0a_stage_idx].as_mad_aligned(aligned_valid_m, MAD_K);
                    // 等待M管线读完旧L0A，MTE1才可重新装载。
                    wait<PIPE_M, PIPE_MTE1>(kL0AEmpty, l0a_stage_idx);
                    // 从当前L1A槽的k_mad_idx位置搬入MAD_K段，M偏移为0覆盖全部有效行。
                    copy_l1_to_l0a(l0a_mad, l1a[l1a_stage_idx], 0, k_mad_idx);
                    // scale新chunk只在第一个K子段等待一次，随后同chunk的MAD复用已经到齐的SF。
                    if (sf_pair_idx % kNumSFPairsPerChunk == 0 and k_mad_idx == 0)
                        // 消费A/B共同的SF Full，确保下面读取的MX scale不是旧chunk。
                        wait<PIPE_MTE2, PIPE_MTE1>(kSFFull, sf_stage_idx);
                    // 将当前A的MX scale送入对应L0A编码区，K偏移使用pair单位。
                    copy_l1_to_l0a_mx(l0a_mad, l1_sfa[sf_stage_idx], 0, sf_pair_idx_in_chunk, MAD_K);
                    // 判断是否为整个task最后一个K子段，以决定累加完成后是否允许FIX输出。
                    const bool is_last_k = k_idx + k_mad_idx + MAD_K == num_k;

                    // Keep B reuse credits independent across MAD shapes
                    // M128拆两次N128，M256做一次N256；这个循环没有重新装载A data。
                    for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N) {
                        // 每个N子块对应一个独立的L0C结果区域，跨K段累加不能混用两半。
                        const uint32_t l0c_idx = n_mad_idx / MAD_N;
                        // B槽按K子段和N子块共同轮转，避免N拆分时覆盖仍被MAD读取的旧B。
                        const uint32_t l0b_stage_idx = (l0a_stage_idx * (BLOCK_N / MAD_N) + l0c_idx) % kNumL0Stages;
                        // MTE1等待当前L0B槽空闲；A槽空闲不代表B槽也已读完。
                        wait<PIPE_M, PIPE_MTE1>(kL0BEmpty, l0b_stage_idx);
                        // shared从一个完整N256槽取子列；routed的N子列已分别放在相邻L1B槽。
                        const auto l1b_src = is_shared ? shared_l1b[l1b_stage_idx] : l1b[(l1b_stage_idx + n_mad_idx / MAD_N) % kNumL1BStages];
                        // 将本次MAD的权重送入L0B；shared需要额外n_mad_idx偏移，routed子槽从列0开始。
                        copy_l1_to_l0b(l0b[l0b_stage_idx], l1b_src, is_shared ? n_mad_idx : 0, k_mad_idx);
                        // 装入权重MX scale，使用同一N偏移与K-pair偏移匹配当前B data。
                        copy_l1_to_l0b_mx(l0b[l0b_stage_idx], l1_sfb[sf_stage_idx], n_mad_idx, sf_pair_idx_in_chunk, MAD_K);
                        // 通知M管线当前L0数据与scale到齐；同队列顺序也覆盖此前A装载。
                        notify<PIPE_MTE1, PIPE_M>(l0_full, l0b_stage_idx);

                        // B readiness also covers the earlier A/scales loads on the same MTE1 queue
                        // L0C结果视图使用实际对齐M和当前MAD_N，保留独立N子块的基址。
                        const auto l0c_mad = l0c[l0c_idx].as_mad_aligned(aligned_valid_m, MAD_N);
                        // M管线等待对应Full事件，禁止在MTE1未完成时读取L0A/B。
                        wait<PIPE_MTE1, PIPE_M>(l0_full, l0b_stage_idx);
                        // 提交当前M/N/K子乘，结果在同一个N子块的FP32 L0C中跨K累加。
                        mad(l0c_mad, l0a_mad, l0b[l0b_stage_idx], aligned_valid_m, MAD_N, MAD_K,
                            // 只有最后K子段更新unit flag，FIX据此知道完整归约何时可读取。
                            is_last_k ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
                            // 首个K子段初始化累加结果，后续K子段保留并累加已有L0C。
                            k_idx + k_mad_idx == 0);
                        // 同一A只有在最后一次N子乘读完后才能回收。
                        if (n_mad_idx + MAD_N == BLOCK_N)
                            // 向MTE1归还这份L0A槽，允许下一轮覆盖。
                            notify<PIPE_M, PIPE_MTE1>(kL0AEmpty, l0a_stage_idx);
                        // 本次MAD已消费当前B，立即归还对应L0B槽；不必等整个逻辑N256结束。
                        notify<PIPE_M, PIPE_MTE1>(kL0BEmpty, l0b_stage_idx);
                        // 不是最后K段时只有部分和，不能执行下面的FIX写出。
                        if (not is_last_k)
                            // 转到下个N/K子乘继续累加，保留L0C内容。
                            continue;

                        // Reuse output rows after all sends complete
                        // 仅routed G2、首次N子块且GM环发生回绕时，检查旧输出是否已经被发送消费。
                        if (block_phase == sched::BlockPhase::kLinear2 and n_mad_idx == 0 and pool_block_idx >= kNumRingBlocks)
                            // 输出环覆盖约束与输入环不同：Linear2 FIX 写块 b 前，等旧块 b-ringBlocks 的所有 AIV1 完成远端发送。
                            // CombineSent 在 MTE3 stores 排空后发布，因此放行同时证明旧输出已被读走且发送使用的源内容不再依赖它。
                            // 只等 Linear2Ready 不够：它仅证明旧 GMM2 生产结束，不能证明 Combine 已消费旧输出。
                            // 此等待出现在最后 K 的 FIX 写出前，前面的计算可先推进，从而把空间反压尽量后移。
                            // 等旧逻辑块所有AIV1完成push，允许覆盖同一物理output环槽；进度索引仍用旧逻辑块号。
                            comm::wait_for_block<workspace>(buffer, rank_idx, layout::kCombineSent,
                                                            num_shared_m_blocks + pool_block_idx - kNumRingBlocks, kNumAICores);

                        // One epilogue credit covers BLOCK_N columns
                        // G1按M128分FIX，G2一次写完整valid_m到GM，二者共享epilogue握手但目的空间不同。
                        const uint32_t fix_step_m = is_linear1 ? STORE_BLOCK_M : valid_m;
                        // 遍历当前N子块需要写出的M片段，尾段按剩余有效行截断。
                        for (uint32_t m_idx_in_block = 0; m_idx_in_block < valid_m; m_idx_in_block += fix_step_m) {
                            // M128片段对应UB stage0/1；G2的整块写出从m=0开始，只用stage0。
                            const uint32_t ub_stage_idx = m_idx_in_block / STORE_BLOCK_M;
                            // 同一M片段的两个N128子列共用一次empty credit，不能每个N子列重复等待。
                            if (n_mad_idx == 0)
                                // FIX等待AIV1归还目标槽，防止覆盖尚在量化或MTE3写出的UB结果。
                                wait_intra_block<PIPE_FIX, 1>(kEpilogueEmpty, ub_stage_idx);
                            // G1输出交给AIV1做激活，G2输出直接落GM供Combine读取。
                            if (is_linear1) {
                                // Store MAD_N gate/up columns with a BLOCK_N destination stride
                                // 构造同一个M128*N256 UB槽中的当前N子列视图，地址加n_mad_idx*2字节。
                                const ub_ptr<bfloat16_t> ub_dst(STORE_BLOCK_M, BLOCK_N,
                                                                ub_accum[ub_stage_idx].addr + n_mad_idx * sizeof(bfloat16_t));
                                // 从L0C的NZ布局按M片段偏移写入UB，保持目的行跨度为物理N256。
                                copy_l0c_to_ub(ub_dst, l0c_mad.offset_nz_m(m_idx_in_block),
                                               // 仅写本片段有效行数，不向尾部padding发布有效结果。
                                               min(STORE_BLOCK_M, valid_m - m_idx_in_block),
                                               // 列数为当前MAD_N，M128配置需等第二个N128也写完才通知Full。
                                               MAD_N, asc_dual_dst_mode::DUAL_DST_DISABLE, 1);
                            } else {
                                // Store shared output to combine; routed output to the ring
                                // shared直接写本rank的shared接收槽；routed写本rank可回收的输出环。
                                const auto output_base = is_shared ? combine_input : routed_output;
                                // shared使用原token行与shared slot定位，routed使用环形物理M起点。
                                const uint64_t output_m_idx = is_shared ?
                                    // shared slot编号等于num_topk，不占用任何routed top-k槽。
                                    workspace.get_combine_m_idx(config.num_topk, static_cast<uint64_t>(pool_block_idx) * BLOCK_M) :
                                    // routed仅data地址按R取模，源身份与完成计数仍保留逻辑块号。
                                    layout::get_ring_m_idx<workspace>(pool_block_idx);
                                // 加上完整H行跨度、任务N块偏移和N子块偏移，得到当前输出子矩阵的GM地址。
                                const auto l2_output = output_base + output_m_idx * config.hidden + n_block_idx * BLOCK_N + n_mad_idx;
                                // FIX写BF16的valid_m*MAD_N输出，行跨度保持H，采用共享写回缓存策略。
                                copy_l0c_to_gm(l2_output, l0c_mad, valid_m, MAD_N, config.hidden, kWriteThroughShare);
                            }
                            // 本M片段所有N子列都已提交FIX后才发Full，避免AIV1读到半份gate/up。
                            if (n_mad_idx + MAD_N == BLOCK_N)
                                // 通知配对AIV1当前stage的结果完成边界，AIV1仍需消费此事件才能处理/发布进度。
                                notify_intra_block<PIPE_FIX, 1>(kEpilogueFull, ub_stage_idx);
                        }
                    }
                }

                // Release L1 after the last L0 load
                // 所有L0装载已提交后，按权重生产者类型归还对应L1B空间。
                if (is_shared) {
                    // shared B的最后一次MTE1读取完成后允许本核MTE2覆盖。
                    notify<PIPE_MTE1, PIPE_MTE2>(kSharedL1BEmpty, l1b_stage_idx);
                    // shared按配置的2/3槽推进，不能按routed四槽跨度索引。
                    l1b_stage_idx = (l1b_stage_idx + 1) % kNumSharedL1BStages;
                } else {
                    // routed逐一释放当前N256占用的N子槽。
                    for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N)
                        // 把B空槽通过AIV0链路归还权重生产者，保护UB->L1写入时不覆盖旧读取。
                        notify_intra_block<PIPE_MTE1, 0>(kL1BEmpty, (l1b_stage_idx + n_mad_idx / MAD_N) % kNumL1BStages);
                    // 本段N占用几槽就前进几槽，M128前进2、M256前进1，再按4取模。
                    l1b_stage_idx = (l1b_stage_idx + (BLOCK_N / MAD_N)) % kNumL1BStages;
                }
                // 当前L1A不再被MTE1读取，归还激活生产者MTE2。
                notify<PIPE_MTE1, PIPE_MTE2>(kL1AEmpty, l1a_stage_idx);
                // 三槽激活环前进一格，独立于两槽L0索引。
                l1a_stage_idx = (l1a_stage_idx + 1) % kNumL1AStages;

                // 整个2048-K SF chunk用完或task已到K尾部时才能归还scale槽。
                if ((sf_pair_idx + BLOCK_K / MX_SF_DIVISOR) % kNumSFPairsPerChunk == 0 or k_idx + valid_k == num_k) {
                    // A/B scale最后一次MTE1读取结束后统一释放对应SF槽。
                    notify<PIPE_MTE1, PIPE_MTE2>(kSFEmpty, sf_stage_idx);
                    // 切换另一份SF缓冲，下次chunk装载先等待它的empty。
                    sf_stage_idx = (sf_stage_idx + 1) % kNumSFStages;
                }
            }

        };
        // Shared tasks keep one shape so their L1B stages retain the same physical stride
        // shared以整轮T选固定配置，避免尾块切shape改变其B槽物理跨度；routed逐块按valid_m选择。
        if ((is_shared ? num_tokens : valid_m) <= 128)
            // 小M实例使用K512/N128分块，单task内部仍覆盖逻辑N256。
            compute.template operator()<128>();
        else
            // 大M实例使用K256/N256分块，实际MAD的M仍是Align16(valid_m)。
            compute.template operator()<256>();

        // issued仅表明当前task的FIX指令已提交，不证明FIX数据完成；AIV1看到变化后仍须等EpilogueFull。
        // AIV1先探测GM计数可避免过早堵在未提交任务的CV事件上，从而保留dispatch/push推进机会。
        // 计数采用uint8回绕，配合epilogue空间反压限制在途task；不能把它当整轮累计任务总数。
        // 在当前task的FIX指令提交之后更新uint8计数；AIV1看到变化仍要等待实际FIX Full事件。
        asc_store_dev(num_issued_gemms_ptr, ++ num_issued_gemms);
    }
    // 存在shared输出时，在本核退出任务流水前设置GM内存顺序边界。
    if constexpr (kHasSharedExperts)
        // 保证本核此前共享GM相关操作的顺序可见性，不替代两侧AIV最终输出握手。
        asc_sync_data_barrier(DSB_DDR);
}

// AIV1: epilogue, dispatch, and combine
// AIV1 同时推进输入拉取、GMM1 激活/量化、GMM2 结果回传，以及可选的输出归并。
// progress 记录逻辑 M-block 的完成贡献，数据槽则环形复用；发布 ready 必须排在数据写回之后。
template <const auto& workspace, bool kOverlapReduce>
// @param buffer 本rank对称GM；output为最终BF16[T,H]，中途仅重叠reduce使用。
// @param num_tokens 本rank有效T；rank_idx本rank编号；core_idx配对AIC号[0,32)。
// @param[in,out] reduce 此AIV1的最终输出归并状态，与Dispatch/epilogue/Combine各自游标独立。
// @param activation_clamp SwiGLU的裁剪界限：gate只限上界，up限制在[-clamp,+clamp]。
// @param sym_buffer 跨rank地址映射，pull读取来源输入，push写回来源的top-k槽。
__aicore__ inline void aiv1_epilogue_comm(__gm__ uint8_t* buffer, __gm__ bfloat16_t* output,
                                          // T为本rank原输入行数；core_idx为配对AIC号0~31，不是全局AIV号。
                                          uint32_t num_tokens, uint32_t rank_idx,
                                          // reduce保存本AIV1已完成的原token/channel进度，和pull/push块游标独立。
                                          CombineReduceState<workspace>& reduce,
                                          // clamp只影响向量激活；sym_buffer用于当前行的peer基址映射。
                                          float activation_clamp, uint32_t core_idx, const layout::SymBuffer<>& sym_buffer) {
    // 引用本实例shape；同一个AIV1同时保存pull、epilogue、push和reduce的独立进度。
    constexpr auto& config = workspace.config;
    // routed G1每N256给出gate128/up128，因此完整I需要I/128个task。
    constexpr uint32_t kNumLinear1NBlocks = config.intermediate_hidden / (BLOCK_N / 2);
    // G2每task输出256个最终通道，完整H需要H/256个task。
    constexpr uint32_t kNumLinear2NBlocks = config.hidden / BLOCK_N;
    // shared中间宽度为S*I，列任务数按合并后的宽度/128计算。
    constexpr uint32_t kNumSharedLinear1NBlocks = config.shared_intermediate_hidden / (BLOCK_N / 2);
    // 取得数据环容量R，输入/中间激活/输出的物理位置按R复用，身份与进度不取模。
    constexpr uint32_t kNumRingBlocks = workspace.get_num_ring_blocks();
    // 编译期决定是否调度shared epilogue与等待shared完成，shared不参与routed pull/push。
    constexpr bool kHasSharedExperts = config.num_shared_experts > 0;

    // AIV1 UB 峰值按地址顺序累加而非逐任务分配：
    //   FIX accum : 2*(128*256*2) 字节=128KiB，两个 M128 输出槽；
    //   route weights : 128*4 字节=512B；output SF: 2*(2*128*2) 字节=1KiB；
    //   pull : 两槽 H 字节 FP8 token + 两槽 Align32(ceil(H/64)*2) 的 scale；
    //   source begin/end : 两个 worldSize*4 字节的 prefix，边界按32B对齐；
    //   push : 四槽 H*2 字节 BF16 行；末尾剩余容量分成两个 reduce 槽并按整向量通道数取整。
    // accum 的 BF16 投影在槽内原地量化成较小 FP8 输出，因此无需再追加完整激活矩阵；MTE3 写出结束才能归还 FIX credit。
    // pull/push/reduce 事件 ID 和空间分别管理，空间足够并不意味着允许同槽绕过 empty 等待。
    // 每行SwiGLU输出128通道，每64通道一个int16 SF pair，所以每行2 pair；每pair打包两个UE8M0指数。
    constexpr uint32_t kNumOutputSFPairs = 2;
    // UB分区起点的32字节对齐要求；不是GM的512字节布局对齐，也不表示一次通信搬32字节。
    constexpr uint32_t kNumUBAlignmentBytes = 32;

    // AIV1 UB in address order (bytes); H=hidden, R=num_ranks
    // Region        Buffers  Bytes/buffer
    // FIX accum           2  STORE_BLOCK_M * BLOCK_N * 2
    // Route weights       1  STORE_BLOCK_M * 4
    // Output SF           2  STORE_BLOCK_M * 4
    // Pull tokens         2  H
    // Pull SF             2  align(H/32, 32)
    // Source begin/end    2  R * 4
    // Push tokens         4  H * 2
    // Reduce              2  Fits remaining UB, rounded down to whole vectors
    // Source begin and push tokens start at 32-byte boundaries
    // 两份FIX结果槽从UB0开始，每份BF16[128,256]为64KiB，与AIC的ub_accum地址一致。
    constexpr ub_ptr<bfloat16_t> ub_accum(STORE_BLOCK_M, BLOCK_N);
    // 唯一的128行FP32路由权重槽紧跟两份结果槽，避免另分一整块token权重矩阵。
    constexpr ub_ptr<float> ub_route_weights(1, STORE_BLOCK_M, ub_accum.offset(kNumEpilogueStages));
    // 量化SF两槽位于route_weights之后，每槽按[pair=2,M=128]保存int16数据。
    constexpr ub_ptr<int16_t> ub_output_sf(kNumOutputSFPairs, STORE_BLOCK_M, ub_route_weights.offset(1));

    // Dispatch inputs and source ranges
    // 两份单行FP8输入槽跟在SF区后，H同时是每个pull token槽的字节数。
    constexpr ub_ptr<float8_e4m3_t> ub_pull_token(1, config.hidden, ub_output_sf.offset(kNumEpilogueStages));
    // 每行scale按32B向上对齐预留；除sizeof(int16_t)把字节容量换回pair元素数。
    constexpr ub_ptr<int16_t> ub_pull_sf(1, aligned(config.num_input_sf_pairs * sizeof(int16_t), kNumUBAlignmentBytes) / sizeof(int16_t),
                                         // scale区起点位于两个H字节token槽之后，不能只跳过一份pull token。
                                         ub_pull_token.offset(kNumPullStages));
    // 每个source的累计起点数组按32B对齐，放在两份pull SF槽之后。
    constexpr ub_ptr<uint32_t> ub_src_m_begin(1, config.num_ranks, aligned(ub_pull_sf.offset(kNumPullStages), kNumUBAlignmentBytes));
    // 累计终点数组紧跟起点数组，用来将置换后的专家行号映射回source rank区间。
    constexpr ub_ptr<uint32_t> ub_src_m_end(1, config.num_ranks, ub_src_m_begin.offset(1));

    // Combine sends and reduction
    // 四份BF16回传行槽从对齐后的source数组末尾开始，每槽H*2字节。
    constexpr ub_ptr<bfloat16_t> ub_push_token(1, config.hidden, aligned(ub_src_m_end.offset(1), kNumUBAlignmentBytes));
    // 用push四槽之后的剩余UB建立reduce小chunk；不能覆盖仍在使用的通信或FIX空间。
    constexpr auto ub_reduce_inputs = make_reduce_buffer<workspace>(ub_push_token.offset(kNumPushStages));

    // Size the reduce window from local output tokens
    // 归并窗口按本rank原token数T换算成块，至少1块使空输入rank也可进入尾部前缀发布路径。
    const uint32_t num_reduce_window_m_blocks = max(ceil_div(num_tokens, BLOCK_M), 1u);
    // shared逻辑块数用于任务调度及progress分区偏移，routed进度起点随本轮T变化。
    const uint32_t num_shared_m_blocks = kHasSharedExperts ? ceil_div(num_tokens, BLOCK_M) : 0;

    // 编译期检查push区不越UB且reduce至少可容纳有效向量宽度，失败时阻止错误布局生成。
    static_assert(ub_push_token.offset(kNumPushStages) <= UBSizeBytes and ub_reduce_inputs.shape_n > 0, "AIV1 UB overflow");

    // 向配对AIC发放两个初始FIX空槽；此后只能在本stage消费/写出完成后归还。
    notify_intra_block<PIPE_MTE3, 1, kNumEpilogueStages>(kEpilogueEmpty);
    // 四个push UB槽初始可写，允许MTE2开始装载第一批BF16输出行。
    notify<PIPE_MTE3, PIPE_MTE2, kNumPushStages>(kPushEmpty);
    // 唯一top-k权重槽初始可写，MTE2可为首个G1 stage装入权重。
    notify<PIPE_V, PIPE_MTE2>(kTopkEmpty);
    // 两个pull UB槽初始可写，允许从源rank拉取FP8 token和scale。
    notify<PIPE_MTE3, PIPE_MTE2, kNumPullStages>(kPullEmpty);

    // 本rank按完整逻辑pool行保留的top-k权重，供后续G1 epilogue逐行读取。
    const auto routed_weights = workspace.routed.topk_weights.get_ptr(buffer);
    // 本rank按完整逻辑pool行保存的返回地址metadata，编码源rank及top-k接收行。
    const auto src_metadata = workspace.routed.src_metadata.get_ptr(buffer);
    // 接收metadata阶段已按[local expert,source rank,slot]写好的uint64路由索引/权重表。
    const auto src_token_metadata = workspace.metadata.src_token_metadata.get_ptr(buffer);

    // routed总块数暂未知，shared块不计入此变量；待metadata_ready后从证书加载。
    uint32_t num_total_m_blocks = 0;  // Routed pool only; shared blocks are separate
    // 暂不能执行routed通信，但调度器仍可推进shared epilogue。
    bool metadata_ready = false;
    // 指向总块数证书，读低24位B之前必须先观察本核metadata_ready。
    const auto num_m_blocks_certificate = workspace.metadata.num_m_blocks_certificate.get_ptr(buffer);

    // 四类routed计数的单位不同：Dispatched/CombineSent每个AIV1贡献1次，目标32；
    // Linear1Ready每个N256投影task贡献1次，目标I/128；Linear2Ready目标H/256。
    // 这里atomic_add只更新完成证书；输出数值通过DMA写回及后续FP32 reduce生成。
    // 把发布数据完成状态封装成局部操作；slot_begin是逻辑进度槽起点，num_slots是连续块数。
    const auto publish_block_progress = [&](layout::ProgressStage stage,
                                            // 每个slot独立加1贡献，允许一次stores排空后批量发布多个块。
                                            uint32_t slot_begin, uint32_t num_slots) __aicore__ {
        // Wait for stores; dispatch and FIX callers already waited
        // G1激活与Combine回传由MTE3写出，发布对应ready前必须额外等这些stores完成。
        if (stage == layout::kLinear1Ready or stage == layout::kCombineSent) {
            // 在MTE3队列尾发出完成通知，覆盖此前已经提交的本核写出。
            notify<PIPE_MTE3, PIPE_S>(kStoresDone);
            // Scalar等到通知，再允许执行下面的原子完成计数。
            wait<PIPE_MTE3, PIPE_S>(kStoresDone);
        }

        // Publish this core's contribution after its Scalar metadata is visible
        // 先确保Scalar路由metadata等GM写入的顺序可见，再发布消费者会读取的完成证书。
        asc_sync_data_barrier(DSB_DDR);
        // 可以一次排空stores后，为多个连续逻辑块逐一贡献计数；并未合并它们的数据缓冲。
        for (uint32_t slot_offset = 0; slot_offset < num_slots; ++ slot_offset)
            // get_progress_idx定位阶段，get_ptr再按每计数独占sector的stride寻址；每个块各加1而非加token数。
            asc_atomic_add(workspace.progress.blocks.get_ptr(buffer, workspace.get_progress_idx(stage, slot_begin) + slot_offset), 1u);
    };

    // 配对AIC的uint8任务提交计数，epilogue据此避免等待一个AIC尚未提交的FIX。
    const auto num_issued_gemms_ptr = workspace.progress.num_issued_gemms.get_ptr(buffer, core_idx);

    // routed G1输入FP8 data环基址；每个目标块的物理首行由ring_m_idx选择。
    const auto l1_acts = workspace.routed.l1_acts.data.get_ptr(buffer);
    // routed G1输入scale环基址，行跨度为H/64个int16 pair。
    const auto l1_acts_sf = workspace.routed.l1_acts.sf.get_ptr(buffer);
    // Dispatch状态机：块/行游标、专家游标和source置换游标都跨advance调用保存。
    struct DispatchPullState {
        // 下一个待整理的逻辑M-block号，从0单调推进，不按环容量回绕这个身份索引。
        uint32_t pool_block_idx = 0;
        // 本核下一条目标行；值256作为pending-stores标记，下一次推进才发布块ready。
        uint32_t m_idx_in_block;  // BLOCK_M marks pending stores
        // 当前块的实际有效行数，由expert接收行数和16行组均分规则决定。
        uint32_t valid_m;
        // 按块顺序寻找当前本地expert，自动越过没有任何接收路由的专家。
        sched::ExpertCursor<workspace> expert;

        // Permute the concatenated source rows without changing their counts
        // src_m_idx为专家内source拼接序列的当前ordinal；stride为置换步长，step为同核跨32行的步长。
        uint32_t src_m_idx = 0, src_m_stride = 0, src_m_step = 0;
        // 缓存当前source累计区间所在rank，置换回绕或切专家时需从0重找。
        uint32_t src_rank_idx = 0;
    };
    // 每核从自身core_idx行开始，32个AIV1覆盖互不重叠的目标行序列。
    DispatchPullState pull{.m_idx_in_block = core_idx};

    // Pull one batch of input rows
    // 输入 GM 环的覆盖条件：逻辑块 b>=ringBlocks 时，先等 b-ringBlocks 的所有 Linear1 N-block 激活完成。
    // Linear1Ready 证明旧 L1 输入已不再被 GMM1 使用，然后才能向同一个物理 ring 槽放新 token/scale。
    // source 行用与专家行数互质的 stride 置换，遍历仍覆盖每个源 ordinal 一次，目标是分散连续读同一 peer 的压力。
    // ordinal代表一条(token,top-k slot)路由，并未按(token,dst rank)去重；同token命中同rank的多个专家仍逐路由拉取。
    // 置换的“不重不漏”只指每条路由被遍历一次，不代表相同token只搬一次；全部32个AIV1共同分摊这些行。
    // 身份元数据和权重按 pool_m_idx 全量保留，数据按 ring_m_idx 复用，两个索引不可互换。
    // pull 每步最多两个 token，末步只置 pending-store 标记；下一步 wait kPullDone 后才发布 Dispatched。
    // 满块每AIV1拉8行就发布该块ready，直接放行GMM1；若也攒4块才发布，会延迟AIC获得输入。
    // 无有效行的 owner 也贡献一次 Dispatched，否则 AIC 等 kNumAICores 个贡献时无法满足目标。
    // 一次最多推进两个输入行，或完成上次pending块的排空/发布；不是整expert一次性通信。
    const auto advance_dispatch_pull = [&]() __aicore__ {
        // 本核已推进到本轮routed块末尾，无需继续扫描source或占用pull槽。
        if (pull.pool_block_idx == num_total_m_blocks)
            // 将控制权返回主循环，epilogue/push仍可能有剩余工作。
            return;

        // 上一轮已经提交本核该块全部输入搬运，当前只需确认stores并发布完成贡献。
        if (pull.m_idx_in_block >= BLOCK_M) {
            // Wait for the previous block's stores before publishing readiness
            // 等待上轮kPullDone，确保消费者看到Dispatched时data与scale已实际写到本地GM。
            wait<PIPE_MTE3, PIPE_S>(kPullDone);

            // Empty row owners must also contribute
            // 为shared区之后的当前routed块加1；无有效行的owner也要贡献，目标总数仍为32。
            publish_block_progress(layout::kDispatched, num_shared_m_blocks + pull.pool_block_idx, 1);
            // 逻辑块前进一格；目标数据环地址稍后另行取模，进度本身不回绕。
            ++ pull.pool_block_idx;
            // 下一块重新从本核负责的首行开始，不沿用上一块已越界的行号。
            pull.m_idx_in_block = core_idx;
            // 发布操作也算本次小步推进，到下一轮再装载新块以便其他阶段获得执行机会。
            return;
        }

        // 仅第一次进入新块时检查环空间、寻找expert并建立source游标。
        if (pull.m_idx_in_block == core_idx) {
            // Reuse input rows after all Linear1 N-blocks finish
            // 逻辑块超出环容量时，同物理槽仍可能保存旧G1输入，必须先确认旧块已消费。
            if (pull.pool_block_idx >= kNumRingBlocks and
                    // 探测旧块Linear1Ready，未ready则返回主循环而不是Scalar死等，给G1/通信继续推进机会。
                    not comm::is_block_ready<workspace>(buffer, layout::kLinear1Ready,
                                                        // 旧块身份为b-R；加shared偏移后得到progress槽，不能用b%R定位完成计数。
                                                        num_shared_m_blocks + pull.pool_block_idx - kNumRingBlocks,
                                                        // 等旧块全部I/128个G1 epilogue任务完成，它们都已经消费旧输入。
                                                        kNumLinear1NBlocks))
                // 环空间暂不可用，当前pull状态保持不变，后续迭代重试。
                return;

            // Load source counts and advance the expert
            // 块号超过当前expert的右开块区间时，前进到下一expert；空expert不占任何逻辑块。
            while (pull.pool_block_idx >= pull.expert.pool_block_end) {
                // 专家游标初值为UINT32_MAX，第一次无符号加1回到本地expert0。
                const uint32_t next_expert_idx = pull.expert.local_expert_idx + 1;
                // 从0累计当前expert的来源行数，并统计有非零贡献的source数量。
                uint32_t num_expert_tokens = 0, num_sources = 0;
                // 按rank顺序读取当前expert的接收计数，构建source区间前缀。
                for (uint32_t src_rank_idx = 0; src_rank_idx < config.num_ranks; ++ src_rank_idx) {
                    // 读的是一个source给这个本地expert的有效路由数，metadata ready已证明全部count到达。
                    const uint32_t count = asc_load_dev(workspace.metadata.expert_recv_count.get_ptr(
                        buffer, workspace.get_expert_recv_idx(src_rank_idx, next_expert_idx)));
                    // 记录这个source区间的左端点，单位为expert内部拼接后的路由ordinal。
                    ub_src_m_begin.ptr()[src_rank_idx] = num_expert_tokens;
                    // 累加该source的有效行数，得到下一个source的起点。
                    num_expert_tokens += count;
                    // 记录当前source的右开终点，之后通过while定位ordinal所属区间。
                    ub_src_m_end.ptr()[src_rank_idx] = num_expert_tokens;
                    // 非零source数决定初始置换步长的尺度；不代表最终每个source拥有相同行数。
                    num_sources += count != 0;
                }
                // 推进专家游标并增加ceil(expert_rows/256)个逻辑块，source身份仍按专家内ordinal保存。
                pull.expert.next_expert(num_expert_tokens);
                // 切换expert后不能沿用旧source区间位置，从rank0重新找。
                pull.src_rank_idx = 0;
                // 只有非空expert才需要做模运算，避免除0并跳过无数据专家。
                if (num_expert_tokens != 0) {
                    // 行号映射 j->j*stride mod expert_rows；stride与总行数互质才能遍历全部源行而不重不漏。
                    // 同一core取 j=expert_m_begin+core_idx+32*k，故其源游标每步加32*stride mod expert_rows。
                    // 从平均非零source段长附近选与总行数互质的步长，置换访问以分散连续peer读取。
                    pull.src_m_stride = next_coprime(ceil_div(num_expert_tokens, num_sources), num_expert_tokens) % num_expert_tokens;
                    // 同核目标行每次加32，所以source ordinal每次加32*stride并对expert行数取模。
                    pull.src_m_step = kNumAICores * pull.src_m_stride % num_expert_tokens;
                }
            }
            // 接收当前块在expert有效行序列中的左端点，单位行，与全局pool物理步长不同。
            uint32_t expert_m_begin;
            // 确定此块valid_m和expert_m_begin；逻辑槽仍为256行，即使此块不足256行。
            pull.valid_m = pull.expert.get_m_block_range(pull.pool_block_idx, expert_m_begin);
            // 把本核首个expert行号经过同一置换映射到source拼接序列，使用uint64进行乘法。
            pull.src_m_idx = static_cast<uint64_t>(expert_m_begin + core_idx) * pull.src_m_stride % pull.expert.num_expert_tokens;
            // 新ordinal落在缓存source区间之前时，从rank0重找；之后的while只向前扫描。
            if (pull.src_m_idx < ub_src_m_begin.ptr()[pull.src_rank_idx])
                // 清除过大的source缓存位置，防止错过前面的有效来源区间。
                pull.src_rank_idx = 0;
        }

        // 每次最多两条有效行，行号达到valid_m后停止，不把padding当成真实路由发送。
        for (uint32_t step_idx = 0; step_idx < kNumPullStages and pull.m_idx_in_block < pull.valid_m; ++ step_idx) {
            // 一个step搬一条路由的完整H维输入与scale；Flash H4096为4096B FP8+128B SF，不是256行DMA。
            // 保存本次正在搬的目标行；下方先推进成员游标不会改变这个局部行号。
            const uint32_t m_idx_in_block = pull.m_idx_in_block;
            // 下次取本核下一条行，跨过其他31个AIV1负责的位置。
            pull.m_idx_in_block += kNumAICores;
            // 按本核第几条行对2取模，事件和UB都使用这个pull槽号。
            const uint32_t stage_idx = m_idx_in_block / kNumAICores % kNumPullStages;
            // 完整逻辑pool行用于保存身份/权重，必须使用uint64乘法且不按R取模。
            const uint64_t pool_m_idx = static_cast<uint64_t>(pull.pool_block_idx) * BLOCK_M + m_idx_in_block;
            // 数据写入环的物理行号，按块取模R后再加块内行。
            const uint64_t ring_m_idx = layout::get_ring_m_idx<workspace>(pull.pool_block_idx, m_idx_in_block);
            // 跳过当前ordinal已经越过的source区间，也会跳过count=0的空区间。
            while (pull.src_m_idx >= ub_src_m_end.ptr()[pull.src_rank_idx])
                // 向后扫描到包含当前source ordinal的rank。
                ++ pull.src_rank_idx;
            // 保存本次来源rank；后面提前推进pull.src_rank_idx不会改变此次传输的来源。
            const uint32_t src_rank_idx = pull.src_rank_idx;
            // 减去区间起点，得到这个source发给当前expert的紧凑metadata槽号。
            const uint32_t src_slot_idx = pull.src_m_idx - ub_src_m_begin.ptr()[src_rank_idx];

            // 为下一条路由预先推进置换ordinal，本次source/slot已经保存在局部变量中。
            pull.src_m_idx += pull.src_m_step;
            // 模步长小于总行数，所以相加后至多回绕一次。
            if (pull.src_m_idx >= pull.expert.num_expert_tokens) {
                // 回绕到expert source序列的开头范围，不能把跨expert的ordinal混在一起。
                pull.src_m_idx -= pull.expert.num_expert_tokens;
                // 回绕后source区间也从rank0重新定位。
                pull.src_rank_idx = 0;
            }
            // 根据source rank把本地对称buffer基址映射为对应peer基址，本地路由也走同一索引路径。
            const auto remote_base = sym_buffer.map(buffer, src_rank_idx);

            // Resolve the source token
            // MTE2等待旧MTE3不再读这个pull UB槽，再装入新token/scale。
            wait<PIPE_MTE3, PIPE_MTE2>(kPullEmpty, stage_idx);
            // 从本地已接收路由表读uint64条目，其内容是高32位权重bits、低32位token-top-k展平索引。
            const uint64_t token_metadata = asc_load_dev(src_token_metadata +
                // metadata地址携带local expert、source rank和source槽号，因此条目内无需重复存expert/rank。
                workspace.get_src_metadata_idx(pull.expert.local_expert_idx, src_rank_idx, src_slot_idx));
            // 截取低32位，取回原source的token*Ktop+slot。
            const uint32_t token_topk_idx = static_cast<uint32_t>(token_metadata);
            // 除top-k槽数还原source rank内的原token行号。
            const uint32_t src_token_idx = token_topk_idx / config.num_topk;
            // 取余还原原token的top-k槽，后续回传必须写回同一槽而非直接覆盖最终output。
            const uint32_t src_topk_idx = token_topk_idx % config.num_topk;
            // 源FP8 token按原token*H定位；同token多条路由可能重复拉取相同输入，未显式去重。
            const auto remote_token = workspace.input.acts.data.get_ptr(remote_base, static_cast<uint64_t>(src_token_idx) * config.hidden);
            // 源scale按每行H/64个int16 pair定位，与FP8 data字节跨度不同。
            const auto remote_sf = workspace.input.acts.sf.get_ptr(remote_base, static_cast<uint64_t>(src_token_idx) * config.num_input_sf_pairs);

            // Pull token data and scales
            // 从peer读一整行FP8 token到当前UB槽，Flash H4096对应4096B。
            copy_gm_to_ub(ub_pull_token[stage_idx].ptr(), remote_token,
                          // 单行H个FP8元素且GM/UB有效行均为H字节，缓存提示只是策略不保证命中。
                          1, config.hidden, config.hidden, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            // 单独拉取该token的scale，不能假设scale紧跟token连续存储。
            copy_gm_to_ub(ub_pull_sf[stage_idx].ptr(), remote_sf, 1, config.num_input_sf_pairs,
                          // scale传输长度是pair数*2字节，Flash对应128B。
                          config.num_input_sf_pairs * sizeof(int16_t), asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
            // 通知MTE3本槽token和scale两次MTE2装载均已完成。
            notify<PIPE_MTE2, PIPE_MTE3>(kPullFull, stage_idx);

            // Stage routed inputs
            // MTE3等待当前pull槽Full，保证写进l1_acts的是本次peer输入。
            wait<PIPE_MTE2, PIPE_MTE3>(kPullFull, stage_idx);
            // 将token写到routed G1输入环的ring_m_idx行，单位FP8元素。
            copy_ub_to_gm(l1_acts + ring_m_idx * config.hidden,
                          // 源来自同一pull stage；写一整行H，不按GMM的N列分工拆token。
                          ub_pull_token[stage_idx].ptr(), 1, config.hidden,
                          config.hidden, kWriteThroughShare);
            // 用同一环形行号写scale，保证data与SF指向同一个expert路由行。
            copy_ub_to_gm(l1_acts_sf + ring_m_idx * config.num_input_sf_pairs,
                          // scale每行元素数为num_input_sf_pairs，不能直接乘H来定位。
                          ub_pull_sf[stage_idx].ptr(), 1, config.num_input_sf_pairs,
                          config.num_input_sf_pairs * sizeof(int16_t), kWriteThroughShare);
            // 本槽UB输入已被MTE3读走，允许MTE2复用；整块ready仍等后面的kPullDone。
            notify<PIPE_MTE3, PIPE_MTE2>(kPullEmpty, stage_idx);

            // Copy route weights as FP32 bits while DMA runs
            // 取metadata高32位，保留FP32权重的原始位型，不做浮点转整数数值转换。
            const uint32_t route_weight_bits = static_cast<uint32_t>(token_metadata >> 32);
            // Scalar将权重bits写到完整pool行位置，epilogue会按float读取它。
            asc_store_dev(reinterpret_cast<__gm__ uint32_t*>(routed_weights + pool_m_idx), route_weight_bits);
            // Record the actual row assignment; combine readiness never infers source/token order
            // 只有overlap reduce需要额外记录每条路由由哪个逻辑块生产，以精确判断早期归并就绪。
            if constexpr (kOverlapReduce)
                // 写回source原token/top-k槽的route_m_blocks；记录本rank生产块号，不记录物理环号。
                asc_store_dev(workspace.route_m_blocks.get_ptr(remote_base, token_topk_idx), pull.pool_block_idx);
            // Source metadata: rank [31:24], combine M [23:0]
            // 保存当前pool行未来的返回地址metadata，和环形data分开保留。
            asc_store_dev(src_metadata + pool_m_idx,
                // 低24位为source的combine接收行，高8位为source rank；get_combine_m_idx使用注册Tmax步长。
                static_cast<uint32_t>(workspace.get_combine_m_idx(src_topk_idx, src_token_idx)) | (src_rank_idx << 24));
        }

        // 本核有效行已全部提交，包含valid_m小于本核首行、完全没有搬运的情况。
        if (pull.m_idx_in_block >= pull.valid_m) {
            // BLOCK_M marks pending stores; publish on the next step
            // 置pending标记，下次advance先确认stores，不立即发布Dispatched以保留流水交错机会。
            pull.m_idx_in_block = BLOCK_M;
            // 在本核MTE3队列尾登记完成事件，下一次Scalar等待它后才发布本块贡献。
            notify<PIPE_MTE3, PIPE_S>(kPullDone);
        }
    };

    // epilogue状态与pull/push分开保存，避免通信进度影响配对AIC结果的消费顺序。
    struct EpilogueState {
        // 重放配对AIC的shared/routed任务顺序，获取每个M/N输出tile。
        sched::LinearTaskState schedule;
        // G1 routed按逻辑块查expert有效行数，shared不使用此游标。
        sched::ExpertCursor<workspace> expert;
        // 当前待消费的本层展平task号，结合阶段确定N块数后解码。
        uint32_t task_idx;
        // 本task所在M-block的有效行数，不是本轮输入T。
        uint32_t valid_m;
        // 初始无当前task；一项消费完成后重置kNone，主循环再取下一项。
        sched::BlockPhase phase = sched::BlockPhase::kNone;
        // G1当前task中下一个待量化行，按最多32行逐步推进。
        uint32_t m_idx_in_block = 0;
        // 本核已消费的GEMM计数，uint8自然回绕，与AIC提交计数配对判断在途任务。
        uint8_t num_completed_gemms = 0;  // Advanced by epilogue, paired with num_issued_gemms
    };
    // 建立初始epilogue状态，各成员默认初始化；首次任务由主循环取出。
    EpilogueState epi{};

    // Process Linear1 rows or wait for Linear2 stores; return true on progress
    // epilogue 每次量化最多32行；完整 M128 stage 的有效行处理完后统一写GM，再归还该 stage 的 FIX credit。
    // GMM1 读 gate/up 配对输出，对 SwiGLU 结果乘路由权重后再量化；量化后才乘权会有不同的舍入语义。
    // 无bias的GMM2在线性实数计算中满足(w*z)*W2=w*(z*W2)，故权重可融合到现有SwiGLU向量计算。
    // 本实现量化的是w*z，amax/scale也从加权值产生；Combine只求和，不能把乘权后移并假定数值逐位一致。
    // Linear2 scale 用 [M-block,K-pair,M] 物理布局写出，不能按 token-major 输入 scale 的 stride 解释。
    // 一个 N-block 完成贡献一个 Linear1Ready，GMM2 等完整列数；更早完成某32行 chunk 不代表整块激活可用。
    // GMM2 分支等待 FIX 的 GM 输出完成，再归还 epilogue credit；这里的 credit 与最终远端 CombineSent 是两层不同约束。
    // 推进当前配对AIC的一项结果；G1最多量化32行，G2只确认FIX到GM完成，bool返回本次是否前进。
    const auto advance_epilogue = [&]() __aicore__ -> bool {
        // 调度器尚未分配任务或任务已耗尽时，没有可消费的FIX。
        if (epi.phase == sched::BlockPhase::kNone)
            // 返回false给主循环，让通信或metadata探测继续推进。
            return false;

        // Issued tasks still need their FIX completion event
        // 新task尚未消费且AIC提交数等于本核完成数，说明没有新的FIX可等；避免提前阻塞而饿住Dispatch。
        if (epi.m_idx_in_block == 0 and asc_load_dev(num_issued_gemms_ptr) == epi.num_completed_gemms)
            // 保持当前task不变，下一轮重新探测提交计数，而非进入跨核Full等待。
            return false;

        // shared和routed使用不同GM布局、权重应用规则，但都消费配对AIC的结果。
        const bool is_shared = epi.phase == sched::BlockPhase::kSharedLinear1 or epi.phase == sched::BlockPhase::kSharedLinear2;
        // G1要执行SwiGLU/乘权/量化；G2不再执行激活，只确认完整输出列的写入。
        const bool is_linear1 = epi.phase == sched::BlockPhase::kLinear1 or epi.phase == sched::BlockPhase::kSharedLinear1;
        // 按本层和shared/routed宽度选择N任务数，用于解析当前task属于哪个M-block。
        const uint32_t num_n_blocks = is_linear1 ? (is_shared ? kNumSharedLinear1NBlocks : kNumLinear1NBlocks) : kNumLinear2NBlocks;
        // 得到不取模的逻辑M块号；稍后选择具体GM地址时再决定是否使用环映射。
        const uint32_t pool_block_idx = epi.task_idx / num_n_blocks;

        // G2结果已经由AIC FIX直接写入GM，不占用BF16投影UB来执行SwiGLU。
        if (not is_linear1) {
            // Wait for Linear2 stores and return FIX credit
            // Scalar等待stage0的FIX Full，证明本task所有输出列子块已写完。
            wait_intra_block<PIPE_S, 1>(kEpilogueFull);
            // 把stage0握手credit归还AIC，允许下一次FIX；仍未证明该GM输出已被Combine消费。
            notify_intra_block<PIPE_S, 1>(kEpilogueEmpty);
        } else {
            // 三层GM环中的中间激活环：GMM1 epilogue写新块b前，必须等旧块b-R的全部GMM2输出列完成。
            // 它与dispatch等待Linear1Ready形成不同反压边；只归还UB/FIX credit不能证明GM中间激活可覆盖。
            // Reuse Linear2 inputs after all output N-blocks finish
            // G1即将覆盖旧中间激活环槽时，先确认旧块的全部G2任务完成。
            if (not is_shared and epi.m_idx_in_block == 0 and pool_block_idx >= kNumRingBlocks and
                    // 只探测是否ready，暂未满足就回主循环；等待的是旧块Linear2Ready而非本块G1 ready。
                    not comm::is_block_ready<workspace>(buffer, layout::kLinear2Ready,
                                                        // 逻辑旧块号b-R加shared进度偏移，和中间data的物理环地址分开计算。
                                                        num_shared_m_blocks + pool_block_idx - kNumRingBlocks,
                                                        // 目标是H/256个G2列任务，保证旧块所有中间通道已经被完整消费。
                                                        kNumLinear2NBlocks))
                // 保留epi状态并让出当前推进机会，等待G2释放中间环空间。
                return false;

            // 新task只计算一次有效M，之后32行小步推进沿用同一个范围。
            if (epi.m_idx_in_block == 0)
                // shared尾块由原T截断；routed按该expert的接收数和块内均分范围决定有效行。
                epi.valid_m = is_shared ? min(num_tokens - pool_block_idx * BLOCK_M, BLOCK_M) :
                    epi.expert.advance_to(buffer, pool_block_idx);
            // 块内行除128选择FIX/epilogue stage0或1，与AIC写入UB片段的索引一致。
            const uint32_t stage_idx = epi.m_idx_in_block / STORE_BLOCK_M;
            // 对128取余得到本stage中的行偏移，用来定位投影、FP8压紧区和路由权重。
            const uint32_t m_idx_in_stage = epi.m_idx_in_block % STORE_BLOCK_M;
            // 当前stage在整个M256块内的起始行，stage0为0、stage1为128。
            const uint32_t stage_m_begin = stage_idx * STORE_BLOCK_M;
            // 尾stage可能不足128行，所有权重装载、量化和写出都只覆盖其有效行。
            const uint32_t stage_valid_m = min(STORE_BLOCK_M, epi.valid_m - stage_m_begin);
            // Yield after at most 32 rows
            // 本次最多做32行且不超过整个task剩余行，处理完返回主循环给通信留机会。
            const uint32_t step_m = min(32u, epi.valid_m - epi.m_idx_in_block);

            // Load route weights and wait for FIX
            // 只有stage第一组行需要加载权重并消费一次FIX Full，后续小组直接使用已就绪的UB。
            if (m_idx_in_stage == 0) {
                // shared输出不应用routed top-k标量，也不访问routed权重数组。
                if (not is_shared) {
                    // 路由权重保存在完整pool身份区，用逻辑块号而非环块号定位本stage的首行。
                    const auto route_weights = workspace.routed.topk_weights.get_ptr(
                        buffer, static_cast<uint64_t>(pool_block_idx) * BLOCK_M + stage_m_begin);
                    // 等V用完旧stage的唯一权重槽，再让MTE2覆盖它。
                    wait<PIPE_V, PIPE_MTE2>(kTopkEmpty);
                    // 把当前stage全部有效行的FP32权重搬到UB，小步量化期间复用这份权重。
                    copy_gm_to_ub(ub_route_weights.ptr(), route_weights, 1, stage_valid_m,
                                  // 传输长度/源行跨度为stage_valid_m*4字节，数据是逐行标量而非H维向量。
                                  stage_valid_m * sizeof(float), asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
                    // 权重装载完成后通知V，必须在乘权前消费这个Full。
                    notify<PIPE_MTE2, PIPE_V>(kTopkFull);
                }
                // 等待AIC FIX写完本stage全部gate/up子列，不能只因为AIC已提交就直接读UB。
                wait_intra_block<PIPE_V, 1>(kEpilogueFull, stage_idx);
                // 只有routed路径需要等待路由权重Full，shared没有对应MTE2装载。
                if (not is_shared)
                    // FIX与权重两条依赖都满足后，才进入下面的向量计算。
                    wait<PIPE_MTE2, PIPE_V>(kTopkFull);
            }

            // Quantize one row group
            // 投影源为BF16，每行物理256列；按本stage行偏移定位本次行组。
            const auto input = ub_accum[stage_idx].ptr() + m_idx_in_stage * BLOCK_N;
            // FP8结果在同一UB槽前部压紧，指针改成FP8后按每行128个输出通道计算偏移。
            const auto output = reinterpret_cast<__ubuf__ float8_e4m3_t*>(ub_accum[stage_idx].ptr()) +
                                // 输入BF16一行512B、输出FP8一行128B，后续未读输入仍位于更高地址，布局按VF约定原地使用。
                                m_idx_in_stage * (BLOCK_N / 2);
            // SF以[pair,M128]存放，本组只在M方向偏移行数，pair跨度由模板STORE_BLOCK_M固定。
            const auto output_sf = ub_output_sf[stage_idx].ptr() + m_idx_in_stage;
            // 根据shared标志编译/选择是否广播逐行路由权重，激活裁剪规则保持一致。
            if (is_shared) {
                // shared做SwiGLU和FP8量化，nullptr明确表示不读取routed权重。
                vf_swiglu_quant<false, STORE_BLOCK_M>(input, output, output_sf, nullptr, step_m, activation_clamp);
            } else {
                // routed先SwiGLU、再乘对应逐行top-k权重，随后生成加权值的FP8 data/SF。
                vf_swiglu_quant<true, STORE_BLOCK_M>(input, output, output_sf, ub_route_weights.ptr() + m_idx_in_stage, step_m, activation_clamp);
            }
            // 推进当前task的待处理行数；这只是V计算提交进度，stage写出/释放在下面单独处理。
            epi.m_idx_in_block += step_m;

            // Finish GM stores before FIX reuses this UB stage
            // 当前stage的全部有效行都已量化，才统一写GM，避免每32行发一次完整stage握手。
            if (m_idx_in_stage + step_m == stage_valid_m) {
                // shared没有路由权重读取，因此也不归还top-k槽credit。
                if (not is_shared)
                    // V不再读取本stage权重，MTE2可装下一stage；这与投影/FP8输出槽是否写完相互独立。
                    notify<PIPE_V, PIPE_MTE2>(kTopkEmpty);
                // 通知MTE3该stage的FP8压紧数据和SF均由V写完。
                notify<PIPE_V, PIPE_MTE3>(kQuantFull);
                // MTE3等待向量结果完成后再读取同一UB区域，防止把未量化完的BF16内容发到GM。
                wait<PIPE_V, PIPE_MTE3>(kQuantFull);

                // task余数得到本G1投影的列块，量化后每task贡献128列而非256列。
                const uint32_t n_block_idx = epi.task_idx % num_n_blocks;
                // shared数据保留全部逻辑块；routed中间data块号按R取模，再乘256得到物理首行。
                const uint64_t output_m_idx = static_cast<uint64_t>(
                    is_shared ? pool_block_idx : layout::get_ring_block_idx<workspace>(pool_block_idx)) * BLOCK_M;
                // 选择shared中间区或routed中间环，二者都供后续G2作为输入。
                const auto dst_layout = is_shared ? workspace.shared_l2_acts : workspace.routed.l2_acts;
                // 每N任务量化后输出128列；完整中间宽度为I或S*I。
                const uint32_t num_cols = num_n_blocks * (BLOCK_N / 2);
                // 中间每行每64个通道一个int16 pair，用于计算GM scale块的跨度。
                const uint32_t num_sf_pairs = num_cols / MX_SF_DIVISOR;
                // data按token行连续存储，加入stage行偏移和本task的128列偏移。
                const auto dst = dst_layout.data.get_ptr(buffer,
                    // 目的行跨度始终为完整中间宽度num_cols，不是本task的128列。
                    (output_m_idx + stage_m_begin) * num_cols + n_block_idx * (BLOCK_N / 2));
                // SF布局为[M-block,K-pair,M256]，先选块，再选本N任务的两个pair列。
                const auto dst_sf = dst_layout.sf.get_ptr(buffer,
                    // stage_m_begin在每个pair的256行中选择前/后128行，不能套用token-major SF地址公式。
                    output_m_idx * num_sf_pairs + n_block_idx * kNumOutputSFPairs * BLOCK_M + stage_m_begin);

                // 从原accum槽读取压紧后的FP8[128,128]视图，只写当前stage有效行。
                copy_ub_to_gm(dst, ub_ptr<float8_e4m3_t>(STORE_BLOCK_M, BLOCK_N / 2, ub_accum[stage_idx].addr),
                              // 目标每行跨度为完整I/SI，缓存模式与跨角色GM共享写回保持一致。
                              stage_valid_m, BLOCK_N / 2, num_cols, kWriteThroughShare);
                // 单独写两个SF pair列，UB每列跨度128行，GM每列跨度256行。
                copy_ub_to_gm(dst_sf, ub_output_sf[stage_idx],
                              // 行数参数为pair数2、列数参数为有效M，目标字节跨度256*sizeof(int16_t)。
                              kNumOutputSFPairs, stage_valid_m, BLOCK_M * sizeof(int16_t), kWriteThroughShare);
                // MTE3完成当前stage读取/写出边界后才归还FIX Empty，AIC可安全复用该UB槽。
                notify_intra_block<PIPE_MTE3, 1>(kEpilogueEmpty, stage_idx);
            }
            // 整个task还有其他行或另一个stage时，只报告本次有进展，不发布整task完成计数。
            if (epi.m_idx_in_block != epi.valid_m)
                // 返回主循环，让pull/push交错；下次从更新后的m_idx_in_block恢复。
                return true;
            // 本task全部有效行已处理，恢复0供下一个task使用。
            epi.m_idx_in_block = 0;
        }

        // 当前G1完整消费或G2 FIX完成，更新与AIC提交计数配对的uint8完成数。
        ++ epi.num_completed_gemms;

        // Shared early reduce counts each active core once in slot 0
        // G1每个N256投影task完成后贡献一次Linear1Ready，shared和routed使用各自块的进度槽。
        if (is_linear1)
            // 发布前内部会等MTE3 stores，使G2看到完成计数时也能读取对应FP8中间列。
            publish_block_progress(layout::kLinear1Ready, (is_shared ? 0 : num_shared_m_blocks) + pool_block_idx, 1);
        // routed G2逐task贡献；shared仅overlap且当前核已完成自己最后一个shared G2任务时贡献一次。
        else if (not is_shared or (kOverlapReduce and epi.task_idx + kNumAICores >= num_shared_m_blocks * kNumLinear2NBlocks))
            // shared统一使用slot0统计活跃核结束，routed使用shared区之后的逻辑M块slot。
            publish_block_progress(layout::kLinear2Ready, is_shared ? 0 : num_shared_m_blocks + pool_block_idx, 1);
        // 清空当前阶段，主循环下一次取配对AIC的下一task。
        epi.phase = sched::BlockPhase::kNone;
        // 本次确实完成/推进了结果消费，主循环可据此调整通信选择。
        return true;
    };

    // Combine源数据是本rankrouted输出环，不是最终output或shared接收槽。
    const auto routed_output = workspace.routed.output.get_ptr(buffer);
    // push独立状态：顺序发送块、块内行和已确认块前缀分开保存。
    struct CombinePushState {
        // 按块查expert有效行数，避免发送尾部padding行。
        sched::ExpertCursor<workspace> expert;
        // 下一个待回传的逻辑块号，按0..B-1递增，不随GM环回绕。
        uint32_t pool_block_idx = 0;
        // 本核下一条要发送的块内行，从core_idx开始每次加32。
        uint32_t m_idx_in_block;
        // 当前块有效行数，第一次进入块时由expert游标计算。
        uint32_t valid_m;
        // 本核已经排空stores并发布CombineSent的右开块前缀，可能落后于已提交发送的pool_block_idx。
        uint32_t num_completed_m_blocks = 0;
    };
    // 从本核负责的首行开始push；默认成员同时初始化expert和已完成前缀。
    CombinePushState push{.m_idx_in_block = core_idx};

    // Send BF16 rows; return true when this step advances
    // Combine 先等当前 M-block 的全部 Linear2 N-block 完成，才从 BF16 输出环读取整行，metadata 恢复源 rank/接收槽。
    // 回传按(token,top-k slot)写独立接收槽，未在生产rank预聚合同一token的多个专家结果；最终reduce再合并这些槽。
    // 每步最多两行随后让出，四个 push UB 槽允许远端写与 epilogue/pull 交错；复用前等 MTE3->MTE2 empty。
    // 完成计数按块批量发布，减少逐块MTE3->S排空等待；最终/末专家边界仍强制排空，保证早期归并前缀准确。
    // 优化增加在途和减少等待频率，不改变 BF16 行传输量，也不把结果直接 AtomicAdd 到最终 y。
    // 一次最多推进两条输出行，未ready返回false，不在这里阻塞整个通信调度循环。
    const auto advance_combine_push = [&]() __aicore__ -> bool {
        // Batch store waits to overlap remote writes with other work
        // 当前常量得到4个M-block：每core最多发送32行后统一等MTE3并发布这4块的贡献。
        // 满M256块由32个AIV1分摊，每core发送8行；32行预算/(256/32)=4块，首个32是固定行预算。
        // 此批量确认间隔与kNumPushStages=4独立；4个UB槽仍逐行用empty保护并循环复用，不存下整批32行。
        // 批处理减少MTE3->S排空等待次数，各逻辑块的atomic_add贡献数不变；32行是否最优需另做性能对比。
        // Combine每块同样只发8行，但本块发送结束后可继续发后续块；不同于Dispatch逐块ready直接放行GMM1。
        // 取8行即每块确认也是可选设计；此处32不能由核数/UB槽数推导，仓库未记录32优于8/16/64的调优依据。
        // 这会延后output ring空间释放，但减少Scalar等待；overlap模式最后一个非空专家改为逐块发布。
        // 每核32行的常规确认预算换算成逻辑块数；当前32核/M256得到4，和UB四槽是独立参数。
        constexpr uint32_t kNumMBlocksPerWait = 32 * kNumAICores / BLOCK_M;
        // 所有块已发完或新块尚未Linear2Ready时，本次不发任何行。
        if (push.pool_block_idx == num_total_m_blocks or (push.m_idx_in_block == core_idx and
                // 只有新块入口检查完整H列的G2完成计数，中途发送时沿用此前已证明的ready。
                not comm::is_block_ready<workspace>(buffer, layout::kLinear2Ready,
                                                    // progress用shared区之后的逻辑块号，不按输出data的环槽号取模。
                                                    num_shared_m_blocks + push.pool_block_idx,
                                                    // 等待本块全部H/256个G2 task，因此能安全读取整行H维BF16。
                                                    kNumLinear2NBlocks)))
            // 返回主循环尝试epilogue或pull，push不越过当前未ready块去偷发后续块。
            return false;

        // 保存本次块号，行发送及后面的递增不会改变此局部索引。
        const uint32_t pool_block_idx = push.pool_block_idx;
        // 第一次进入块时查valid_m，后续两行推进继续使用同一块范围。
        if (push.m_idx_in_block == core_idx)
            // 专家游标与Dispatch相同分块规则，确保不会把逻辑槽padding当有效行发送。
            push.valid_m = push.expert.advance_to(buffer, pool_block_idx);
        // 回传身份按完整pool行保留，使用逻辑块号定位，不随输出环复用被覆盖。
        const auto src_metadata_base = src_metadata + static_cast<uint64_t>(pool_block_idx) * BLOCK_M;
        // 输出data按R取模映射物理块首行，行跨度是完整H。
        const auto src_block = routed_output + layout::get_ring_m_idx<workspace>(pool_block_idx) * config.hidden;

        // Send at most two rows before yielding
        // 一个step回传一行完整H维BF16；Flash为8192B，每次advance最多两行，而ready判断单位仍是整个M-block。
        // 每次最多发两行，达到当前块valid_m后立即停止，避免长通信压住epilogue。
        for (uint32_t step_idx = 0; step_idx < 2 and push.m_idx_in_block < push.valid_m; ++ step_idx) {
            // 保存本次要回传的块内行号。
            const uint32_t m_idx_in_block = push.m_idx_in_block;
            // 本核第几条行对4取模，选择四个BF16 UB槽之一。
            const uint32_t stage_idx = (m_idx_in_block / kNumAICores) % kNumPushStages;
            // 读取Dispatch保存的uint32返回身份：高8位source rank、低24位combine接收行。
            const uint32_t combine_metadata = asc_load_dev(src_metadata_base + m_idx_in_block);
            // 截取接收行号，其内部已经包含source的top-k槽与注册Tmax步长。
            const uint32_t combine_m_idx = combine_metadata & 0x00ffffffu;
            // 在本地routed输出环定位当前完整H维BF16行。
            const auto src = src_block + m_idx_in_block * config.hidden;

            // Load a BF16 output row
            // 等旧MTE3读完此UB槽，再允许MTE2装入新输出行。
            wait<PIPE_MTE3, PIPE_MTE2>(kPushEmpty, stage_idx);
            // 本地GM->UB装载H个BF16，Flash每行8192B，不是GMM2的单个N256子片段。
            copy_gm_to_ub(ub_push_token[stage_idx].ptr(), src, 1, config.hidden,
                          // 字节跨度为H*2，尽量不为仅回传一次的输出占用额外L2空间。
                          config.hidden * sizeof(bfloat16_t), asc_load_l2_cache_mode::NOTALLOC_KEEP);
            // 通知MTE3本槽的完整BF16行已加载，可以开始回传。
            notify<PIPE_MTE2, PIPE_MTE3>(kPushFull, stage_idx);

            // 还原原source rank，专家生产rank与token来源rank可以不同。
            const uint32_t dst_rank_idx = combine_metadata >> 24;
            // 先在同构布局中定位combine行，再映射到source rank；目的槽不会直接覆盖最终output。
            const auto dst = sym_buffer.map(workspace.combine_input.get_ptr(buffer, static_cast<uint64_t>(combine_m_idx) * config.hidden), dst_rank_idx);

            // Send the BF16 row to its source token
            // MTE3等待本行GM->UB装载完成，保证远端写出的数据完整。
            wait<PIPE_MTE2, PIPE_MTE3>(kPushFull, stage_idx);
            // 将当前UB行写到source的独立top-k接收槽，无数值AtomicAdd，也未聚合同token多expert输出。
            copy_ub_to_gm(dst, ub_push_token[stage_idx].ptr(), 1, config.hidden,
                          // 按完整H*2字节写一行，store缓存策略用于本次通信写回。
                          config.hidden * sizeof(bfloat16_t), asc_store_l2_cache_mode::NOTALLOC_CLEAN);
            // 当前MTE3读完UB后归还该push槽，允许后续MTE2装载复用。
            notify<PIPE_MTE3, PIPE_MTE2>(kPushEmpty, stage_idx);
            // 下一条行跨32，保持32个AIV1的通信行分工互不重叠。
            push.m_idx_in_block += kNumAICores;
        }

        // 本核当前块所有有效行均已提交，空owner同样会进入此分支。
        if (push.m_idx_in_block >= push.valid_m) {
            // 为下一块恢复本核首行号，不把上一块已越界的行号带过去。
            push.m_idx_in_block = core_idx;
            // 推进已提交发送的逻辑块号；这时尚不一定已经发布CombineSent。
            ++ push.pool_block_idx;

            // Wait for batched stores; publish each last-expert block for early reduce
            // 常规每4块确认一次；尾部条件另行强制确认，不能等不到整批时一直留着完成证书。
            if (push.pool_block_idx % kNumMBlocksPerWait == 0 or
                    // 非overlap在最后块确认；overlap进入最后一个非空expert后逐块确认，以及时发布归并前缀。
                    (kOverlapReduce ? push.expert.pool_block_end == num_total_m_blocks : push.pool_block_idx == num_total_m_blocks)) {
                // 一次等MTE3排空后，为自上次已确认前缀以来的每个块分别发布本核CombineSent贡献。
                publish_block_progress(layout::kCombineSent,
                                       // 起点加shared进度偏移，数量是新提交块数，不是本核实际发送的行数。
                                       num_shared_m_blocks + push.num_completed_m_blocks, push.pool_block_idx - push.num_completed_m_blocks);
                // 记录本核确认到的新右开前缀，下一批只发布更后面的块，避免重复加完成计数。
                push.num_completed_m_blocks = push.pool_block_idx;
            }
        }
        // 本次推进了行或块游标，供主循环在push/pull之间交替选择。
        return true;
    };

    // Publish the exact completed-block prefix shared by all source ranks
    // 跨rank前缀发布状态与本核push完成前缀分开：还需等其他31个AIV1的贡献。
    struct CombinePrefixState {
        // 已经向source rank公布的右开块前缀，初始0表示尚无块被认证可归并。
        uint32_t num_published_m_blocks = 0;
        // 记录本核是否无需发布或已经发布最终结束标记，作为主循环退出条件之一。
        bool finished;
    };
    // 非overlap不发前缀；overlap时只有分到source rank发布任务的核需要等待发布完成。
    CombinePrefixState ready_prefix{.finished = not kOverlapReduce or core_idx >= config.num_ranks};

    // 早期归并的远端证书是已完成逻辑块前缀，不是最新完成的任意块号。
    // 先等最后一个前缀块的所有 AIV1 CombineSent，DSB_DDR 后向各源 rank 发布同一生产 rank 的前缀。
    // 每个 token/top-k 的 route_m_blocks 记录实际生产块；归并逐路由检查块号<peer前缀，不能按输入 token 顺序推断。
    // 全部块已发送发布 UINT64_MAX，空路由也可结束；缓存前缀单调增长，避免每个通道 chunk 重复读 peer flag。
    // 把本rank已经全核确认的连续发送前缀写到各source rank，用于早期reduce逐路由ready判断。
    const auto publish_combine_prefix = [&]() __aicore__ {
        // 从本核已经确认stores的前缀起步，不能使用仅提交发送的push.pool_block_idx。
        const uint32_t pool_block_end = push.num_completed_m_blocks;
        // 已结束或前缀未增加则不重复写peer；B=0特殊允许发布空生产方的最终结束标记。
        if (ready_prefix.finished or (num_total_m_blocks != 0 and pool_block_end <= ready_prefix.num_published_m_blocks))
            // 本次没有新证书可发，返回主循环继续计算/通信。
            return;
        // 非空前缀还需检查最后一块的全部32个AIV1贡献，而不只相信本核已发送完成。
        if (pool_block_end != 0 and not comm::is_block_ready<workspace>(
                // 各核按块单调推进，因此末块全核完成能证明此前所有块的连续前缀都已完成。
                buffer, layout::kCombineSent, num_shared_m_blocks + pool_block_end - 1, kNumAICores))
            // 其他owner尚未完成，延后发布而不阻塞本核的其他小步工作。
            return;

        // 记录已获全核确认的新前缀，避免下次重复发布相同范围。
        ready_prefix.num_published_m_blocks = pool_block_end;
        // 前缀等于B说明本rank所有routed结果已发送，后续可用最终哨兵替代普通块号。
        ready_prefix.finished = pool_block_end == num_total_m_blocks;
        // 最终标记为UINT64_MAX，普通前缀是pool_block_end右开边界；空生产方也写最终标记。
        const uint64_t prefix = ready_prefix.finished ? ~uint64_t{0} : pool_block_end;
        // 32个AIV1按source rank号分摊通知；P超过32时每核跨32继续覆盖其他source。
        for (uint32_t src_rank_idx = core_idx; src_rank_idx < config.num_ranks; src_rank_idx += kNumAICores) {
            // 先建立此前GM数据与证书操作的顺序，再把可读前缀写给消费者。
            asc_sync_data_barrier(DSB_DDR);
            // 在目标source的combine_ready_prefix[本生产rank]槽写入前缀，不是写[source rank]槽。
            asc_store_dev(sym_buffer.map(workspace.metadata.combine_ready_prefix.get_ptr(buffer, rank_idx), src_rank_idx), prefix);
        }
    };

    // AIV1 的稳态优先推进 epilogue，再交替选择 push/pull；一个未就绪阶段返回而非堵住全部工作。
    // prefer_push 只在当前可推进任务之间交替，不取消数据 ready 与环形空间的依赖条件。
    // reduce 仅在尾部窗口打开且本核没有未消费 GEMM 时插入；每次小步推进，减少最终全量归并暴露的尾部。
    // 该策略与实验版的固定优先级有相似目的，但这里以 M-block 环和 task schedule 为单位，不能套用专家 slice 游标。
    // 初始不优先push；第一次通信机会先pull，之后在可推进的push/pull之间切换偏好。
    bool prefer_push = false;
    // 以带超时的循环驱动多个状态机，predicate不仅检查完成也会实际推进工作。
    comm::wait_until([&]() __aicore__ {
        // Service the reduce window; check token readiness inside
        // 尾部窗口要求overlap已编译开启、metadata已知且push接近末尾；窗口跨度来自本rankT。
        const bool reduce_window_open = kOverlapReduce and metadata_ready and push.pool_block_idx + num_reduce_window_m_blocks >= num_total_m_blocks and
            // 本次观察到提交数等于消费数时才插入reduce，避免已有FIX结果长期无人消费。
            asc_load_dev(num_issued_gemms_ptr) == epi.num_completed_gemms;

        // 窗口满足后先尝试发布全核发送前缀，再考虑本rank输出的早期求和。
        if (reduce_window_open) {
            // 前缀发布负责让远端source看到本rank已发送结果可读，即使本核没有可归并token也有作用。
            publish_combine_prefix();
            // 路由块数超过物理环容量才在主体内插入reduce，小规模任务留到尾处理统一归并。
            if (num_total_m_blocks > kNumRingBlocks)
                // 每次最多推进两个通道chunk，以保留epilogue和通信的推进机会。
                advance_combine_reduce<workspace, kOverlapReduce>(buffer, output, num_tokens, reduce, ub_reduce_inputs, 2);
        }

        // Fetch the next epilogue task
        // 只有上一task全部消费完才从静态序列取下一task；部分M行未完时继续保留当前任务。
        if (epi.phase == sched::BlockPhase::kNone)
            // 为配对AIC的下一项FIX结果取得M/N任务号，shared任务可在metadata未知时先执行。
            epi.task_idx = sched::take_linear_task<workspace>(
                // epilogue使用本rankT与同一个routed B重放任务，不按pull/push块号直接推算task。
                num_tokens, num_total_m_blocks, epi.schedule, core_idx, epi.phase, metadata_ready);

        // Poll metadata arrival
        // metadata未到且当前不在shared G1时轮询本核标志，避免将metadata检查插入shared G1的每个小组。
        if (not metadata_ready and epi.phase != sched::BlockPhase::kSharedLinear1 and
            // AIV0按core_idx独立发布此GM字节标志；非零证明整个本rank路由计数和块数证书已完整。
            asc_load_dev(workspace.progress.metadata_ready.get_ptr(buffer, core_idx)) != 0) {
            // 观察ready后建立GM读取顺序，再读取与之配套的总块数证书。
            asc_sync_data_barrier(DSB_DDR);
            // 低24位为本轮routed B，高位是汇合计数，不参与任务块数计算。
            num_total_m_blocks = asc_load_dev(num_m_blocks_certificate) & layout::kCountValueMask;
            // 此后可以推进routed Dispatch/Combine并在调度器中枚举routed task。
            metadata_ready = true;
        }

        // 只有metadata已知且pull游标到B，才能判定输入整理全部结束，避免把初始B=0误认成完成。
        const bool pull_finished = metadata_ready and pull.pool_block_idx == num_total_m_blocks;
        // 输入全部整理、epilogue无剩余任务、push全部提交且前缀已结束，才可离开此通信主循环。
        if (pull_finished and epi.phase == sched::BlockPhase::kNone and push.pool_block_idx == num_total_m_blocks and ready_prefix.finished)
            // predicate返回true结束wait_until；最终输出剩余归并仍由外层combine_reduce处理。
            return true;

        // Advance epilogue before choosing push or pull
        // 优先消费已经提交的AIC结果，及时归还FIX/UB credit，避免计算流水被输出槽反压。
        const bool epi_progress = advance_epilogue();
        // metadata到达后才做routed通信；pull已结束且本轮epilogue前进时暂不插push，下一轮再试。
        if (metadata_ready and not (pull_finished and epi_progress)) {
            // 偏好push或pull已耗尽时尝试回传；push未ready返回false后仍可回退到pull。
            if ((prefer_push or pull_finished) and advance_combine_push()) {
                // 本次push有进展，下次在pull仍有工作时优先给输入搬运机会。
                prefer_push = false;
            // push未推进且输入尚未全部整理，就尝试当前pull块。
            } else if (not pull_finished) {
                // Fall back to pull; prefer push again on the next step
                // pull小步搬运或发布其块ready；若环暂不可覆盖，也会返回而不占住全部循环。
                advance_dispatch_pull();
                // 下次先尝试push，避免持续拉输入导致已完成输出长期不回传。
                prefer_push = true;
            }
        }
        // 本轮尚未满足全部退出条件，继续下一次探测/小步推进。
        return false;
    // 超过统一等待预算时打印本地rank/core，wait_until再触发超时终止。
    }, [&]() __aicore__ { comm::print_local_timeout(rank_idx); });
    // 有shared结果时，在离开本AIV1的shared/routed消费阶段前补充GM顺序边界。
    if constexpr (kHasSharedExperts)
        // 这条数据屏障不是跨rank的全同步，不能替代Combine前缀或最终AIV->AIC握手。
        asc_sync_data_barrier(DSB_DDR);
}

// 融合 kernel 入口：AIC 执行矩阵流水，AIV0 建元数据并转换权重，AIV1 执行 epilogue/通信。
// 角色共用静态 Workspace；输出归并按工作量选择等待末尾或与计算重叠，两条路径均写回原 token 顺序。
// AIC 在两侧 AIV 完成最终存储后退出，使整个 launch 的完成覆盖通信与输出写回。
// 模板固定EP大小P、全局routed专家E、shared专家数S，决定专家归属和共享中间宽度。
template <uint32_t kNumRanks, uint32_t kNumExperts, uint32_t kNumSharedExperts,
          // 其余参数固定注册Tmax、top-k、H、I；本轮实际T通过函数参数传入，不能改变已注册地址跨度。
          uint32_t kNumMaxTokensPerRank, uint32_t kNumTopk, uint32_t kHidden, uint32_t kIntermediateHidden>
// @param output 最终BF16[T,H]；cumulative_local_expert_recv_stats可空，累计本地专家有效路由数。
// @param l1_weights/l2_weights routed packed FP4，物理形状[E/P,Nblock,256,K/2]；l1配对gate128/up128。
// @param l1_weights_sf/l2_weights_sf int16 scale pair，形状[E/P,Nblock,K/64,256]，已补偿FP4展开的/64。
// @param shared_l1_weights/shared_l2_weights 可选shared FP8 NZ权重，无expert维；对应sf仍为K-pair/N布局。
// @param num_tokens 本rank有效T<=注册Tmax；activation_clamp是host转为BF16精度后的SwiGLU裁剪界限。
// @param sym_buffer 本rank基址、rank编号及peer字节偏移；输入x/scale/top-k已经写在该buffer中。
// 模板参数固定P/E/S/Tmax/Ktop/H/I，决定布局与JIT特化；运行时T不会改变注册步长。
// 融合kernel每个逻辑核组为一个AIC配两个AIV；output最终写BF16，stats是可选累计路由计数。
__global__ __mix__(1, 2) void mega_moe_impl(__gm__ bfloat16_t* output, __gm__ int32_t* cumulative_local_expert_recv_stats,
                                            // routed G1的packed FP4 data与int16 SF pair；data为按N256预排的gate/up配对布局。
                                            __gm__ const uint8_t* l1_weights, __gm__ const int16_t* l1_weights_sf,
                                            // routed G2的packed FP4 data与SF，归约宽度为I，输出宽度为H。
                                            __gm__ const uint8_t* l2_weights, __gm__ const int16_t* l2_weights_sf,
                                            // shared G1使用已经预排NZ的FP8 data与SF，无需AIV0展开FP4。
                                            __gm__ const uint8_t* shared_l1_weights, __gm__ const int16_t* shared_l1_weights_sf,
                                            // shared G2使用FP8 data与SF；shared关闭时相应路径不执行。
                                            __gm__ const uint8_t* shared_l2_weights, __gm__ const int16_t* shared_l2_weights_sf,
                                            // num_tokens为当前rank有效T；activation_clamp传给SwiGLU，不参与任务分核公式。
                                            uint32_t num_tokens, float activation_clamp,
                                            // sym_buffer携带本rank号/基址和各peer偏移，用于对称GM寻址，不是实际复制所有输入。
                                            layout::SymBuffer<> sym_buffer) {
    // 由完整模板参数选定唯一静态Workspace，所有角色用相同区域偏移解释同一buffer。
    constexpr auto& workspace = layout::kWorkspace<kNumRanks, kNumExperts, kNumSharedExperts,
                                                   // 注册Tmax参与布局容量与peer步长，实际T再小也不能据它重新压缩本次布局。
                                                   kNumMaxTokensPerRank, kNumTopk, kHidden, kIntermediateHidden>;
    // 引用静态模型/通信配置，后续constexpr条件可在JIT实例中消除无关分支。
    constexpr auto& config = workspace.config;

    // 该分支按注册T_max而非本轮T编译确定，估算BF16 combine_input容量是否严格大于32MiB。
    // 开启后也只在尾部窗口且总路由块数大于ring时插入reduce；不是从kernel开头持续抢占计算。
    // 阈值是当前实现的经验选择，具体收益需结合peer等待、UB余量与任务分布测量。
    // Measured 32 MiB combine-work threshold for tail reduction
    // 把注册容量先提升到uint64再连乘，避免大shape的combine工作量估计在32位溢出。
    constexpr bool kOverlapReduce = static_cast<uint64_t>(config.num_max_tokens_per_rank) *
        // BF16接收区估计为Tmax*(Ktop+shared槽)*H*2B，严格大于32MiB时选择overlap协议。
        (config.num_topk + (config.num_shared_experts > 0)) * config.hidden * sizeof(bfloat16_t) > 32ull * 1024 * 1024;

    // 每rank连续持有E/P个routed专家，用于清计数和根据expert ID定位生产rank。
    constexpr uint32_t kNumExpertsPerRank = config.num_experts / config.num_ranks;

    // Block readiness reads the low byte of each completion counter
    // G2每块的完成贡献数必须能放进uint8，因ready探测只读取uint32计数的低字节。
    static_assert(config.hidden / BLOCK_N <= UINT8_MAX and
                  // 同样约束routed G1列任务数I/128，避免累计到256后低字节回绕。
                  config.intermediate_hidden / (BLOCK_N / 2) <= UINT8_MAX and
                  // shared G1按合并中间宽度S*I检查，不能只验证单expert的I。
                  config.shared_intermediate_hidden / (BLOCK_N / 2) <= UINT8_MAX);

    // 初始化Ascend C API内核执行状态，在任何管线/同步指令前完成基础设置。
    asc_init();
    // 逻辑core_idx范围0~31，AIC和其配对AIV使用相同编号重放M/N任务。
    const uint32_t core_idx = static_cast<uint32_t>(block_idx);

    // 取得本rank对称GM的字节基址，后续各typed Buffer按固定offset定位。
    const auto buffer = sym_buffer.get_base_ptr<__gm__ uint8_t*>();
    // 当前EP rank号用于peer映射、返回metadata及等待诊断，不等于core_idx。
    const uint32_t rank_idx = sym_buffer.rank_idx;
    // 输入top-k expert表的GM指针，metadata VF扫描它构建source/expert路由表。
    const auto topk_idx = workspace.input.topk_idx.get_ptr(buffer);
    // 取得本rankrouted总块数证书指针，入口先清零，AIV0汇合后发布。
    const auto num_m_blocks_certificate = workspace.metadata.num_m_blocks_certificate.get_ptr(buffer);

    // 把G1/G2 routed的data/SF指针打包成统一权重视图，不在入口复制矩阵内容。
    const layout::Weights weights{{l1_weights, l1_weights_sf}, {l2_weights, l2_weights_sf},
                                  // 再补齐shared两层视图，AIC按任务类型选择对应指针。
                                  {shared_l1_weights, shared_l1_weights_sf}, {shared_l2_weights, shared_l2_weights_sf}};

    // AIC: GEMMs, then wait for both AIVs
    // AIC只进入矩阵分支，不参与下面64个AIV的入口清理屏障。
    if ASCEND_IS_AIC {
        // shared可先按本地T执行，routed会在内部等待AIV0 metadata及AIV1输入ready。
        aic_gemm_pipeline<workspace>(buffer, num_tokens, rank_idx, weights, core_idx);
        // 矩阵任务提交完后，Scalar等配对AIV0/AIV1各自完成最终output stores，覆盖通信/归并尾部。
        asc_sync_block_wait(PIPE_S, kCombineDoneID);
        // AIC结束；不落入后面的AIV token分工和通信代码。
        return;
    }

    // Partition tokens across AIVs
    // 取本核组内AIV角色0/1，分别对应metadata/权重与epilogue/通信。
    const uint32_t subblock_idx = static_cast<uint32_t>(asc_get_sub_block_id());
    // 把(core,subblock)展平为0~63的AIV编号，最终reduce和入口清理使用此全局编号。
    const uint32_t aiv_idx = 2 * core_idx + subblock_idx;

    // Overlap mode revisits round-robin tokens; the final-only path uses contiguous ranges
    // overlap按AIV余数轮转分token；非overlap按ceil(T/64)的连续区间分摊，起点最多截到T。
    const uint32_t first_token_idx = kOverlapReduce ? aiv_idx : min(aiv_idx * ceil_div(num_tokens, kNumAIVs), num_tokens);
    // 计算本AIV实际负责的原token数，不能把routed复制后的路由数用作最终输出行数。
    const uint32_t num_reduce_tokens = kOverlapReduce ?
        // overlap只计算aiv_idx+64*k落在T以内的数量；没有分到行的AIV记0，仍参加同步。
        (num_tokens > aiv_idx ? ceil_div(num_tokens - aiv_idx, kNumAIVs) : 0) :
        // 非overlap尾区间按T截断，避免64份向上取整后访问不存在的原token。
        min(ceil_div(num_tokens, kNumAIVs), num_tokens - first_token_idx);
    // 初始化本AIV独立的归并状态；其余成员默认清零，stage按哨兵等待首次使用初始化。
    CombineReduceState<workspace> reduce{.token_idx = first_token_idx, .num_remaining_tokens = num_reduce_tokens};

    // 启动顺序：host同stream清progress；AIV清本地expert arrival与M-block证书；
    // 所有rank通过barrier 0后才允许新的metadata/count写入，防止远端增量被较慢rank清零。
    // AIC走独立分支，可先执行shared GMM1/GMM2；AIV1通过epilogue配合shared，AIV0并行构造路由。
    // Clear counts before metadata publication
    // 全部64个AIV按expert编号分摊清零本rank接收总证书，避免单核串行清所有expert。
    for (uint32_t expert_idx = aiv_idx; expert_idx < kNumExpertsPerRank; expert_idx += kNumAIVs)
        // 清掉旧轮的高位arrival与低位接收行数；source单独count表由本轮各source覆盖更新。
        asc_store_dev(workspace.metadata.expert_recv_count_sum.get_ptr(buffer, expert_idx), uint32_t{0});
    // 整个rank只由全局AIV0号清一次总块数证书，防止多核互相覆盖新贡献。
    if (aiv_idx == 0)
        // 证书清零后，本轮32个AIV0会分别贡献到达位和负责expert的块数。
        asc_store_dev(num_m_blocks_certificate, uint32_t{0});
    // 确实负责清计数的AIV先设置GM顺序边界，避免清零滞后于对外放行。
    if (aiv_idx < kNumExpertsPerRank)
        // 保证本核清理相关GM store的顺序可见性；本条自身不是跨rank屏障。
        asc_sync_data_barrier(DSB_DDR);
    // 本地64 AIV汇合再跨rank握手，所有rank完成清理后才允许新的远端count/arrival写入。
    comm::rank_barrier<workspace>(buffer, sym_buffer, aiv_idx);

    // AIV0: metadata and routed weights
    // AIV0角色先完成metadata，再展开routed权重；不是只启动rank内一个AIV0。
    if (subblock_idx == 0) {
        // 按每512个路由槽配置一个SIMT worker，至少1个、至多32个；T=0也需发布零计数到达。
        const uint32_t num_metadata_workers = min(max(ceil_div(num_tokens * config.num_topk, layout::kNumDispatchThreads), 1u), kNumAICores);
        // 只有前num_metadata_workers个AIV0调用VF；未参与构建的AIV0稍后也参加接收与块数汇合。
        if (core_idx < num_metadata_workers) {
            // SIMT reads UB offsets; metadata owns this scratch until the VF completes
            // 在本AIV0的UB保存每个peer相对本rank的字节偏移，供SIMT寻址读取。
            __ubuf__ int64_t offsets[kNumRanks];
            // 逐rank复制offset表，避免VF直接依赖调用方寄存器中不可访问的参数表示。
            for (uint32_t i = 0; i < kNumRanks; ++ i)
                // 保存peer_base-local_base，后面map可由本地buffer基址加偏移得到远端地址。
                offsets[i] = sym_buffer.offsets[i];
            // offsets先对SIMT读取可见，再调用metadata VF；VF结束前这段UB必须保持有效。
            asc_sync_data_barrier(DSB_UB);
            // 启动当前worker的metadata构建，内部计数/整理/发送并在完整数据之后发布count。
            asc_vf_call<comm::build_dispatch_metadata<workspace, kOverlapReduce>>(
                // 每worker逻辑线程块为512*1*1，worker数量是AIV0数量而非这512个线程数。
                dim3(layout::kNumDispatchThreads, 1, 1),
                // 传入本轮T、rank、worker号/总数与UB偏移表，各worker处理互不重叠的路由索引范围。
                buffer, topk_idx, num_tokens, rank_idx, core_idx, num_metadata_workers, offsets);
            // 在V队列尾登记metadata结束，覆盖VF中的UB使用和已发布的远端metadata操作。
            notify<PIPE_V, PIPE_S>(kMetadataDone);
            // Scalar等待V结束后才能继续使用该AIV0的UB来准备权重，不能提前覆盖offsets/直方图。
            wait<PIPE_V, PIPE_S>(kMetadataDone);
        }

        // 所有AIV0等各本地expert的P个source到齐，再汇合32核块数；可选stats累加的是有效路由行。
        const uint32_t num_total_m_blocks = comm::wait_dispatch_metadata<workspace>(buffer, rank_idx, core_idx, cumulative_local_expert_recv_stats);

        // 元数据先经所有源rank到达、再经全部32个AIV0汇总M-block的两级证书，最后通知配对C/V。
        // AIV1轮询GM标志时仍可做shared epilogue；AIC消费flag2后，AIV0链路才复用于L1B ready。
        // AIV0可先在自身UB准备FP4/NZ，但要等AIC完成shared B消费并发放L1BEmpty后才能写L1。
        // Notify AIV1 via GM and AIC via event
        // 给配对AIV1发布本rankmetadata可用字节，AIV1轮询后读取总块数并开始routed通信。
        asc_store_dev(workspace.progress.metadata_ready.get_ptr(buffer, core_idx), uint8_t{1});
        // 通过AIV0->AIC链路flag2发布相同就绪状态；AIC消费后B槽事件才能复用该编号。
        notify_intra_block<PIPE_S, 0>(kMetadataFull);

        // AIV0按配对AIC任务序列持续供给FP4展开权重，尾部可在剩余UB里插入少量reduce。
        aiv0_weight_pipeline<workspace, kOverlapReduce>(buffer, output, num_tokens, reduce, num_total_m_blocks, weights, core_idx);
    } else {
        // AIV1: epilogue and communication
        // AIV1按小步状态机交错推进epilogue、pull、push和前缀发布，不另起独立通信kernel。
        aiv1_epilogue_comm<workspace, kOverlapReduce>(buffer, output, num_tokens, rank_idx, reduce, activation_clamp, core_idx, sym_buffer);
    }

    // 两种AIV角色都进入最终归并，排空旧缓冲后复用整块UB，继续各自保存的token/channel游标。
    combine_reduce<workspace, kOverlapReduce>(buffer, output, num_tokens, rank_idx, aiv_idx, reduce, sym_buffer);

    // Notify AIC after final output stores
    // 在本AIV最终MTE3输出完成边界通知配对AIC；两个AIV都到达后AIC的block wait才结束。
    asc_sync_block_arrive(PIPE_MTE3, kCombineDoneID);
}

} // namespace deep_gemm::mega_moe
