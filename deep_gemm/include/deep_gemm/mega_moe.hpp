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

// MegaMoE 主流水：输入已在 symmetric buffer 中量化为 FP8，routed 权重为 packed FP4。
// 每个混合 block 配置 1 个 AIC + 2 个 AIV：
//   AIC：按同一静态任务序列交错执行 Linear1(GMM1) / Linear2(GMM2)。
//   AIV0：先构建路由元数据，再将 FP4 权重展开为 FP8 NZ，向 AIC 的 L1B 供数。
//   AIV1：推进 dispatch pull、SwiGLU/量化和 combine push；尾段可插入本地归并。
// 核心依赖：dispatch M-block -> GMM1 各 N-block -> SwiGLU/FP8 -> GMM2 各 N-block
//           -> BF16 回传 -> 源 rank 求和。各箭头由事件或 GM 完成计数保护。
using layout::BLOCK_M;
using layout::BLOCK_N;

// Pipeline choices: double-buffer L0A/B and keep three A / four routed B stages in L1
// L0A/L0B双缓冲，每个槽由MMAD完成事件归还；stage数也决定MAD_K与MAD_N的切分。
constexpr uint32_t kNumL0Stages = 2;
// L1A使用3份激活槽、routed L1B使用4份权重槽、scale各2份；三种槽独立推进。
constexpr uint32_t kNumL1AStages = 3, kNumL1BStages = 4, kNumSFStages = 2;

// Split each logical M-block between two FIX/epilogue buffers
// GMM1的M256输出拆为两个M128 FIX/Vector消费stage。
constexpr uint32_t kNumEpilogueStages = 2;
// M256 的 FIX/epilogue 输出按两份 M128 UB 切开；只缩小写回/消费粒度，不改变逻辑 M-block。
constexpr uint32_t STORE_BLOCK_M = BLOCK_M / kNumEpilogueStages;

// FP8 compute shapes
// BLOCK_M  BLOCK_N  BLOCK_K  MAD_M  MAD_N  MAD_K
//     256      256      512    128    128    256
//     256      256      256    256    256    128
template <uint32_t kMadM>
struct MMAConfig {
    // 逻辑数据块固定M256，不因选用M128 MAD而改变GM stride/进度槽编号。
    static constexpr uint32_t BLOCK_M = layout::BLOCK_M;
    // 每GEMM任务物理N256；GMM1中其中一半为gate、一半为up。
    static constexpr uint32_t BLOCK_N = layout::BLOCK_N;
    // FP8元素1B，完整L0A容量除MAD_M，得到一个L1A K块宽度，使3个L1A槽stride恒定。
    static constexpr uint32_t BLOCK_K = L0ASizeBytes / kMadM;
    // 本实例的MMAD行形状，实际参与行数另按FRAC_MN对齐。
    static constexpr uint32_t MAD_M = kMadM;
    // 将当前K块分成两个L0A stage，M128时MAD_K256，M256时MAD_K128。
    static constexpr uint32_t MAD_K = BLOCK_K / kNumL0Stages;
    // 每份L0B的可用容量除MAD_K，得MAD_N；M128分两份N128，M256一份N256。
    static constexpr uint32_t MAD_N = L0BSizeBytes / kNumL0Stages / MAD_K;
    // 物理N256必须能分成完整MAD_N子块，避免最后N尾片需要额外布局。
    static_assert(BLOCK_N % MAD_N == 0);
    // FP32累加的整任务输出必须放进L0C，不能因动态valid_m误算最大容量。
    static_assert(MAD_M * BLOCK_N * sizeof(float) <= L0CSizeBytes);

    // Shared B reuses routed B's region; events 5-7 permit at most three stages
    static constexpr __aicore__ uint32_t get_num_shared_l1b_stages() {
        // 共享权重每stage装完整N256，按当前K块计算可容纳几份；与routed N子块stage的物理粒度不同。
        constexpr uint32_t num_slots = kNumL1BStages * L0BSizeBytes / (BLOCK_N * BLOCK_K);
        // 即使空间还能多放，也最多使用事件5/6/7对应的3个共享B槽。
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

// 本核事件约定：Full 表示生产者完成写、消费者可读；Empty 表示消费者用完、生产者可覆盖。
// 不同 PIPE 方向的事件号有各自语义，不能把任意 SetFlag 当成跨核/跨 rank 的可见性保证。
// AIC: L1A 0–2, SF 3–4, shared L1B 5–6 or 5–7
// MTE2写L1完成后通知MTE1读取；激活/scale/shared权重各占独立事件起点。
constexpr PipeEvent<PIPE_MTE2, PIPE_MTE1> kL1AFull{0}, kSFFull{3}, kSharedL1BFull{5};
// MTE1把L1数据搬到L0后归还给MTE2，L1槽可在MMAD继续算L0时提前复用。
constexpr PipeEvent<PIPE_MTE1, PIPE_MTE2> kL1AEmpty{0}, kSFEmpty{3}, kSharedL1BEmpty{5};

// AIC: A uses 0–1; split-N B uses 2–3, otherwise readiness shares A's events
// L0已填完的通知；split-N时B事件与A区分，整N时共享顺序保证。
constexpr PipeEvent<PIPE_MTE1, PIPE_M> kL0AFull{0}, kL0BFull{2};
// MMAD消费完L0A或L0B后分别释放该slot，不能只等L1->L0拷贝结束。
constexpr PipeEvent<PIPE_M, PIPE_MTE1> kL0AEmpty{0}, kL0BEmpty{2};

// AIV0: FP4 has 3 stages at 0; NZ has 2 stages at 0
// packed FP4 DMA结束，Vector可开始解包该FP4 stage。
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kFP4Full{0};
// Vector已不读该packed stage，MTE2可覆盖下一权重子块。
constexpr PipeEvent<PIPE_V, PIPE_MTE2> kFP4Empty{0};
// Vector已产生FP8 NZ，MTE3可把它写到AIC的L1B。
constexpr PipeEvent<PIPE_V, PIPE_MTE3> kNZFull{0};
// MTE3已消费NZ UB，Vector可覆盖该NZ slot。
constexpr PipeEvent<PIPE_MTE3, PIPE_V> kNZEmpty{0};
// metadata VF在Vector/SIMT侧结束后通知Scalar，允许复用其UB区域。
constexpr PipeEvent<PIPE_V, PIPE_S> kMetadataDone{0};

// pull 的两个 UB 槽经历远端 MTE2 读 -> 本地 MTE3 写；push 的四槽经历本地 MTE2 读 -> 远端 MTE3 写。
// 两组槽和事件号分开，因此交替推进两条路径不会覆盖彼此的在途 payload。
// AIV1: pull 0–1 and push 2–5, full forward and empty backward
// pull暂存每核2行FP8，push暂存每核4行BF16，都是UB行槽而非GM M-block数。
constexpr uint32_t kNumPullStages = 2, kNumPushStages = 4;
// 两条DMA路径各自MTE2读完后通知MTE3写，事件号按stage加到基础ID上。
constexpr PipeEvent<PIPE_MTE2, PIPE_MTE3> kPullFull{0}, kPushFull{2};
// MTE3不再读取对应UB行后归还给MTE2；发送路径远端写慢会延后此credit。
constexpr PipeEvent<PIPE_MTE3, PIPE_MTE2> kPullEmpty{0}, kPushEmpty{2};

// AIV1: top-k weights, epilogue output, and store completion
// route weight DMA结束才能给Vector用于SwiGLU乘权重。
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kTopkFull{1};
// Vector处理完当前stage的weight，允许下一M128 stage重用唯一weight UB。
constexpr PipeEvent<PIPE_V, PIPE_MTE2> kTopkEmpty{0};
// SwiGLU/FP8/scale生成结束，允许MTE3读取量化结果。
constexpr PipeEvent<PIPE_V, PIPE_MTE3> kQuantFull{0};
// Scalar需要确认的MTE3完成点：pull整块发布和量化/回传发布分别使用事件。
constexpr PipeEvent<PIPE_MTE3, PIPE_S> kPullDone{0}, kStoresDone{2};

// AIV0/AIV1: reduce stages 6–7 and the final communication drain
// 本地最终reduce也使用两个UB槽，独立于push/pull行缓冲。
constexpr uint32_t kNumReduceStages = 2;
// 归并输入DMA读完，允许Vector开始该channel chunk的FP32累加。
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kReduceFull{6};
// Vector完成输出BF16，允许MTE3写最终y。
constexpr PipeEvent<PIPE_V, PIPE_MTE3> kReduceOutputFull{6};
// y写出不再读该reduce UB stage，允许下一chunk复用。
constexpr PipeEvent<PIPE_MTE3, PIPE_MTE2> kReduceEmpty{6};
// 函数收尾等待本核通信MTE3队列完成，再扩大UB范围执行最终归并。
constexpr PipeEvent<PIPE_MTE3, PIPE_S> kCommDone{3};

// 这组是同 mixed block 的跨核握手：AIV0 写入 L1B 后通知 AIC，AIC 搬到 L0B 后释放 L1B。
// metadata 阶段先用事件 2 通知 AIC；AIC 消费该通知后再初始化 routed L1B，避免阶段复用冲突。
// AIC/AIV0: MTE3 -> MTE1 full, MTE1 -> MTE3 empty
// Metadata uses flag 2 before routed L1B stages 0–3
// 同block AIV0向AIC发布路由块总数就绪；此事件阶段性复用L1B的事件资源。
constexpr IntraBlockEvent<0> kMetadataFull{2};
// 跨核L1B生产/消费握手，Full由AIV0生产，Empty由AIC消费完后归还。
constexpr IntraBlockEvent<0> kL1BFull{0}, kL1BEmpty{0};

// GMM1：FIX 将 BF16 写 UB -> AIV1 执行激活/量化并写 GM -> 归还该 UB stage。
// GMM2：FIX 直接写 BF16 GM -> AIV1 等待完成并归还 credit -> 发布 Linear2Ready。
// AIC/AIV1: FIX -> V/S full, MTE3/S -> FIX empty
// Linear1 and Linear2 share the same epilogue credits
// GMM1保护两份FIX UB输出stage；GMM2保护FIX GM输出的完成通知，读者同为AIV1。
constexpr IntraBlockEvent<1> kEpilogueFull{0}, kEpilogueEmpty{0};

// Block sync: both AIVs' MTE3 -> AIC Scalar after final output stores
// 两份AIV最终y store完成后共同通知AIC，防止AIC提前结束本mixed block。
constexpr uint32_t kCombineDoneID = 4;

// CANN 9.2 forwards mode 12 but does not name write-through-share in its C API enum
// 选择当前CANN API未命名的write-through-share模式；这是cache策略，不是软件ready计数。
constexpr auto kWriteThroughShare = static_cast<asc_store_l2_cache_mode>(12);

template <const auto& workspace>
struct CombineReduceState {
    // 一份uint32 bitmap记录本AIV拥有的32个token完成状态。
    static constexpr uint32_t kNumBitsPerWord = sizeof(uint32_t) * 8;
    // stage尚未发初始Empty credit时用全1标识，与真实0/1槽号区分。
    static constexpr uint32_t kUninitializedStage = ~0u;

    // Resume at a channel boundary
    // 保存下一token、下一hidden channel和下一UBstage，支持尾段每次只做少量chunk后恢复。
    uint32_t token_idx, channel_idx = 0, stage_idx = kUninitializedStage;
    // 本AIV尚未完成整条hidden向量的token数，不是尚未检查的route数。
    uint32_t num_remaining_tokens;

    // valid_topk_mask 只表示当前被归并 token 的有效 route；completed_blocks 缓存各 producer 单调前缀。
    // reduced_token_mask 则按本 AIV 拥有的 token 集合保留已完成位，两种 mask 生命周期不同。
    uint32_t valid_topk_mask = 0;
    // 共享输出只需确认一次全部共享活跃核完成，之后本AIV所有token均可使用。
    bool shared_output_ready = false;
    // 记录最后阻挡当前token的producer rank，仅供超时打印，不改变任务owner。
    uint32_t blocking_rank_idx = 0;
    // 每peer单调prefix本地缓存，减少重复读取热GM信号。
    uint64_t completed_blocks[workspace.config.num_ranks]{};

    // One bit per owned token, strided by the AIV count
    // 每核分到约Tmax/64 token，再按32bit一word压缩，空间随注册容量而非活跃T。
    uint32_t reduced_token_mask[ceil_div(workspace.config.num_max_tokens_per_rank, kNumAIVs * kNumBitsPerWord)]{};
};

// Describe staged UB scratch without allocating memory
template <const auto& workspace>
__aicore__ constexpr auto make_reduce_buffer(uint32_t addr = 0) {
    constexpr auto& config = workspace.config;
    // 归并最多读取topk个routed结果，加一份已合并shared结果。
    constexpr uint32_t kNumInputs = config.num_topk + (config.num_shared_experts > 0);
    // BF16向量的元素数作为chunk对齐单位，后面的VF不需要额外channel尾mask。
    constexpr uint32_t kNumChannelsPerVector = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);
    // 剩余 UB 分给两份 reduce stage，每 stage 预留 top-k（加 shared 槽）个输入行片段。
    // channel 宽度向量对齐并不超过 H；stage stride 按预留输入数计算，有效输入少也不改变槽间距。
    const uint32_t num_channels = (UBSizeBytes - addr) / (kNumReduceStages * kNumInputs * sizeof(bfloat16_t)) /
        // 先除完整stage所需字节，再向下对齐至整BF16向量，防止UB尾部越界。
        kNumChannelsPerVector * kNumChannelsPerVector;
    // 返回只是UB视图描述，不执行分配；实际宽度取min(H,可用channel数)。
    return ub_ptr<bfloat16_t>(kNumInputs, config.hidden < num_channels ? config.hidden : num_channels, addr);
}

// True when all tokens are processed; each step issues one output chunk
template <const auto& workspace, bool kOverlapReduce>
__aicore__ inline bool advance_combine_reduce(__gm__ uint8_t* buffer, __gm__ bfloat16_t* output,
                                      uint32_t num_tokens, CombineReduceState<workspace>& state,
                                      const ub_ptr<bfloat16_t>& ub_reduce_inputs, uint32_t num_steps = ~0u) {
    constexpr auto& config = workspace.config;
    // 共享完成计数和routed GMM2块完成均需要知道每块H/256个输出任务。
    constexpr uint32_t kNumLinear2NBlocks = config.hidden / BLOCK_N;
    // 专家ID到producer rank的映射采用全局专家连续分区。
    constexpr uint32_t kNumExpertsPerRank = config.num_experts / config.num_ranks;
    // 无shared时编译器消除共享输入检查与额外reduce槽。
    constexpr bool kHasSharedExperts = config.num_shared_experts > 0;

    // 一个reduce小步实际处理的hidden列数上限，受当前借用UB剩余空间限制。
    const uint32_t num_chunk_channels = ub_reduce_inputs.shape_n;

    // 本rank接收所有top-k结果的GM区域，按[route slot][Tmax][H]存放BF16。
    const auto combine_input = workspace.combine_input.get_ptr(buffer);
    // 以无符号视图读取路由ID，负数自然成为越界专家而被跳过。
    const auto topk_idx = workspace.input.topk_idx.template get_ptr<uint64_t>(buffer);
    // 引用状态中的stage索引，修改会保留给下一次小步调用。
    auto& stage_idx = state.stage_idx;

    // Wait for all active shared Linear2 cores
    // 仅overlap模式需要在归并过程中探测shared就绪；普通路径前面的rank barrier已完成。
    if (kOverlapReduce and kHasSharedExperts and state.num_remaining_tokens > 0 and not state.shared_output_ready) {
        // shared Linear2 的 slot 0 统计完成全部 shared 任务的活跃核；目标是 min(shared task 总数,32)。
        // 它不是 routed 的 N-block 计数，不能用 H/256 作为此处的等待目标。
        state.shared_output_ready = comm::is_block_ready<workspace>(buffer, layout::kLinear2Ready, 0,
            min(ceil_div(num_tokens, BLOCK_M) * kNumLinear2NBlocks, kNumAICores));
        // 共享尚未完成时立即返回，将AIV执行机会留给其他阶段，避免读未写完的shared槽。
        if (not state.shared_output_ready)
            return false;
    }

    // 处理本核剩余token，未就绪token可以轮转跳过，已完成token由bitmap跳过。
    while (state.num_remaining_tokens > 0) {
        // 当前候选token是原输入token编号，不是dispatch置换后的专家行号。
        const uint32_t token_idx = state.token_idx;
        // overlap 模式每个 AIV 按步长 64 分配 token，绕回该核的首 token 继续探测。
        // 不就绪时换一个 token；bitmap 保证已归并 token 不会被第二次写出。
        const uint32_t next_token_idx = not kOverlapReduce ? token_idx + 1 :
            // overlap按64步长前进并绕回本AIV首token；普通路径按连续区间+1顺序推进。
            token_idx + kNumAIVs < num_tokens ? token_idx + kNumAIVs : token_idx % kNumAIVs;
        // 本核拥有 token=aiv_idx+n*64；n/32 定位 bitmap word，n%32 定位 bit，避免为每核分配全 T bitmap。
        const uint32_t word_idx = token_idx / (kNumAIVs * state.kNumBitsPerWord);
        // token/64是本核拥有集合中的序号，再取mod32定位完成bit。
        const uint32_t token_bit = 1u << (token_idx / kNumAIVs % state.kNumBitsPerWord);

        // Skip completed tokens
        // 只在overlap模式使用bitmap，普通顺序归并无需重复扫描已完成token。
        if (kOverlapReduce and (state.reduced_token_mask[word_idx] & token_bit)) {
            // 已归并token仅前进游标，绝不重复发y写回。
            state.token_idx = next_token_idx;
            continue;
        }

        // 早归并必须逐 route 核对实际 producer block：dispatch 做过行置换，不能从 token 顺序推断。
        // prefix 是已完成回传的块数（右开边界）；只有该 token 的所有有效 route 都落在 prefix 内才可读取。
        // Check the actual producer block for each route before the first chunk
        if (kOverlapReduce and state.channel_idx == 0) {
            // 从该 token 的第一个 channel chunk 开始建立完整 valid mask；中间 chunk 恢复时沿用同一 mask。
            state.valid_topk_mask = 0;
            // 遍历当前token原top-k槽，按真实route检查返回来源。
            for (uint32_t topk_slot_idx = 0; topk_slot_idx < config.num_topk; ++ topk_slot_idx) {
                // route扁平编号与metadata构建/route_m_blocks写回使用同一token*topk+j公式。
                const uint32_t token_topk_idx = token_idx * config.num_topk + topk_slot_idx;
                // 读取专家ID以确定负责回传的rank。
                const uint64_t expert_idx = asc_load_dev(topk_idx + token_topk_idx);
                // 非法/负ID不要求对应返回结果，它们不会占有效reduce输入。
                if (expert_idx >= config.num_experts)
                    continue;
                // 该route唯一producer是专家owner rank，不是源rank。
                const uint32_t peer_idx = expert_idx / kNumExpertsPerRank;
                // 初值 ~0u 表示专家端尚未公布这条 route 的 producer block；有限 prefix 不会放行它。
                // 只有所有有效 route 的实际 block 都位于对应 producer 的完成前缀内，才允许归并当前 token。
                const uint32_t pool_block_idx = asc_load_dev(workspace.route_m_blocks.get_ptr(buffer, token_topk_idx));
                // Completed prefixes only advance; refresh a peer when its cached value is insufficient
                // 引用该producer在本AIV缓存的完成前缀，只在不足时刷新。
                auto& completed_blocks = state.completed_blocks[peer_idx];
                // prefix是右开边界，所以块号等于prefix仍未完成。
                if (pool_block_idx >= completed_blocks) {
                    // cached prefix 不足才重读远端更新的本地槽；prefix 只前进，不逐 token 重复读取所有 peer 状态。
                    completed_blocks = asc_load_dev(workspace.metadata.combine_ready_prefix.get_ptr(buffer, peer_idx));
                    // 刷新后仍不足则说明当前route结果不可用，不能仅凭其他route已到达就提前求和。
                    if (pool_block_idx >= completed_blocks) {
                        // 保留阻塞peer以便最终等待超时指出具体来源。
                        state.blocking_rank_idx = peer_idx;
                        // 让下一次调用改探测其他token，避免一个慢token挡住全部已就绪输出。
                        state.token_idx = next_token_idx;
                        // 当前小步没有归并该token的任何channel，外层仍可继续通信/计算。
                        return false;
                    }
                }
                // 只有该route已确认返回才置有效bit，后面chunk据此压紧有效输入。
                state.valid_topk_mask |= 1u << topk_slot_idx;
            }
        }

        // Initialize UB credits on first use
        // 首次使用当前reduce UB布局时初始化两个槽的可复用credit。
        if (stage_idx == state.kUninitializedStage) {
            // 初始两槽都没有旧store，直接标空供MTE2第一次写入。
            notify<PIPE_MTE3, PIPE_MTE2, kNumReduceStages>(kReduceEmpty);
            // 从reduce第0槽开始；后续每chunk按两槽轮转。
            stage_idx = 0;
        }

        // 从状态中保存的channel断点继续，避免每次调用重新处理整H。
        for (uint32_t channel_base = state.channel_idx; channel_base < config.hidden; channel_base += num_chunk_channels) {
            // 最后chunk取H剩余列数，且H/UB宽度配置保证向量对齐要求。
            const uint32_t num_channels = min(config.hidden - channel_base, num_chunk_channels);
            // 复用 reduce stage 前等上一轮最终输出 MTE3 完成；随后 MTE2 读取各 route 的同一 channel 片段。
            wait<PIPE_MTE3, PIPE_MTE2>(kReduceEmpty, stage_idx);

            // Pack valid routes, then shared output; keep the reserved stage stride
            // 非法专家 route 不读取旧接收槽；有效 route 在 UB 内压紧，shared 输出作为最后一个输入。
            // FP32 sum 输入 stride 是本次 num_channels，UB stage 之间仍保留最大 stride。
            uint32_t num_inputs = 0;
            // 最后额外的shared槽不走topk合法性过滤，其整体就绪由shared_output_ready保护。
            for (uint32_t topk_slot_idx = 0; topk_slot_idx < config.num_topk + kHasSharedExperts; ++ topk_slot_idx) {
                // routed槽按overlap的valid mask或普通路径的原始专家合法性决定是否跳过。
                if (topk_slot_idx < config.num_topk and (kOverlapReduce ? (state.valid_topk_mask & (1u << topk_slot_idx)) == 0 :
                        asc_load_dev(topk_idx + static_cast<uint64_t>(token_idx) * config.num_topk + topk_slot_idx) >= config.num_experts))
                    continue;
                // 以原top-k槽和原token定位回传行，再加当前channel起点。
                const auto src = combine_input + workspace.get_combine_m_idx(topk_slot_idx, token_idx) * config.hidden + channel_base;
                // GM->UB按有效输入压紧；num_inputs后增，下一条route紧接前一条channel片段。
                copy_gm_to_ub(ub_reduce_inputs[stage_idx].ptr() + num_inputs++ * num_channels,
                              // 参数为1行、num_channels列、源行byte跨度；本次reduce数据不作长期L2驻留。
                              src, 1, num_channels, num_channels * sizeof(bfloat16_t), asc_load_l2_cache_mode::NOTALLOC_KEEP);
            }
            // 所有有效route与shared的DMA读都排在此通知之前，一个Full覆盖整个chunk输入。
            notify<PIPE_MTE2, PIPE_V>(kReduceFull, stage_idx);

            // Accumulate in FP32, then round once to BF16
            // Vector必须等当前stage的全部GM读取完成再访问UB。
            wait<PIPE_MTE2, PIPE_V>(kReduceFull, stage_idx);
            // 等待本次输入 DMA 后在原 UB 中求和，结果覆盖第一输入行；MTE3 写输出后才能释放该 stage。
            vf_combine_reduce(ub_reduce_inputs[stage_idx].ptr(), num_inputs, num_channels);
            // Vector结果可读通知先于MTE3最终输出写，保护原地覆盖后的第0输入行。
            notify<PIPE_V, PIPE_MTE3>(kReduceOutputFull, stage_idx);

            // Store BF16 output
            // MTE3等Vector输出完成，不能仅等输入DMA完成就直接写y。
            wait<PIPE_V, PIPE_MTE3>(kReduceOutputFull, stage_idx);
            // 输出按原token的[T,H]布局定位，dispatch置换在此完全消除。
            copy_ub_to_gm(output + static_cast<uint64_t>(token_idx) * config.hidden + channel_base,
                          // 源为当前reduce stage第0行，复制一条channel chunk。
                          ub_reduce_inputs[stage_idx].ptr(), 1, num_channels,
                          // 最终y采用write-through-share写出，列字节数为BF16的2*num_channels。
                          num_channels * sizeof(bfloat16_t), kWriteThroughShare);
            // 最终输出store后归还stage，下一次GM读取不能覆盖仍被MTE3读取的UB。
            notify<PIPE_MTE3, PIPE_MTE2>(kReduceEmpty, stage_idx);
            // 下个channel chunk切到另一个reduce槽。
            stage_idx = (stage_idx + 1) % kNumReduceStages;

            // Save the resume position
            // token 可能分多个 channel chunk；记录下一 chunk 边界，避免让出流水后重读或重写已完成片段。
            state.channel_idx = channel_base + num_channels;
            // 只有整H都写出才把当前token计为归并完成。
            if (state.channel_idx == config.hidden) {
                // 下个token从channel0重新开始合法route与prefix检查。
                state.channel_idx = 0;
                // 完成当前token后切到下一候选token，不保留其route掩码给新token。
                state.token_idx = next_token_idx;
                // 普通路径不维护完成bitmap，只有overlap轮转路径需要置位。
                if constexpr (kOverlapReduce)
                    // 记录该token完成；之后即使轮转回同token，也不会二次归并。
                    state.reduced_token_mask[word_idx] |= token_bit;
                // 未完成token数减一，与num_steps的chunk预算是两种不同计数。
                -- state.num_remaining_tokens;
            }

            // 每发一个channel chunk耗费一个小步预算，减到0立即把执行权交回上层。
            if (-- num_steps == 0)
                // 预算用尽时仍区分全部完成与仅本次小步结束。
                return state.num_remaining_tokens == 0;
        }
    }
    // 剩余token数为0时返回true，外层wait_until可结束。
    return true;
}

template <const auto& workspace, bool kOverlapReduce>
__aicore__ __forceinline__ void combine_reduce(__gm__ uint8_t* buffer, __gm__ bfloat16_t* output,
                                              uint32_t num_tokens, uint32_t rank_idx, uint32_t aiv_idx,
                                              CombineReduceState<workspace>& reduce, const layout::SymBuffer<>& sym_buffer) {
    constexpr auto& config = workspace.config;

    // 小容量路径用跨 rank barrier 确认回传完成；大容量路径用各 peer 的单调 prefix 判断 token 就绪。
    // 最后扩大 reduce 的 UB 使用范围前，先排空原来的通信/归并事件，避免覆盖仍在使用的槽。
    // Drain communication before reusing UB
    // overlap路径用prefix证明逐token远端就绪，先仅排空本核MTE3；普通路径执行跨rank结束屏障。
    if constexpr (kOverlapReduce) {
        // 在当前MTE3通信队列尾发一个到Scalar的完成通知。
        notify<PIPE_MTE3, PIPE_S>(kCommDone);
        // 等到该通知，保证随后用更大UB布局前不再有旧通信读取scratch。
        wait<PIPE_MTE3, PIPE_S>(kCommDone);
    } else {
        // 使用第二组rank barrier完成全卡回传汇合，然后可一次性顺序reduce。
        comm::rank_barrier<workspace, 1>(buffer, sym_buffer, aiv_idx);
    }

    // Drain reduce stages before changing chunk width
    // 先前尾段reduce可能留有y store在途；若从未用过则无需等旧reduce credit。
    if (reduce.stage_idx != reduce.kUninitializedStage)
        // 同时消费两份reduce Empty，确保旧布局两个stage都不再被MTE3读取。
        wait<PIPE_MTE3, PIPE_MTE2, kNumReduceStages>(kReduceEmpty);
    // 布局变宽后需重新初始化Empty，不能保留旧stage索引和旧stage跨度。
    reduce.stage_idx = reduce.kUninitializedStage;

    // Reduce remaining tokens
    // After draining, both AIVs reuse UB from offset 0 for two reduce stages
    // 从UB地址0借整个剩余空间归并，比通信/权重路径尾部借用的chunk更宽。
    constexpr auto ub_reduce_inputs = make_reduce_buffer<workspace>();
    comm::wait_until([&]() __aicore__ {
        // 持续推进所有尚未完成token，直到返回true；overlap路径仍按producer prefix逐token检查。
        return advance_combine_reduce<workspace, kOverlapReduce>(buffer, output, num_tokens, reduce, ub_reduce_inputs);
    }, [&]() __aicore__ {
        // 超时若shared尚未就绪，优先报告本地shared依赖，而非无关peer。
        if (config.num_shared_experts > 0 and not reduce.shared_output_ready) {
            comm::print_local_timeout(rank_idx);
            return;
        }
        // This is the last observed blocker; token_idx already points to the next probe
        // 否则报告最后一次阻挡token的producer rank及当前prefix。
        const uint32_t peer_idx = reduce.blocking_rank_idx;
        comm::print_peer_timeout(rank_idx, aiv_idx, peer_idx,
            asc_load_dev(workspace.metadata.combine_ready_prefix.get_ptr(buffer, peer_idx)), false);
    });

    // 本核 token 已全部归并不等于所有 peer 停止写：空 rank 也必须等每个 peer 的 all-ones 完成标记。
    // 该收尾保护下一次调用的输入/metadata/progress 复用，不能因本地无输出而跳过。
    // Empty ranks also wait for peer writes before the next invocation reuses scratch
    if constexpr (kOverlapReduce) {
        // 64个AIV分摊peer最终完成检查，空输出rank也不能省略这些检查。
        for (uint32_t peer_idx = aiv_idx; peer_idx < config.num_ranks; peer_idx += kNumAIVs) {
            // 每producer在本rank有一份独立完成prefix地址。
            const auto prefix_ptr = workspace.metadata.combine_ready_prefix.get_ptr(buffer, peer_idx);
            // 全1表示producer本轮全部写结束；有限prefix只证明此前一段数据结束。
            comm::wait_until([&]() __aicore__ { return asc_load_dev(prefix_ptr) == ~uint64_t{0}; }, [&]() __aicore__ {
                comm::print_peer_timeout(rank_idx, aiv_idx, peer_idx, asc_load_dev(prefix_ptr), true);
            });
        }
    }
}

// AIV0: FP4 weights -> FP8 NZ -> L1B
template <const auto& workspace, bool kOverlapReduce>
__aicore__ inline void aiv0_weight_pipeline(__gm__ uint8_t* buffer, __gm__ bfloat16_t* output,
                                            uint32_t num_tokens, CombineReduceState<workspace>& reduce, uint32_t num_total_m_blocks,
                                            const layout::Weights& weights, uint32_t core_idx) {
    constexpr auto& config = workspace.config;
    // GMM1每个物理N256任务产出N128激活，所以每逻辑块需要I/128份权重任务。
    constexpr uint32_t kNumLinear1NBlocks = config.intermediate_hidden / (BLOCK_N / 2);
    // GMM2每任务输出N256，任务数为H/256。
    constexpr uint32_t kNumLinear2NBlocks = config.hidden / BLOCK_N;
    // 权重路径不分配ring，但借该容量判断是否启用尾段reduce插空。
    constexpr uint32_t kNumRingBlocks = workspace.get_num_ring_blocks();

    // metadata VF 已结束，原 UB scratch 可改作权重流水；不是同时占用两份 UB。
    // 同一 GEMM 任务内：GM FP4 -> UB -> Vector 展开/NZ -> MTE3 写 L1B -> AIC MTE1 消费。
    // Empty/Full credit 保护槽所有权；L1B 的回收来自 AIC，不能只等本核 UB 搬运完成。
    // AIV0 UB after metadata (KiB)
    // Region   Offset  Stages x size
    // FP4           0      3 x 32
    // FP8 NZ       96      2 x 64.5
    // Reduce      225      2 stages within the remaining 31 KiB
    // FP4 staging 用 3*32KiB，展开后的 NZ 用 2*(64KiB+padding)；余下 UB 才可插入尾段 reduce。
    constexpr uint32_t kNumFP4Stages = 3, kNumNZStages = 2;
    // 三份packed FP4槽从UB0起，每份容量为64KiB L0B的一半，即32KiB。
    constexpr ub_ptr<uint8_t> ub_fp4(1, L0BSizeBytes / 2);
    // 两份NZ槽起于packed槽之后；每份64KiB再加M128 K512对应的padding空间。
    constexpr ub_ptr<uint8_t> ub_nz(1, L0BSizeBytes + MMAConfig<128>::BLOCK_K, ub_fp4.offset(kNumFP4Stages));
    // 权重生产期间只允许在NZ槽之后的剩余UB放reduce输入，不能覆盖在途权重。
    constexpr auto ub_reduce_inputs = make_reduce_buffer<workspace>(ub_nz.offset(kNumNZStages));

    // 静态检查NZ末尾与reduce至少一整向量都能放入UB。
    static_assert(ub_nz.offset(kNumNZStages) <= UBSizeBytes and ub_reduce_inputs.shape_n > 0, "AIV0 UB overflow");

    // Initialize UB reuse credits; AIC releases the initial L1B credits
    // 第一次packed权重搬运前，Vector不存在旧输入，预置三份Empty。
    notify<PIPE_V, PIPE_MTE2, kNumFP4Stages>(kFP4Empty);
    // 第一次NZ转换前，MTE3不存在旧输出，预置两份Empty。
    notify<PIPE_MTE3, PIPE_V, kNumNZStages>(kNZEmpty);

    // Size the reduce window from local output tokens
    // 本地T对应的块数用作尾段窗口，不等于远端收到的专家块数。
    const uint32_t num_reduce_window_m_blocks = ceil_div(num_tokens, BLOCK_M);
    // 权重生产者私有调度游标，重放与AIC相同的routed任务顺序。
    sched::LinearTaskState schedule;
    // 两层各有独立专家游标，因为GMM1/GMM2的logical M推进速度不同。
    sched::ExpertCursor<workspace> expert_cursors[2]{};
    // packed/NZ槽均从0开始，后续按各自容量取模。
    uint32_t fp4_stage_idx = 0, nz_stage_idx = 0;
    // routed L1B stage从0开始，AIC结束shared阶段后为其初始化credit。
    uint32_t l1b_stage_idx = 0;

    while (true) {
        // 由调度器返回本次权重属于GMM1还是GMM2。
        sched::BlockPhase block_phase;
        // AIC handles shared weights; zero shared tokens skips those tasks
        // 传 num_shared_tokens=0 只重放 routed 任务；shared 权重已经是 FP8，由 AIC 自行装入 L1。
        const auto task_idx = sched::take_linear_task<workspace>(0, num_total_m_blocks, schedule, core_idx, block_phase);
        // routed序列耗尽即可结束权重生产；AIV仍要执行后面的最终reduce。
        if (block_phase == sched::BlockPhase::kNone)
            break;

        // Select task weights
        // 当前权重任务仅routed，故只需区分Linear1/Linear2。
        const bool is_linear1 = block_phase == sched::BlockPhase::kLinear1;
        // 选择当前层每M-block的N任务数，供task解码。
        const uint32_t num_n_blocks = is_linear1 ? kNumLinear1NBlocks : kNumLinear2NBlocks;
        // GMM1归约K=H，GMM2归约K=I，对应读取不同形状的packed权重。
        const uint32_t num_k = is_linear1 ? config.hidden : config.intermediate_hidden;
        // task 按 M-block 再 N-block 展平，两层 N-block 数不同，用各自 num_n_blocks 解码。
        const uint32_t pool_block_idx = task_idx / num_n_blocks;
        // task余数确定当前专家的N256权重块，不是激活输出N128列号之外的另一个维度。
        const uint32_t n_block_idx = task_idx % num_n_blocks;
        // 层号0/1索引两个独立ExpertCursor。
        const uint32_t layer_idx = is_linear1 ? 0 : 1;
        // 根据logical块找到本地专家，并取得均衡切分后的有效M用于选择MAD形状。
        const uint32_t valid_m = expert_cursors[layer_idx].advance_to(buffer, pool_block_idx);
        // 同专家有多个 M-block 时偏向保留权重的 L2 局部性；这里仍逐任务搬运、展开权重，
        // 不能把缓存 hint 理解成跨 M-block 的 L1B 常驻或显式跳过重复加载。
        const auto weight_l2_hint = expert_cursors[layer_idx].num_expert_tokens > BLOCK_M ?
            // 同专家多个M块可能重复读相同权重，cache hint倾向保留；单块则避免长期占L2。
            asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP;
        // expert*Nblocks+n_block定位本层专家权重tile序号。
        const uint64_t weight_block_idx =
            // 先跨完整专家N-block，再跨当前K；每个tile存N256行、每行K/2 packed字节。
            static_cast<uint64_t>(expert_cursors[layer_idx].local_expert_idx) * num_n_blocks + n_block_idx;
        // 权重按 [expert][N256][K/2 packed bytes] 存储；每字节两份 FP4，offset 的 K 维以字节计。
        const auto weight_data = (is_linear1 ? weights.l1.data : weights.l2.data) +
                                 weight_block_idx * BLOCK_N * (num_k / 2);

        // 内部模板保证权重展开和AIC消费采用相同M128/M256对应的K/N切分。
        const auto load_weights = [&]<uint32_t MAD_M>() __aicore__ {
            // 从当前MAD_M推导权重每次准备的K宽。
            constexpr uint32_t BLOCK_K = MMAConfig<MAD_M>::BLOCK_K;
            // 从同一配置推导N子块宽，M128要准备两个N128子块。
            constexpr uint32_t MAD_N = MMAConfig<MAD_M>::MAD_N;
            // UB中每行packed FP4目标列宽，以字节计为K块宽的一半。
            constexpr uint32_t kNumPackedCols = BLOCK_K / 2;
            // 展开后NZ按FRAC_K分组，复制到L1时以该组数组织数据。
            constexpr uint32_t kNumColGroups = BLOCK_K / get_frac_k<uint8_t>();
            // L1B基址在3份L1A之后，与AIC建立的L1布局严格一致。
            constexpr l1_ptr<float8_e4m3_t, Major::K> l1b(MAD_N, BLOCK_K, kNumL1AStages * L0ASizeBytes);

            // 按完整归约K遍历当前权重tile，每次生产一组L1B子块。
            for (uint32_t k_idx = 0; k_idx < num_k; k_idx += BLOCK_K) {
                // valid_k remains a multiple of MAD_K, including the final K block
                // K尾块只读取真实valid_k，完整模板BLOCK_K仍决定UB/L1槽stride。
                const uint32_t valid_k = min(BLOCK_K, num_k - k_idx);
                // 沿当前物理N256任务分为MAD_N宽的一个或两个权重子块。
                for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N) {
                    // Load packed FP4
                    // Vector 已读完该 packed FP4 槽后，MTE2 才能填新 K/N 子块；源行 stride 为完整专家 K/2。
                    wait<PIPE_V, PIPE_MTE2>(kFP4Empty, fp4_stage_idx);
                    // GM->UB仅加载packed数据；scale由AIC独立读取，不在这里重复拉取。
                    copy_gm_to_ub(ub_fp4[fp4_stage_idx],
                                  // 源起点=N子块偏移*K/2+K块偏移/2，二者都以原FP4字节为单位。
                                  const_cast<__gm__ uint8_t*>(weight_data + static_cast<uint64_t>(n_mad_idx) * (num_k / 2) + k_idx / 2),
                                  // 有效shape为MAD_N×valid_k/2；GM行stride=完整K/2，UB行stride=BLOCK_K/2。
                                  MAD_N, valid_k / 2, num_k / 2, kNumPackedCols, weight_l2_hint);
                    // MTE2读完当前packed槽，Vector才能消费。
                    notify<PIPE_MTE2, PIPE_V>(kFP4Full, fp4_stage_idx);

                    // Dequantize to FP8 NZ
                    // Vector确认FP4读取结束后才调用位展开VF。
                    wait<PIPE_MTE2, PIPE_V>(kFP4Full, fp4_stage_idx);
                    // NZ 槽要等上一轮 UB->L1 读完才能覆盖；packed FP4 与 NZ 两组独立轮转，分别回收。
                    wait<PIPE_MTE3, PIPE_V>(kNZEmpty, nz_stage_idx);
                    // 把packed FP4展开为FP8并直接排NZ，模板使用完整BLOCK_K形状含尾块padding。
                    vf_fp4_to_fp8_nz<MAD_N, BLOCK_K>(ub_fp4[fp4_stage_idx].ptr(), ub_nz[nz_stage_idx].ptr());
                    // 展开已消费完原 FP4 槽，可提早归还给下一轮 MTE2；NZ 输出仍由 MTE3 占用。
                    notify<PIPE_V, PIPE_MTE2>(kFP4Empty, fp4_stage_idx);
                    // 向MTE3发布NZ转换结果；packed槽已提前返还。
                    notify<PIPE_V, PIPE_MTE3>(kNZFull, nz_stage_idx);

                    // Publish L1B to AIC
                    // MTE3先等本核Vector生产完NZ，随后还要等远端AIC释放L1B。
                    wait<PIPE_V, PIPE_MTE3>(kNZFull, nz_stage_idx);
                    // 等待 AIC MTE1 释放目标 L1B 后才能写入；即使 NZ 已就绪也不能覆盖 AIC 尚未读取的权重。
                    wait_intra_block<PIPE_MTE3, 0>(kL1BEmpty, l1b_stage_idx);
                    // 源 NZ UB 含额外 padding，复制指令按 kNumColGroups/MAD_N 跳过 padding，L1 仅保存真实矩阵块。
                    // 它只改变局部布局，不改变全局 packed FP4 的 expert/N/K 排列。
                    asc_copy_ub2l1(l1b[l1b_stage_idx].template ptr<uint8_t>(), ub_nz[nz_stage_idx].ptr(), kNumColGroups, MAD_N, 1, 0);
                    // 通知保护的是该 L1B N/K 子块；AIC 只在收到 Full 后搬入 L0B。随后本核可以释放 NZ UB。
                    notify_intra_block<PIPE_MTE3, 0>(kL1BFull, l1b_stage_idx);
                    // L1B写完不再读当前NZ槽，向Vector归还该槽。
                    notify<PIPE_MTE3, PIPE_V>(kNZEmpty, nz_stage_idx);

                    // 三个 stage index 分别为 FP4(3槽)、NZ(2槽)、L1B(4槽)；不要要求它们同余或同时回收。
                    fp4_stage_idx = (fp4_stage_idx + 1) % kNumFP4Stages;
                    // NZ两个stage按0/1轮转。
                    nz_stage_idx = (nz_stage_idx + 1) % kNumNZStages;
                    // L1B四个stage按0..3轮转，AIC按相同子块顺序消费。
                    l1b_stage_idx = (l1b_stage_idx + 1) % kNumL1BStages;
                }

                // Reduce one chunk in unused UB
                // 仅大于ring容量且进入尾段才借UB插空reduce，避免常规权重生产路径承担轮询开销。
                if (kOverlapReduce and num_total_m_blocks > kNumRingBlocks and pool_block_idx + num_reduce_window_m_blocks >= num_total_m_blocks)
                    // 一次最多一个输出channel chunk，处理后继续生产下一K块权重。
                    advance_combine_reduce<workspace, kOverlapReduce>(buffer, output, num_tokens, reduce, ub_reduce_inputs, 1);
            }
        };
        // 依据当前专家逻辑块有效行数选择与AIC一致的两套权重布局。
        if (valid_m <= 128)
            // M<=128时使用N128/K512子块展开路径。
            load_weights.template operator()<128>();
        else
            // M>128时使用N256/K256子块展开路径。
            load_weights.template operator()<256>();
    }
}

// AIC: shared and routed GEMMs
template <const auto& workspace>
__aicore__ inline void aic_gemm_pipeline(__gm__ uint8_t* buffer, uint32_t num_tokens, uint32_t rank_idx,
                                         const layout::Weights& weights, uint32_t core_idx) {
    constexpr auto& config = workspace.config;
    constexpr uint32_t kNumRingBlocks = workspace.get_num_ring_blocks();
    // shared权重已经FP8，可在AIC侧直接读取；无shared时删除相关初始化和收尾。
    constexpr bool kHasSharedExperts = config.num_shared_experts > 0;
    // 每个L1 scale chunk容纳32份int16 pair，覆盖K2048个元素。
    constexpr uint32_t kNumSFPairsPerChunk = 32;
    // 激活scale L1区shape [M256,32pair]，放在全部A/B数据区之后。
    constexpr l1_ptr<int16_t> l1_sfa(BLOCK_M, kNumSFPairsPerChunk,
                                   // 字节偏移=3*L0A容量+4*L0B容量，即448KiB。
                                   kNumL1AStages * L0ASizeBytes + kNumL1BStages * L0BSizeBytes);
    // B scale区紧接两份A scale stage之后，shape [N256,32pair]。
    constexpr l1_ptr<int16_t> l1_sfb(BLOCK_N, kNumSFPairsPerChunk, l1_sfa.offset(kNumSFStages));
    // 描述AIV1 UB输出的M128×N256 BF16 FIX目标，AIC可跨核向该UB写结果。
    constexpr ub_ptr<bfloat16_t> ub_accum(STORE_BLOCK_M, BLOCK_N);
    // 两份B scale末尾不得超过L1总容量，避免数据/scale区重叠。
    static_assert(l1_sfb.offset(kNumSFStages) <= L1SizeBytes, "L1 overflow");

    // issued 序号只属于本 AIC 与它的 AIV1；各 core 的进度可不同，跨 core 的 M-block 输入用 progress 原子计数汇合。
    const auto num_issued_gemms_ptr = workspace.progress.num_issued_gemms.get_ptr(buffer, core_idx);
    // 实际shared M块数用于routed progress索引的运行时前缀，不使用Tmax推断当前块数。
    const uint32_t num_shared_m_blocks = kHasSharedExperts ? ceil_div(num_tokens, BLOCK_M) : 0;
    // 超过一个shared M块时可能复用权重，在cache hint中偏向保留。
    const bool reuse_shared_weights = num_shared_m_blocks > 1;
    // routed GMM2结果写到BF16 ring，后继由AIV1回传。
    const auto routed_output = workspace.routed.output.get_ptr(buffer);
    // shared GMM2结果直接写本rank combine额外槽，省去远端回传。
    const auto combine_input = workspace.combine_input.get_ptr(buffer);

    // 首次给 L0A/B、L1A 和 SF 槽发 Empty，表示尚无旧消费者；routed L1B 另由 shared 阶段结束后的 AIC 初始化。
    // Initialize reuse credits
    // 配置MMAD的N方向，匹配本函数N子块的累加/输出次序。
    asc_set_mmad_direction_n();
    // 预置两份L0A Empty，让MTE1开始第一次L1->L0A搬运。
    notify<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0AEmpty);
    // 预置两份L0B Empty，与A的credit独立管理。
    notify<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0BEmpty);
    // 三个L1A均未被MTE1占用，可以开始激活预取。
    notify<PIPE_MTE1, PIPE_MTE2, kNumL1AStages>(kL1AEmpty);
    // 两份scale stage初始可写；后续到chunk末再回收。
    notify<PIPE_MTE1, PIPE_MTE2, kNumSFStages>(kSFEmpty);
    if constexpr (kHasSharedExperts)
        // 共享B最多三份credit，实际stage数由当前MAD配置决定。
        notify<PIPE_MTE1, PIPE_MTE2, 3>(kSharedL1BEmpty);

    // AIC调度游标；必须与AIV0 routed权重和AIV1 epilogue选择相同任务。
    sched::LinearTaskState schedule;
    // GMM1/GMM2各自的专家位置状态，避免交错执行时相互覆盖。
    sched::ExpertCursor<workspace> expert_cursors[2]{};
    // 下一份L1A激活将使用的物理slot。
    uint32_t l1a_stage_idx = 0;
    // 下一份L1B权重消费slot，shared与routed阶段会显式交接。
    uint32_t l1b_stage_idx = 0;
    // 下一份A/B scale chunk共同使用的stage索引。
    uint32_t sf_stage_idx = 0;
    // uint8已提交序号与AIV1已消费序号配对，不直接承担跨rank完成证明。
    uint8_t num_issued_gemms = 0;
    // 标记是否已把共享B区交给routed AIV0，确保Empty只初始化一次。
    bool routed_l1b_initialized = false;
    // routed总块数初始未知；shared可独立在其到达前执行。
    uint32_t num_total_m_blocks = 0;  // Routed pool only; shared blocks are separate
    // metadata_ready是本核调度状态，不是任意某块dispatch已完成标志。
    bool metadata_ready = false;
    // 从AIV0汇总证书读取本rank最终routed块数。
    const auto num_m_blocks_certificate = workspace.metadata.num_m_blocks_certificate.get_ptr(buffer);

    while (true) {
        // 保存本次调度层和shared/routed身份。
        sched::BlockPhase block_phase;
        // 按静态序列取下一任务；未取得metadata时只可能先返回shared任务。
        const auto task_idx = sched::take_linear_task<workspace>(
            num_tokens, num_total_m_blocks, schedule, core_idx, block_phase, metadata_ready);
        if (block_phase == sched::BlockPhase::kNone) {
            // metadata已到但没有剩余task，AIC才可离开计算循环。
            if (metadata_ready)
                break;

            // shared 任务允许在 routed metadata 未就绪时先执行；无可执行任务时才等待 AIV0 的 metadata 事件。
            // 消费后读取 certificate，后续 kNone 才表示 routed 调度结束。
            // Consume metadata before routed L1B reuses the same intra-block event
            // 等待AIV0向本AIC发的元数据事件，先消费它再重用L1B相关事件资源。
            wait_intra_block<PIPE_S, 0>(kMetadataFull);
            // 仅取证书低24位块数，高位是AIV0到达数。
            num_total_m_blocks = asc_load_dev(num_m_blocks_certificate) & layout::kCountValueMask;
            // 标记后继routed任务可调度，实际payload仍由每M块ready保护。
            metadata_ready = true;
            // 重新进入调度器，使用更新后的routed总块数分配任务。
            continue;
        }

        // Decode the GEMM task
        // 共享两层均走独立输入/权重路径，但仍复用同一AIC MMAD资源。
        const bool is_shared = block_phase == sched::BlockPhase::kSharedLinear1 or block_phase == sched::BlockPhase::kSharedLinear2;
        // 两种Linear1都要FIX到UB并做SwiGLU，两种Linear2都FIX到GM。
        const bool is_linear1 = block_phase == sched::BlockPhase::kLinear1 or block_phase == sched::BlockPhase::kSharedLinear1;

        // Shared experts widen Linear1 N and Linear2 K
        // 共享路径把S个专家的I拼为S*I，routed保持单专家I。
        const uint32_t intermediate_hidden = is_shared ? config.shared_intermediate_hidden : config.intermediate_hidden;
        // 当前路径的GMM1每M块输出任务数，shared可能比routed更多。
        const uint32_t num_linear1_n_blocks = intermediate_hidden / (BLOCK_N / 2);
        // 按当前层选择解码task需要的N任务数。
        const uint32_t num_n_blocks = is_linear1 ? num_linear1_n_blocks : config.hidden / BLOCK_N;
        // 两层归约K分别为H和当前intermediate宽度。
        const uint32_t num_k = is_linear1 ? config.hidden : intermediate_hidden;
        // MX scale每64个K元素为一个int16 pair。
        const uint32_t num_sf_pairs = num_k / MX_SF_DIVISOR;
        // 线性任务号除当前层N任务数恢复逻辑M-block。
        const uint32_t pool_block_idx = task_idx / num_n_blocks;
        // 余数恢复N256物理tile序号。
        const uint32_t n_block_idx = task_idx % num_n_blocks;
        // 当前层索引，用于选择独立ExpertCursor。
        const uint32_t layer_idx = is_linear1 ? 0 : 1;
        // 共享块简单按本地T切256；routed块按专家行组均衡切分。
        const uint32_t valid_m = is_shared ? min(num_tokens - pool_block_idx * BLOCK_M, BLOCK_M) :
            expert_cursors[layer_idx].advance_to(buffer, pool_block_idx);
        // B scale与data访问策略分别配置；共享多M或routed多M时倾向保留scale。
        const auto weight_sf_l2_hint = (is_shared ? reuse_shared_weights : expert_cursors[layer_idx].num_expert_tokens > BLOCK_M) ?
            asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP;
        // MMAD 的 M 对齐到 FRAC_MN；逻辑有效行 valid_m 仍用于 FIX 写回，避免把对齐 padding 当真实 token。
        const uint32_t aligned_valid_m = aligned(valid_m, FRAC_MN);
        // shared无专家维，直接用N块号；routed需要先跨当前专家全部N块。
        const uint64_t weight_block_idx = is_shared ? n_block_idx :
            static_cast<uint64_t>(expert_cursors[layer_idx].local_expert_idx) * num_n_blocks + n_block_idx;
        // B scale 物理布局 [expert][N-block][K-pair][N256]；weight_block_idx 选择 expert/N tile，sf_offset 跨完整 K。
        const uint64_t sf_offset = weight_block_idx * num_sf_pairs * BLOCK_N;
        // 两层×shared/routed四组权重在此统一选择，后续data/scale地址来自同一组。
        const auto task_weights = is_linear1 ? (is_shared ? weights.shared_l1 : weights.l1) : (is_shared ? weights.shared_l2 : weights.l2);

        // Hand L1B over to AIV0 after shared tasks
        // 仅第一次进入routed时交接L1B；共享阶段已按调度器排在此前。
        if (not is_shared and not routed_l1b_initialized) {
            // 交接后routed从第0个L1B槽开始，和AIV0初值一致。
            l1b_stage_idx = 0;
            // 向AIV0归还全部4份routed L1B初始credit，它才可覆盖原共享B物理区域。
            notify_intra_block<PIPE_MTE1, 0, kNumL1BStages>(kL1BEmpty);
            // 避免后继每个routed任务再次发Empty导致过量credit。
            routed_l1b_initialized = true;
        }

        // pool_block_idx 是不回绕的逻辑块号，input_block_idx 才是有限 ring 的物理槽号。
        // shared 输入另有空间；routed Linear1/Linear2 分别读取各自独立的输入 ring。
        // Select input data and scales
        const uint32_t input_block_idx = is_shared ? pool_block_idx : layout::get_ring_block_idx<workspace>(pool_block_idx);
        // 按层选择数据源：GMM1读原/dispatch输入，GMM2读shared/routed量化激活。
        const auto input_layout = is_linear1 ?
            // shared GMM1直接读取本rank原量化输入；routed读取dispatch重排结果。
            (is_shared ? workspace.input.acts : workspace.routed.l1_acts) :
            // shared GMM2使用独立shared_l2_acts，不受routed ring回绕影响。
            (is_shared ? workspace.shared_l2_acts : workspace.routed.l2_acts);
        // M-block物理行起点始终乘256，即便valid_m小于256。
        const uint64_t input_m_idx = static_cast<uint64_t>(input_block_idx) * BLOCK_M;
        // 数据行stride为当前层完整K，块内第0行当前K起点由后续k_idx再偏移。
        const auto input = input_layout.data.get_ptr(buffer, input_m_idx * num_k);
        // scale起点先跨整M-block的pair数，后续按token-major/K-major视图读取。
        const auto input_sf = input_layout.sf.get_ptr(buffer, input_m_idx * num_sf_pairs);

        const auto compute = [&]<uint32_t MAD_M>() __aicore__ {
            // M128 使用较宽 K512 和 N128 MAD，M256 使用 K256 和 N256 MAD；各自的 L0 stage/累加布局配套。
            constexpr uint32_t BLOCK_K = MMAConfig<MAD_M>::BLOCK_K;
            // 每次MAD的归约K宽，是L1K块的一半，对应两份L0输入stage。
            constexpr uint32_t MAD_K = MMAConfig<MAD_M>::MAD_K;
            // 每次MAD的输出N宽，决定同任务需一份还是两份L0C累加区。
            constexpr uint32_t MAD_N = MMAConfig<MAD_M>::MAD_N;
            // 单N块时沿用A Full事件顺序；split-N时用独立B Full，避免不同MAD形状的credit冲突。
            constexpr auto l0_full = BLOCK_N == MAD_N ? kL0AFull : kL0BFull;
            // shared L1B stage数按完整N256的K块容量计算，不能直接用routed的4槽。
            constexpr uint32_t kNumSharedL1BStages = MMAConfig<MAD_M>::get_num_shared_l1b_stages();
            // 激活L1起点0；M128/K512与M256/K256每stage均占64KiB。
            constexpr l1_ptr<float8_e4m3_t, Major::K> l1a(MAD_M, BLOCK_K, 0);
            // routed权重L1紧跟3份激活，每stage也是64KiB，N子块数随MAD形状变化。
            constexpr l1_ptr<float8_e4m3_t, Major::K> l1b(MAD_N, BLOCK_K, l1a.offset(kNumL1AStages));
            // shared与routed权重共用L1地址区域，但shared每stage完整N256，物理跨度不同。
            constexpr l1_ptr<float8_e4m3_t, Major::K> shared_l1b(BLOCK_N, BLOCK_K, l1b.addr);
            // L0A每stage最多32KiB，两个stage复用64KiB。
            constexpr l0a_ptr<float8_e4m3_t> l0a(MAD_M, MAD_K);
            // L0B两个32KiB stage，按K/N子块序号轮转。
            constexpr l0b_ptr<float8_e4m3_t> l0b(MAD_N, MAD_K);
            // L0C存FP32；M128时有两份N128输出区，M256时一份完整N256输出区。
            constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);

            // 当前任务完整K循环，所有K块累计到同一组L0C，只有最后K才FIX结果。
            for (uint32_t k_idx = 0; k_idx < num_k; k_idx += BLOCK_K) {
                // valid_k remains a multiple of MAD_K, including the final K block
                // 当前L1 K块的有效列数，尾块不越过真实K；仍按完整BLOCK_K保留slot stride。
                const uint32_t valid_k = min(BLOCK_K, num_k - k_idx);
                // 每 int16 scale pair 覆盖 K64；SF 按 32 对一批驻留，即覆盖 K2048，多个 K-block 可共享同一批 scale。
                const uint32_t sf_pair_idx = k_idx / MX_SF_DIVISOR;
                // 最后一个scale chunk可能不足32对，用实际剩余pair数限制GM读取。
                const uint32_t num_chunk_pairs = min(kNumSFPairsPerChunk, num_sf_pairs - sf_pair_idx);

                // Load shared FP8 weights
                // shared权重已经FP8，不调用AIV0展开路径，AIC自己用MTE2读入L1。
                if (is_shared) {
                    // 等待上一轮共享B已被MTE1消费后才填该shared stage。
                    wait<PIPE_MTE1, PIPE_MTE2>(kSharedL1BEmpty, l1b_stage_idx);
                    // 当前共享N256 tile在权重GM中的完整起点，以FP8元素字节计。
                    const uint64_t weight_offset = weight_block_idx * BLOCK_N * num_k;
                    // 为这次GM->L1设置单次loop，避免沿用其他拷贝指令的循环配置。
                    asc_set_gm2l1_loop_size(1, 1);
                    // 一次把当前K块的完整N256 shared权重装入所选L1stage。
                    asc_copy_gm2l1_align(shared_l1b[l1b_stage_idx].template ptr<uint8_t>(),
                                         // shared transformed布局按K块前进，k_idx每步跨N256元素。
                                         const_cast<__gm__ uint8_t*>(task_weights.data + weight_offset + k_idx * BLOCK_N),
                                         // 有效复制长度为N256*valid_k字节，无左右padding请求。
                                         1, BLOCK_N * valid_k, 0, 0, false,
                                         // shared多M时对同权重复读，倾向L2保留；单M时不长期分配cache。
                                         reuse_shared_weights ? asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP,
                                         // GM/L1的loop跨度都按完整BLOCK_K*N256，尾块仅缩短有效长度。
                                         BLOCK_N * BLOCK_K, BLOCK_N * BLOCK_K);
                    // shared B数据读取完成后发布本核MTE2->MTE1通知。
                    notify<PIPE_MTE2, PIPE_MTE1>(kSharedL1BFull, l1b_stage_idx);
                }

                // Load weight scales once per SF chunk
                // 只有进入新 SF chunk 才加载权重 scale；在其最后一个 K-block 被搬到 L0 后才归还 SF stage。
                if (sf_pair_idx % kNumSFPairsPerChunk == 0) {
                    // 旧scale chunk已搬到L0后可覆盖，不必等所有对应MMAD完成。
                    wait<PIPE_MTE1, PIPE_MTE2>(kSFEmpty, sf_stage_idx);
                    // B scale从GM加载到当前L1 B scale stage。
                    copy_gm_to_l1_mx(l1_sfb[sf_stage_idx],
                                     // GM B scale按[Kpair][N256]解释，起点含专家/N块以及当前pair chunk偏移。
                                     gm_ptr<int16_t, Major::MN>(BLOCK_N, reinterpret_cast<uintptr_t>(task_weights.sf + sf_offset + sf_pair_idx * BLOCK_N)),
                                     // 复制N256列的num_chunk_pairs份scale，L2策略与多M复读情况配套。
                                     BLOCK_N, num_chunk_pairs, weight_sf_l2_hint);
                }

                // 等待的是整个 M-block 的输入依赖，而不是当前物理核的上一个 GEMM：
                // GMM1 等所有 AIV1 的 dispatch 完成；GMM2 等该块全部 GMM1 N-block 的量化输出。
                // 权重/scale 与激活就绪独立，先提交它们可隐藏部分输入等待。
                // Prefetch independent weights before waiting for activations
                // 每任务只在K起点检查整M-block输入依赖，后续K块沿用该块已就绪事实。
                if (k_idx == 0 and (not is_shared or not is_linear1))
                    comm::wait_for_block<workspace>(buffer, rank_idx,
                        // GMM1等Dispatched，GMM2等Linear1Ready，shared GMM1不需要dispatch。
                        is_linear1 ? layout::kDispatched : layout::kLinear1Ready,
                        // shared进度从0起，routed从实际shared M块前缀后开始，避免同stage槽冲突。
                        (is_shared ? 0 : num_shared_m_blocks) + pool_block_idx,
                        // dispatch目标是32搬运核，GMM1量化目标是当前intermediate宽度/128任务。
                        is_linear1 ? kNumAICores : num_linear1_n_blocks);

                // Load activations and scales
                // 旧 L1A 已搬入 L0A 后可覆盖，和 L0A 等 MMAD 完成后才能覆盖是不同的释放点。
                wait<PIPE_MTE1, PIPE_MTE2>(kL1AEmpty, l1a_stage_idx);
                // 复制当前M/K子块激活到L1A，每stage物理stride仍按配置固定。
                copy_gm_to_l1(l1a[l1a_stage_idx],
                              // GM激活按每行完整K跨度解释，k_idx只偏移本次归约列范围。
                              gm_ptr<float8_e4m3_t, Major::K>(num_k, reinterpret_cast<uintptr_t>(input + k_idx)),
                              // 只装valid_m×valid_k有效数据，尾块padding不算route。
                              valid_m, valid_k, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                // MTE2完成激活data读取后，MTE1可访问该L1A。
                notify<PIPE_MTE2, PIPE_MTE1>(kL1AFull, l1a_stage_idx);

                // 激活scale仅在新32pair chunk起点装载，和权重scale共用chunk生命周期。
                if (sf_pair_idx % kNumSFPairsPerChunk == 0) {
                    // GMM1 输入 scale 是 token-major [M][K-pair]；GMM2 输入是块内 [K-pair][M256]。
                    // 布局来自 dispatch 与 SwiGLU 两种生产路径，读取模板的 Major 和 stride 必须分别匹配。
                    // GMM1输入来自token-major scale布局，由route dispatch逐行复制得到。
                    if (is_linear1) {
                        // 以[M行][Kpair]视图读取GMM1激活scale。
                        copy_gm_to_l1_mx(l1_sfa[sf_stage_idx],
                                         // row stride为完整num_sf_pairs，当前K块起点偏移sf_pair_idx。
                                         gm_ptr<int16_t, Major::K>(num_sf_pairs, reinterpret_cast<uintptr_t>(input_sf + sf_pair_idx)),
                                         // 读取valid_m行、当前chunk有效pair数；SF尾部不超过真实K。
                                         valid_m, num_chunk_pairs, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    } else {
                        // GMM2使用SwiGLU生成的块内K-major scale。
                        copy_gm_to_l1_mx(l1_sfa[sf_stage_idx],
                                         // 每pair跨固定M256行，起点偏移sf_pair_idx*256而不是逐token行stride。
                                         gm_ptr<int16_t, Major::MN>(BLOCK_M, reinterpret_cast<uintptr_t>(input_sf + sf_pair_idx * BLOCK_M)),
                                         // 有效行仍valid_m，固定256仅用于布局stride，不把空行传为真实token。
                                         valid_m, num_chunk_pairs, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    }

                    // One completion covers both scale loads on the same MTE2 queue
                    // 同一MTE2队列中两份scale均排在通知前，因此一个Full保护A/B scale。
                    notify<PIPE_MTE2, PIPE_MTE1>(kSFFull, sf_stage_idx);
                }

                // Wait for L1 inputs
                // 激活data已落L1后才进行MTE1->L0搬运。
                wait<PIPE_MTE2, PIPE_MTE1>(kL1AFull, l1a_stage_idx);
                if (is_shared) {
                    // 共享B由本核MTE2生产，等待本核B Full。
                    wait<PIPE_MTE2, PIPE_MTE1>(kSharedL1BFull, l1b_stage_idx);
                } else {
                    // routed B若拆成两个N128 stage，要分别等两份跨核Full。
                    for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N)
                        // 从AIV0写入的L1B子块顺序与load_weights的stage递增一致。
                        wait_intra_block<PIPE_MTE1, 0>(kL1BFull, (l1b_stage_idx + n_mad_idx / MAD_N) % kNumL1BStages);
                }

                // M128 的 N256 分成两个 N128 MAD，共用一次装入的 L0A；最后一个 N 子块算完才归还 L0A。
                // L0B 每次 MMAD 后独立归还，不能把 A/B 的复用周期混为一谈。
                // Reuse L0A across the BLOCK_N / MAD_N multiplies
                // 在每L1 K块内逐MAD_K装入双槽L0A/B并累加。
                for (uint32_t k_mad_idx = 0; k_mad_idx < valid_k; k_mad_idx += MAD_K) {
                    // K块内第0/1份MAD_K直接映射到两份L0A。
                    const uint32_t l0a_stage_idx = k_mad_idx / MAD_K;
                    // 全局 K-pair 在当前 32-pair SF chunk 内取模；K-block 内的 MAD_K 子块同时推进 data 和 scale。
                    const uint32_t sf_pair_idx_in_chunk = (sf_pair_idx + k_mad_idx / MX_SF_DIVISOR) % kNumSFPairsPerChunk;
                    // 建立按FRAC_MN对齐行数的MMAD输入视图，不改变GM有效M。
                    const auto l0a_mad = l0a[l0a_stage_idx].as_mad_aligned(aligned_valid_m, MAD_K);
                    // MMAD已不读此L0A stage时，MTE1才能覆盖它。
                    wait<PIPE_M, PIPE_MTE1>(kL0AEmpty, l0a_stage_idx);
                    // 当前K子块从L1A按k_mad_idx偏移搬到L0A。
                    copy_l1_to_l0a(l0a_mad, l1a[l1a_stage_idx], 0, k_mad_idx);
                    // SF chunk首次进入时才等待其Full，同chunk后续MAD沿用已经就绪的scale。
                    if (sf_pair_idx % kNumSFPairsPerChunk == 0 and k_mad_idx == 0)
                        // 确保两份A/B scale的MTE2读取均已结束。
                        wait<PIPE_MTE2, PIPE_MTE1>(kSFFull, sf_stage_idx);
                    // 当前 L0A data 与相同 K 范围的 MX scale 成对装载；scale queue 的 Full 等待在 chunk 起点只执行一次。
                    copy_l1_to_l0a_mx(l0a_mad, l1_sfa[sf_stage_idx], 0, sf_pair_idx_in_chunk, MAD_K);
                    // 整个 K 的最后 MAD 子块才允许 FIX 消费累计结果；不是每个 BLOCK_K 结束都写回一次部分和。
                    const bool is_last_k = k_idx + k_mad_idx + MAD_K == num_k;

                    // Keep B reuse credits independent across MAD shapes
                    // 沿物理N256任务逐MAD_N处理，split-N时两次共用当前L0A。
                    for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N) {
                        // 第几个N子块对应第几个独立L0C累加器。
                        const uint32_t l0c_idx = n_mad_idx / MAD_N;
                        // 将 K 子块和 N 子块映射到双槽 L0B；M128 的两个 N128 累加区独立，M256 只有一个 N256 区。
                        const uint32_t l0b_stage_idx = (l0a_stage_idx * (BLOCK_N / MAD_N) + l0c_idx) % kNumL0Stages;
                        // 等待旧MAD消费完相应L0B slot，避免数据/scale被新B覆盖。
                        wait<PIPE_M, PIPE_MTE1>(kL0BEmpty, l0b_stage_idx);
                        // shared B stage含完整N256，routed每stage只含MAD_N，因此二者N起点取法不同。
                        const auto l1b_src = is_shared ? shared_l1b[l1b_stage_idx] : l1b[(l1b_stage_idx + n_mad_idx / MAD_N) % kNumL1BStages];
                        // B数据搬运选择当前N/K子块，scale也必须使用相同坐标。
                        copy_l1_to_l0b(l0b[l0b_stage_idx], l1b_src, is_shared ? n_mad_idx : 0, k_mad_idx);
                        // B scale 使用 n_mad_idx 选 N128/N256 子块；与 L0B 的实际权重列和 sf_pair_idx_in_chunk 必须一致。
                        copy_l1_to_l0b_mx(l0b[l0b_stage_idx], l1_sfb[sf_stage_idx], n_mad_idx, sf_pair_idx_in_chunk, MAD_K);
                        // 同一MTE1队列里A/B数据和scale都在该通知之前，可统一放行MMAD。
                        notify<PIPE_MTE1, PIPE_M>(l0_full, l0b_stage_idx);

                        // B readiness also covers the earlier A/scales loads on the same MTE1 queue
                        // 累加器shape为对齐M×MAD_N，按当前N子块索引选择独立L0C。
                        const auto l0c_mad = l0c[l0c_idx].as_mad_aligned(aligned_valid_m, MAD_N);
                        // MMAD等待MTE1数据/scale全部完成，不以Scalar已发指令代替就绪。
                        wait<PIPE_MTE1, PIPE_M>(l0_full, l0b_stage_idx);
                        // 只在整个 K 的第一步清零累加器；中间 K 步累加，最后一步更新 unit flag 供 FIX 正确等待计算完成。
                        mad(l0c_mad, l0a_mad, l0b[l0b_stage_idx], aligned_valid_m, MAD_N, MAD_K,
                            // 中间K保留unit flag，最后K更新以便FIX等待最终累计完成。
                            is_last_k ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
                            // 仅k_idx+k_mad_idx==0时初始化累加器，其他K步都继续累加已有部分和。
                            k_idx + k_mad_idx == 0);
                        // 同一 L0A 会被该 K 子块的所有 N 子块复用；仅最后一次 MAD 后释放，防止提前填下一 K 覆盖输入。
                        if (n_mad_idx + MAD_N == BLOCK_N)
                            // 当前K子块所有N计算都完成后归还L0A slot。
                            notify<PIPE_M, PIPE_MTE1>(kL0AEmpty, l0a_stage_idx);
                        // 当前B子块完成就独立归还L0B，不必等另一N子块。
                        notify<PIPE_M, PIPE_MTE1>(kL0BEmpty, l0b_stage_idx);
                        // 不是整个K的末步时不执行FIX，继续下一个N/K子块累计。
                        if (not is_last_k)
                            continue;

                        // routed GMM2 的 BF16 输出也使用 ring。FIX 覆盖槽之前，必须确认上一代
                        // 同槽输出已被所有发送核消费并完成回传；GMM2 计算完成本身不能释放输出槽。
                        // Reuse output rows after all sends complete
                        // 只有routed GMM2输出会回绕；覆盖新块的第一个N子块前检查旧槽回传完成。
                        if (block_phase == sched::BlockPhase::kLinear2 and n_mad_idx == 0 and pool_block_idx >= kNumRingBlocks)
                            comm::wait_for_block<workspace>(buffer, rank_idx, layout::kCombineSent,
                                                            // 旧逻辑块b-Bring的目标为所有32发送核；不能读新块自己的CombineSent。
                                                            num_shared_m_blocks + pool_block_idx - kNumRingBlocks, kNumAICores);

                        // One epilogue credit covers BLOCK_N columns
                        // GMM1 每次 FIX 写 M128 的 UB stage，GMM2 一次写当前 valid_m 行到 GM。
                        // M128 MAD 的两个 N128 子块写同一个 N256 stage，待最后一半写完才发布 Full。
                        const uint32_t fix_step_m = is_linear1 ? STORE_BLOCK_M : valid_m;
                        // GMM1按128行FIX到UB，GMM2按完整valid_m一次FIX到GM。
                        for (uint32_t m_idx_in_block = 0; m_idx_in_block < valid_m; m_idx_in_block += fix_step_m) {
                            // FIX目标UBstage由当前起始行/128决定，最多两个stage。
                            const uint32_t ub_stage_idx = m_idx_in_block / STORE_BLOCK_M;
                            // 第一 N 子块等 epilogue credit，后续 N 子块共同填同一 stage；消费者归还之前 FIX 不能重写它。
                            if (n_mad_idx == 0)
                                wait_intra_block<PIPE_FIX, 1>(kEpilogueEmpty, ub_stage_idx);
                            if (is_linear1) {
                                // Store MAD_N gate/up columns with a BLOCK_N destination stride
                                // UB输出行宽固定N256 BF16，split-N的子块地址在同一行内偏移。
                                const ub_ptr<bfloat16_t> ub_dst(STORE_BLOCK_M, BLOCK_N,
                                                                // n_mad_idx*sizeof(BF16)仅移动N列起点，不改变UB整行stride。
                                                                ub_accum[ub_stage_idx].addr + n_mad_idx * sizeof(bfloat16_t));
                                // L0C的NZ布局通过FIX转换成UB的N256行布局。
                                copy_l0c_to_ub(ub_dst, l0c_mad.offset_nz_m(m_idx_in_block),
                                               // 尾stage只写真实剩余行，不把M256/M128对齐部分作为有效数据。
                                               min(STORE_BLOCK_M, valid_m - m_idx_in_block),
                                               // 每次FIX只写MAD_N列，关闭双目的写回，使用该调用的unit flag参数。
                                               MAD_N, asc_dual_dst_mode::DUAL_DST_DISABLE, 1);
                            } else {
                                // Store shared output to combine; routed output to the ring
                                // 共享结果直接给本地最终reduce，routed结果先给ring/通信流水。
                                const auto output_base = is_shared ? combine_input : routed_output;
                                // shared Linear2 直接落本 rank 的 combine 额外槽，无需远端 push；routed 输出落 BF16 ring 供 AIV1 回传。
                                const uint64_t output_m_idx = is_shared ?
                                    // 共享输出位于topk编号的额外槽，token仍是本rank原始行号。
                                    workspace.get_combine_m_idx(config.num_topk, static_cast<uint64_t>(pool_block_idx) * BLOCK_M) :
                                    // routed按逻辑M块映射到输出ring的物理起始行。
                                    layout::get_ring_m_idx<workspace>(pool_block_idx);
                                // 完整GM行Hstride上加当前N256块与MAD_N子块的列偏移。
                                const auto l2_output = output_base + output_m_idx * config.hidden + n_block_idx * BLOCK_N + n_mad_idx;
                                // 这里只写当前 N 子块列，GM row stride 为完整 H；全部 N-block 完成后才可发送完整 hidden 行。
                                // FIX写BF16 valid_m×MAD_N到GM，行stride H；不做远端发送或原子加。
                                copy_l0c_to_gm(l2_output, l0c_mad, valid_m, MAD_N, config.hidden, kWriteThroughShare);
                            }
                            // 这条跨核 Full 位于最后一个 N 子块的 FIX 后；AIV1 不会在 gate 半块完成但 up 半块缺失时启动激活。
                            // split-N最后一半FIX完成才发布Full，保证gate/up配对完整。
                            if (n_mad_idx + MAD_N == BLOCK_N)
                                notify_intra_block<PIPE_FIX, 1>(kEpilogueFull, ub_stage_idx);
                        }
                    }
                }

                // L1A/B 的消费者是 MTE1，而非 MMAD：最后一次 L1->L0 拷贝完成即可释放 L1，MMAD 继续用 L0。
                // Release L1 after the last L0 load
                if (is_shared) {
                    // MTE1已消费完共享B的本K块，归还sharedstage给下一轮MTE2。
                    notify<PIPE_MTE1, PIPE_MTE2>(kSharedL1BEmpty, l1b_stage_idx);
                    // 共享B stage按实际可用2/3槽回绕，不按routed4槽回绕。
                    l1b_stage_idx = (l1b_stage_idx + 1) % kNumSharedL1BStages;
                } else {
                    // 归还本任务K块消费的每个routed L1B N子块。
                    for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N)
                        // 跨核Empty让AIV0可用同一L1B槽生产后续权重。
                        notify_intra_block<PIPE_MTE1, 0>(kL1BEmpty, (l1b_stage_idx + n_mad_idx / MAD_N) % kNumL1BStages);
                    // 本次消耗BLOCK_N/MAD_N份routed B槽，stage据此前进。
                    l1b_stage_idx = (l1b_stage_idx + (BLOCK_N / MAD_N)) % kNumL1BStages;
                }
                // 当前K块的激活已完全搬入L0，L1A可被后续K块填充。
                notify<PIPE_MTE1, PIPE_MTE2>(kL1AEmpty, l1a_stage_idx);
                // L1A三个stage按0..2轮转，独立于L0双缓冲。
                l1a_stage_idx = (l1a_stage_idx + 1) % kNumL1AStages;

                // SF chunk 的最后 K-block（或整个 K 尾块）搬到 L0 后释放；data L1A/B 已可按各自周期提前回收。
                if ((sf_pair_idx + BLOCK_K / MX_SF_DIVISOR) % kNumSFPairsPerChunk == 0 or k_idx + valid_k == num_k) {
                    // scale已被当前chunk所有K子块搬入L0，归还当前A/B scale stage。
                    notify<PIPE_MTE1, PIPE_MTE2>(kSFEmpty, sf_stage_idx);
                    // 切到另一份scale stage准备下一K2048 chunk。
                    sf_stage_idx = (sf_stage_idx + 1) % kNumSFStages;
                }
            }

        };
        // Shared tasks keep one shape so their L1B stages retain the same physical stride
        // routed 每块按 valid_m 选择 MAD 形状；shared 全部块用相同 num_tokens 判定，避免共享 L1B 物理 stride 随尾块变化。
        if ((is_shared ? num_tokens : valid_m) <= 128)
            // 小M分支M128，K512/N128；实际有效行valid_m仍可能少于128。
            compute.template operator()<128>();
        else
            // 大M分支M256，K256/N256，单任务的L0C输出更大。
            compute.template operator()<256>();

        // 发布本核已提交任务序号，使 AIV1 避免在尚未发起的任务上阻塞；它不替代 FIX Full 完成事件。
        // 序号为 uint8 自然回绕，AIV1 用同宽消费序号配对；每个 UB stage 仍有 credit 限制在途任务。
        asc_store_dev(num_issued_gemms_ptr, ++ num_issued_gemms);
    }
    // 存在shared时收尾发布相关DDR访问，保证共享GM结果与后续同步顺序。
    if constexpr (kHasSharedExperts)
        asc_sync_data_barrier(DSB_DDR);
}

// AIV1: epilogue, dispatch, and combine
template <const auto& workspace, bool kOverlapReduce>
__aicore__ inline void aiv1_epilogue_comm(__gm__ uint8_t* buffer, __gm__ bfloat16_t* output,
                                          uint32_t num_tokens, uint32_t rank_idx,
                                          CombineReduceState<workspace>& reduce,
                                          float activation_clamp, uint32_t core_idx, const layout::SymBuffer<>& sym_buffer) {
    constexpr auto& config = workspace.config;
    // routed GMM1每块需要I/128次epilogue完成，作为Linear1Ready目标。
    constexpr uint32_t kNumLinear1NBlocks = config.intermediate_hidden / (BLOCK_N / 2);
    // 每块全部H/256份GMM2输出完成后才能回传完整hidden行。
    constexpr uint32_t kNumLinear2NBlocks = config.hidden / BLOCK_N;
    // shared GMM1按合并后的S*I宽度计算任务数，与routed目标分开。
    constexpr uint32_t kNumSharedLinear1NBlocks = config.shared_intermediate_hidden / (BLOCK_N / 2);
    // 三份routed payload ring使用相同块容量，但每份地址与释放条件独立。
    constexpr uint32_t kNumRingBlocks = workspace.get_num_ring_blocks();
    // shared存在与否决定额外输入、进度前缀以及直接结果槽。
    constexpr bool kHasSharedExperts = config.num_shared_experts > 0;

    // N128激活输出有128/64=2份scale pair，每份含两个E8M0字节。
    constexpr uint32_t kNumOutputSFPairs = 2;
    // UB scratch起点按32B对齐，避免行缓冲/源区间数组违反向量/DMA对齐。
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
    // 两份M128×N256 BF16槽从UB0起，AIC FIX向这里生产GMM1输出。
    constexpr ub_ptr<bfloat16_t> ub_accum(STORE_BLOCK_M, BLOCK_N);
    // 唯一一份M128行FP32权重紧跟两份FIX槽，stage之间靠TopkEmpty串行复用。
    constexpr ub_ptr<float> ub_route_weights(1, STORE_BLOCK_M, ub_accum.offset(kNumEpilogueStages));
    // 每epilogue stage独立保存[2pair][M128] scale，放在weight scratch之后。
    constexpr ub_ptr<int16_t> ub_output_sf(kNumOutputSFPairs, STORE_BLOCK_M, ub_route_weights.offset(1));

    // Dispatch inputs and source ranges
    // 两份pull行缓冲紧接scale区，每行H个FP8字节。
    constexpr ub_ptr<float8_e4m3_t> ub_pull_token(1, config.hidden, ub_output_sf.offset(kNumEpilogueStages));
    // 每份pull scale按32B补齐，int16单位换算后定义UB列宽。
    constexpr ub_ptr<int16_t> ub_pull_sf(1, aligned(config.num_input_sf_pairs * sizeof(int16_t), kNumUBAlignmentBytes) / sizeof(int16_t),
                                         // scale缓冲起点在两份完整pull token行之后，防止二者重叠。
                                         ub_pull_token.offset(kNumPullStages));
    // 各源rank拼接区间的begin数组，起点再按32B对齐。
    constexpr ub_ptr<uint32_t> ub_src_m_begin(1, config.num_ranks, aligned(ub_pull_sf.offset(kNumPullStages), kNumUBAlignmentBytes));
    // end数组紧接begin，两者只由本核Scalar控制，不参与DMA槽轮转。
    constexpr ub_ptr<uint32_t> ub_src_m_end(1, config.num_ranks, ub_src_m_begin.offset(1));

    // Combine sends and reduction
    // 四份BF16 push行槽放在源区间表之后，起点32B对齐。
    constexpr ub_ptr<bfloat16_t> ub_push_token(1, config.hidden, aligned(ub_src_m_end.offset(1), kNumUBAlignmentBytes));
    // 剩余UB构建尾段reduce视图；权重/激活/通信缓冲仍在使用时不能占地址0。
    constexpr auto ub_reduce_inputs = make_reduce_buffer<workspace>(ub_push_token.offset(kNumPushStages));

    // Size the reduce window from local output tokens
    // 此处窗口按实际本地 T 估算尾段位置，但 ring/overlap 编译策略按 Tmax；active T 与 registered capacity 分别用于不同决策。
    const uint32_t num_reduce_window_m_blocks = max(ceil_div(num_tokens, BLOCK_M), 1u);
    // 实际shared块数作为routed进度槽起点，不影响routed pool_block_idx自己的编号。
    const uint32_t num_shared_m_blocks = kHasSharedExperts ? ceil_div(num_tokens, BLOCK_M) : 0;

    // 即使H很大也必须容纳push与至少一份有效reduce向量，否则编译期拒绝。
    static_assert(ub_push_token.offset(kNumPushStages) <= UBSizeBytes and ub_reduce_inputs.shape_n > 0, "AIV1 UB overflow");

    // 初始 Empty credit 允许 AIC 第一次 FIX 写两份 UB；后续 credit 只能由当前 stage 的消费者归还。
    notify_intra_block<PIPE_MTE3, 1, kNumEpilogueStages>(kEpilogueEmpty);
    // push槽首次未存旧token，为四stage各预置一份MTE3->MTE2 Empty。
    notify<PIPE_MTE3, PIPE_MTE2, kNumPushStages>(kPushEmpty);
    // 当前无旧weight Vector消费，允许第一份权重DMA直接开始。
    notify<PIPE_V, PIPE_MTE2>(kTopkEmpty);
    // 两份pull槽首次可写；后续由本地MTE3落ring完成后归还。
    notify<PIPE_MTE3, PIPE_MTE2, kNumPullStages>(kPullEmpty);

    // 置换后的每逻辑route行对应FP32weight的GM全量表。
    const auto routed_weights = workspace.routed.topk_weights.get_ptr(buffer);
    // 置换后每逻辑route行的source rank与返回槽编码。
    const auto src_metadata = workspace.routed.src_metadata.get_ptr(buffer);
    // 源端已push至本rank的专家/source metadata，不是远程token payload。
    const auto src_token_metadata = workspace.metadata.src_token_metadata.get_ptr(buffer);

    // metadata到达前routed块总数未知；零不代表已确认没有路由工作。
    uint32_t num_total_m_blocks = 0;  // Routed pool only; shared blocks are separate
    // 主循环需先看到AIV0的metadata_ready再解释num_total_m_blocks。
    bool metadata_ready = false;
    // certificate低位总块数将在metadata_ready后读取。
    const auto num_m_blocks_certificate = workspace.metadata.num_m_blocks_certificate.get_ptr(buffer);

    // 完成计数与 payload 的发布顺序：先等相应 MTE3/FIX，DSB_DDR 发布 Scalar metadata，再 atomic_add。
    // Dispatched 的调用者已等 kPullDone，Linear2Ready 的调用者已等 FIX Full；其余在这里排空 MTE3。
    // 统一ready发布函数：num_slots可为1个计算块或若干批量完成回传块。
    const auto publish_block_progress = [&](layout::ProgressStage stage,
                                            // slot_begin是含shared前缀的逻辑进度下标，num_slots为连续计数槽个数。
                                            uint32_t slot_begin, uint32_t num_slots) __aicore__ {
        // Wait for stores; dispatch and FIX callers already waited
        // SwiGLU写data/scale和combine远端写需要在此等MTE3；其他两种stage由调用者已等待。
        if (stage == layout::kLinear1Ready or stage == layout::kCombineSent) {
            // 在MTE3当前队列尾发到Scalar的store完成事件。
            notify<PIPE_MTE3, PIPE_S>(kStoresDone);
            // 该wait会阻塞Scalar直到此前MTE3完成，并非只做非阻塞ready探测。
            wait<PIPE_MTE3, PIPE_S>(kStoresDone);
        }

        // Publish this core's contribution after its Scalar metadata is visible
        // 发布Scalar写入的metadata/路由映射，再允许其他核通过原子count观察就绪。
        asc_sync_data_barrier(DSB_DDR);
        // 一次对多个连续块贡献 +1 只用于批量回传完成；计数是各逻辑块独立的，不随 ring 槽回绕。
        // DSB 还覆盖 Scalar 写的 route weight/src_metadata/route->block 映射；
        // 只等待 MTE3 不能替代这些 Scalar 控制字段的可见性发布。
        // 每个逻辑块各加一份本核贡献，不能对整个批次只更新一个共享count。
        for (uint32_t slot_offset = 0; slot_offset < num_slots; ++ slot_offset)
            // 地址按stage+logical slot取值；每槽sector stride由Buffer定义，原子更新uint32。
            asc_atomic_add(workspace.progress.blocks.get_ptr(buffer, workspace.get_progress_idx(stage, slot_begin) + slot_offset), 1u);
    };

    // 对应AIC的uint8任务提交序号；只用于判断是否有下一份FIX任务，真正完成还需Full。
    const auto num_issued_gemms_ptr = workspace.progress.num_issued_gemms.get_ptr(buffer, core_idx);

    // dispatch目标FP8 data ring基址。
    const auto l1_acts = workspace.routed.l1_acts.data.get_ptr(buffer);
    // dispatch目标scale ring基址，输入布局为token-major pair顺序。
    const auto l1_acts_sf = workspace.routed.l1_acts.sf.get_ptr(buffer);
    // pull 状态同时保留逻辑块、当前 owner 行、专家游标和源行置换；一次推进后可暂停，DMA 槽由事件保护。
    struct DispatchPullState {
        // 从routed逻辑块0开始，后续顺序推进，不在此变量上取ring模。
        uint32_t pool_block_idx = 0;
        // 本核下一行号；256作为“行已发完、待store发布”的哨兵。
        uint32_t m_idx_in_block;  // BLOCK_M marks pending stores
        // 当前块真实有效行数，首次进入块时由专家游标计算。
        uint32_t valid_m;
        // 保存当前专家及其逻辑块区间，用于跳过空专家并保持均衡切分一致。
        sched::ExpertCursor<workspace> expert;

        // Permute the concatenated source rows without changing their counts
        // 置换源行位置、每目标行的置换stride、同核下一行的32*stride增量。
        uint32_t src_m_idx = 0, src_m_stride = 0, src_m_step = 0;
        // 源rank区间搜索游标，置换回绕后需回到0。
        uint32_t src_rank_idx = 0;
    };
    // 每AIV1从core_idx行开始，以32行步长处理整个M-block；32核覆盖互不重叠行。
    DispatchPullState pull{.m_idx_in_block = core_idx};

    // 每次只推进有限行，随后返回主循环给 epilogue/push 执行机会。
    // pull 是专家端远程读取：每条 route 都独立拉取源 token；同卡多专家命中同一 token 不做通信去重。
    // Pull one batch of input rows
    const auto advance_dispatch_pull = [&]() __aicore__ {
        // 全部逻辑块已拉完立即返回；最终通信/归并仍可能未结束。
        if (pull.pool_block_idx == num_total_m_blocks)
            return;

        // 上一小步把行游标置为256后，在此排空该块已发stores。
        if (pull.m_idx_in_block >= BLOCK_M) {
            // Wait for the previous block's stores before publishing readiness
            // 等待MTE3队列尾的PullDone，确保token与scale落GM后才发布Dispatched。
            wait<PIPE_MTE3, PIPE_S>(kPullDone);

            // kDispatched 计的是参与搬运的核数，不是有效行数。空行 owner 也必须发布，
            // 否则尾块/小 M 的 AIC 等待 kNumAICores 永远无法满足。
            // Empty row owners must also contribute
            // 给当前routed块加本核1份贡献，逻辑slot含shared前缀；32核齐备才可GMM1读取。
            publish_block_progress(layout::kDispatched, num_shared_m_blocks + pull.pool_block_idx, 1);
            // 本核dispatch游标前进一个逻辑块，状态表保持旧块完成记录。
            ++ pull.pool_block_idx;
            // 新块重新从本核core_idx行开始，稍后首次进入将更新有效M与源行起点。
            pull.m_idx_in_block = core_idx;
            // 发布一个块后主动让出执行，避免连续跨块dispatch占住AIV1。
            return;
        }

        // 仅在新块的第一行检查旧ring槽释放条件，块内后续小步沿用已取得的槽所有权。
        if (pull.m_idx_in_block == core_idx) {
            // 仅回收 Linear1 输入 ring：等旧块所有 GMM1 epilogue 完成后才覆盖。
            // 这里用量化完成作为保守释放点；其他 ring 各有自己的消费完成条件。
            // Reuse input rows after all Linear1 N-blocks finish
            if (pull.pool_block_idx >= kNumRingBlocks and
                    not comm::is_block_ready<workspace>(buffer, layout::kLinear1Ready,
                                                        // 检查与新块映射同物理slot的旧逻辑块b-Bring。
                                                        num_shared_m_blocks + pull.pool_block_idx - kNumRingBlocks,
                                                        // 旧GMM1全部I/128份量化完成才放行；不是按有效行数计数。
                                                        kNumLinear1NBlocks))
                // 旧槽还被消费时不进入阻塞wait，而是返回主循环给epilogue/push推进机会。
                return;

            // Load source counts and advance the expert
            // 当前块越过专家右界时继续读下一专家；空专家会保持相同block_end并被跳过。
            while (pull.pool_block_idx >= pull.expert.pool_block_end) {
                // 专家游标初始为全1，第一次+1得到本地专家0。
                const uint32_t next_expert_idx = pull.expert.local_expert_idx + 1;
                // 累计当前专家所有source的route数，并记录非空source个数供互素stride选择。
                uint32_t num_expert_tokens = 0, num_sources = 0;
                // 按source rank顺序建立原始route拼接区间，和metadata接收分区一致。
                for (uint32_t src_rank_idx = 0; src_rank_idx < config.num_ranks; ++ src_rank_idx) {
                    // 读取[src rank][当前local expert]的独立count，不读取高位arrival/sum作为行数。
                    const uint32_t count = asc_load_dev(workspace.metadata.expert_recv_count.get_ptr(
                        buffer, workspace.get_expert_recv_idx(src_rank_idx, next_expert_idx)));
                    // 每个源 count 转为专家内拼接区间 [begin,end)；空 peer 的区间长度为 0，定位时自然跳过。
                    ub_src_m_begin.ptr()[src_rank_idx] = num_expert_tokens;
                    // count累加形成下一peer begin，当前peer end即本次累加后的值。
                    num_expert_tokens += count;
                    // 保存当前source右开边界，定位时源行等于end应进入下一个peer。
                    ub_src_m_end.ptr()[src_rank_idx] = num_expert_tokens;
                    // 只统计count>0的source，避免空source影响均衡读取stride估计。
                    num_sources += count != 0;
                }
                // 把总route数交给统一ExpertCursor，更新专家逻辑块末尾。
                pull.expert.next_expert(num_expert_tokens);
                // 新专家从source0开始定位，空peer由end比较自然跳过。
                pull.src_rank_idx = 0;
                // 空专家不做除法/取模；while会继续跳到下一个非空专家。
                if (num_expert_tokens != 0) {
                    // 互素 stride 使 [0, num_expert_tokens) 的行恰好遍历一次，分散源 rank 访问。
                    // 这是路由行的排列优化，不合并 route，也不减少 token payload 字节数。
                    pull.src_m_stride = next_coprime(ceil_div(num_expert_tokens, num_sources), num_expert_tokens) % num_expert_tokens;
                    // 本核负责目标行 core_idx+n*32，因此下一源行在置换序列中跳 32*stride（mod 专家 route 数）。
                    pull.src_m_step = kNumAICores * pull.src_m_stride % num_expert_tokens;
                }
            }
            // 当前逻辑块在专家内紧凑行序列的起点，用来初始化置换。
            uint32_t expert_m_begin;
            // 统一计算当前块有效M与专家内begin，必须与AIC/回传游标一致。
            pull.valid_m = pull.expert.get_m_block_range(pull.pool_block_idx, expert_m_begin);
            // 专家内原行号 = expert_m_begin + core_idx，再作乘 stride 置换。
            // 各块共用同一置换，从而跨块也不重复/遗漏 route；并非在每块内部重新取模。
            pull.src_m_idx = static_cast<uint64_t>(expert_m_begin + core_idx) * pull.src_m_stride % pull.expert.num_expert_tokens;
            // 置换后的新块起点可能退到当前source之前，不能只单向继续旧source游标。
            if (pull.src_m_idx < ub_src_m_begin.ptr()[pull.src_rank_idx])
                // 退回source0重新定位，随后循环会找到实际非空source区间。
                pull.src_rank_idx = 0;
        }

        // 每次最多推进2份pull stage，也受当前块剩余有效行限制。
        for (uint32_t step_idx = 0; step_idx < kNumPullStages and pull.m_idx_in_block < pull.valid_m; ++ step_idx) {
            // 保留本次目标行号，再提前推进下一行游标以便下次恢复。
            const uint32_t m_idx_in_block = pull.m_idx_in_block;
            // 同核行号步长32，保持各核目标写区间互斥。
            pull.m_idx_in_block += kNumAICores;
            // 每核的第 n 行使用 n%2 pull 槽；pool_m_idx 保存全量 metadata，ring_m_idx 定位可回绕 payload。
            const uint32_t stage_idx = m_idx_in_block / kNumAICores % kNumPullStages;
            // 例 ring_blocks=32，逻辑块b=35：payload 使用物理块3，metadata/ready 仍使用逻辑块35。
            // 覆盖块3前等逻辑块3的旧消费者完成；不能拿 ready[35] 去判断是否可覆盖旧槽。
            const uint64_t pool_m_idx = static_cast<uint64_t>(pull.pool_block_idx) * BLOCK_M + m_idx_in_block;
            // payload物理行用get_ring_m_idx，metadata逻辑行用pool_m_idx，二者回绕策略不同。
            const uint64_t ring_m_idx = layout::get_ring_m_idx<workspace>(pull.pool_block_idx, m_idx_in_block);
            // 用源前缀区间定位 source rank，再减 begin 得该源的专家内 metadata slot；置换回绕时 rank 游标归零。
            while (pull.src_m_idx >= ub_src_m_end.ptr()[pull.src_rank_idx])
                // 跳过源区间右界；零count peer的begin=end，因此也会被跳过。
                ++ pull.src_rank_idx;
            // 固定本次route source身份，后面推进置换游标不应修改此局部变量。
            const uint32_t src_rank_idx = pull.src_rank_idx;
            // 源metadata slot = 置换后的专家内行号 - source区间begin。
            const uint32_t src_slot_idx = pull.src_m_idx - ub_src_m_begin.ptr()[src_rank_idx];

            // 下一次本核route沿32*stride置换步长前进。
            pull.src_m_idx += pull.src_m_step;
            // src_m_step 已取模，当前 idx 与 step 都小于专家总行数，最多减一次即可回绕；回绕后源 rank 搜索从0重启。
            if (pull.src_m_idx >= pull.expert.num_expert_tokens) {
                // 取模增量已小于总行数，所以越界只需减一次总行数。
                pull.src_m_idx -= pull.expert.num_expert_tokens;
                // 回绕后源rank搜索从0重启，不能继续向更高rank扫描。
                pull.src_rank_idx = 0;
            }
            // metadata 是源端先推给本专家 rank 的本地 GM；此处 map 仅用于读取真正 FP8 token/scale 的源端 peer 地址。
            const auto remote_base = sym_buffer.map(buffer, src_rank_idx);

            // metadata 低 32 位是 token * topk + slot，高 32 位是 FP32 路由权重的原始位。
            // 索引仍保留 top-k 槽身份：即使 token 或专家重复，combine 也写入不同 route 槽。
            // Resolve the source token
            // 旧pull槽的MTE3已消费完UB才可安排下一份远端MTE2读取；此wait仍可能阻塞。
            wait<PIPE_MTE3, PIPE_MTE2>(kPullEmpty, stage_idx);
            // 64-bit metadata在本地GM中读取，低32位route索引、高32位FP32 weight位。
            const uint64_t token_metadata = asc_load_dev(src_token_metadata +
                // 布局定位[local expert][source rank][source slot]，与源端push使用同一公式。
                workspace.get_src_metadata_idx(pull.expert.local_expert_idx, src_rank_idx, src_slot_idx));
            // 截取低32位恢复原token/top-k扁平索引，不是专家内slot。
            const uint32_t token_topk_idx = static_cast<uint32_t>(token_metadata);
            // 除topk恢复原source token行号，去掉top-k槽维度。
            const uint32_t src_token_idx = token_topk_idx / config.num_topk;
            // 同 token 的 j0/j1 即使同卡也会各拉一遍 token；与 DeepEP 的 (token,dst rank) 去重发送不同。
            // 独立 top-k 槽保证两个 route 权重和返回结果不会互相覆盖。
            const uint32_t src_topk_idx = token_topk_idx % config.num_topk;
            // FP8源token起点按原source token*H定位，同token多route将引用同一源地址。
            const auto remote_token = workspace.input.acts.data.get_ptr(remote_base, static_cast<uint64_t>(src_token_idx) * config.hidden);
            // scale独立源区按token*(H/64 pair)定位，不与FP8数据连续打包。
            const auto remote_sf = workspace.input.acts.sf.get_ptr(remote_base, static_cast<uint64_t>(src_token_idx) * config.num_input_sf_pairs);

            // 每 route 发两次 MTE2：H 字节 FP8 token + H/64 个 int16 scale pair（共 H/32 字节）。
            // Full 事件位于同一 MTE2 队列的两次读取之后，等待它才可 MTE3 消费两份 UB 数据。
            // Pull token data and scales
            // 本次远端MTE2读token到当前pull UB行槽。
            copy_gm_to_ub(ub_pull_token[stage_idx].ptr(), remote_token,
                          // 1行H列，完整Hbyte stride；普通cache hint用于此处源输入读取。
                          1, config.hidden, config.hidden, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            // 第二次MTE2读同token的scale pair，保留int16类型与2B字节宽度。
            copy_gm_to_ub(ub_pull_sf[stage_idx].ptr(), remote_sf, 1, config.num_input_sf_pairs,
                          // 一行scale字节数为(H/64)*2；与token读取同队列，一个Full可覆盖二者。
                          config.num_input_sf_pairs * sizeof(int16_t), asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
            // 通知MTE3当前stage两份输入DMA均已完成。
            notify<PIPE_MTE2, PIPE_MTE3>(kPullFull, stage_idx);

            // 落本地 GMM1 ring 后发 Empty，表示 MTE3 已不再读取该 UB 槽；跨块 ready 还要等 kPullDone。
            // Stage routed inputs
            // MTE3等pull Full后再读token/scale UB，防止搬运未完成内容。
            wait<PIPE_MTE2, PIPE_MTE3>(kPullFull, stage_idx);
            // 将完整H宽token写本地GMM1输入ring的物理目标行。
            copy_ub_to_gm(l1_acts + ring_m_idx * config.hidden,
                          // 源为pull stage行，shape1×H，data每元素1B。
                          ub_pull_token[stage_idx].ptr(), 1, config.hidden,
                          // 目的行stride为完整H，使用当前实现的write-through-share策略。
                          config.hidden, kWriteThroughShare);
            // scale写入另一区域，与相同ring物理行的data配对。
            copy_ub_to_gm(l1_acts_sf + ring_m_idx * config.num_input_sf_pairs,
                          // 每行num_input_sf_pairs个int16，不把32B UB padding当有效scale写出。
                          ub_pull_sf[stage_idx].ptr(), 1, config.num_input_sf_pairs,
                          // scale目标行stride=num_input_sf_pairs*2B，GM紧凑保存有效pair。
                          config.num_input_sf_pairs * sizeof(int16_t), kWriteThroughShare);
            // MTE3队列用完UB行槽后归还给MTE2，逻辑块完成还要另等PullDone。
            notify<PIPE_MTE3, PIPE_MTE2>(kPullEmpty, stage_idx);

            // Copy route weights as FP32 bits while DMA runs
            // Scalar 写 weight 与 DMA token 搬运重叠；weight 按 pool_m_idx 全量保存，避免输入 ring 回绕后元数据丢失。
            const uint32_t route_weight_bits = static_cast<uint32_t>(token_metadata >> 32);
            // 把weight原始位写到float区域，实现无精度损失的Scalar拷贝，与DMA并行。
            asc_store_dev(reinterpret_cast<__gm__ uint32_t*>(routed_weights + pool_m_idx), route_weight_bits);
            // Record the actual row assignment; combine readiness never infers source/token order
            // 只有早归并模式需要远端回填producer块映射，普通最终barrier模式可删除此写。
            if constexpr (kOverlapReduce)
                // 专家端向源 rank 写回 route 对应的真实逻辑 M-block，供源端早归并核对 producer prefix。
                // 该映射按原 route 存，不受 ring 回绕及源行置换影响。
                asc_store_dev(workspace.route_m_blocks.get_ptr(remote_base, token_topk_idx), pull.pool_block_idx);
            // Source metadata: rank [31:24], combine M [23:0]
            // 高 8 位存 source rank，低 24 位存 topk_slot*Tmax+token；combine 从此恢复目标 rank/槽。
            asc_store_dev(src_metadata + pool_m_idx,
                // 低24位为topk_slot*Tmax+token，高8位为source rank，供push解析目标。
                static_cast<uint32_t>(workspace.get_combine_m_idx(src_topk_idx, src_token_idx)) | (src_rank_idx << 24));
        }

        // 本核可能最后一轮不足2行，甚至有效M小于core_idx；也必须走完成发布路径。
        if (pull.m_idx_in_block >= pull.valid_m) {
            // BLOCK_M marks pending stores; publish on the next step
            // BLOCK_M 是“本核行已发完、尚待 stores 完成”的哨兵；下一次进入先等 kPullDone，再贡献块计数。
            // 没有有效行的 owner 也走该分支，保证尾块的完成参与数固定为 32。
            pull.m_idx_in_block = BLOCK_M;
            // 在本块最后的MTE3后发PullDone，下一次进入函数再wait并更新Dispatched。
            notify<PIPE_MTE3, PIPE_S>(kPullDone);
        }
    };

    // phase/task_idx 是下一份消费任务，m_idx_in_block 是已经处理的行前缀；
    // num_completed_gemms 与 AIC issued 同宽配对，只有完成当前任务全部有效行才推进。
    struct EpilogueState {
        // epilogue私有调度进度，独立于AIC但按相同规则取任务。
        sched::LinearTaskState schedule;
        // epilogue只需当前层有效M，对routed GMM1使用其专家游标。
        sched::ExpertCursor<workspace> expert;
        // 当前已选中、尚未完整消费的线性GEMM任务号。
        uint32_t task_idx;
        // 当前任务M-block有效行数；仅GMM1需要逐行VF处理。
        uint32_t valid_m;
        // 空闲phase允许主循环取下一任务；有任务时保留到其全部行消费完成。
        sched::BlockPhase phase = sched::BlockPhase::kNone;
        // 当前任务已处理行数，32行一步推进，跨M128 stage时切换FIX credit。
        uint32_t m_idx_in_block = 0;
        // uint8已消费序号与AIC issued自然回绕配对，受stage credit限制在途任务。
        uint8_t num_completed_gemms = 0;  // Advanced by epilogue, paired with num_issued_gemms
    };
    // 首次状态全部游标归零；无已选任务，等待主循环调用调度器。
    EpilogueState epi{};

    // Process Linear1 rows or wait for Linear2 stores; return true on progress
    const auto advance_epilogue = [&]() __aicore__ -> bool {
        // 没有当前epilogue任务时立即返回，不等待任何FIX事件。
        if (epi.phase == sched::BlockPhase::kNone)
            return false;

        // Issued tasks still need their FIX completion event
        // 只在新任务起点比较生产/消费序号；已开始的 M-block 必须继续处理其剩余 stage/行。
        // 序号相等时立即返回，让主循环执行通信，而不是在尚未提交的 FIX 事件上等待。
        if (epi.m_idx_in_block == 0 and asc_load_dev(num_issued_gemms_ptr) == epi.num_completed_gemms)
            return false;

        // 识别shared路径以选择是否读route weight、是否使用独立shared_l2区域。
        const bool is_shared = epi.phase == sched::BlockPhase::kSharedLinear1 or epi.phase == sched::BlockPhase::kSharedLinear2;
        // GMM1要SwiGLU量化，GMM2只确认FIX并更新完成计数。
        const bool is_linear1 = epi.phase == sched::BlockPhase::kLinear1 or epi.phase == sched::BlockPhase::kSharedLinear1;
        // 依当前路径/层选择task的N维展开宽度，避免shared任务解码成routed列数。
        const uint32_t num_n_blocks = is_linear1 ? (is_shared ? kNumSharedLinear1NBlocks : kNumLinear1NBlocks) : kNumLinear2NBlocks;
        // 通过当前层N任务数恢复该epilogue任务所属逻辑M-block。
        const uint32_t pool_block_idx = epi.task_idx / num_n_blocks;

        // Linear2输出已由FIX直接写GM，Vector不再做数值激活。
        if (not is_linear1) {
            // Wait for Linear2 stores and return FIX credit
            // GMM2 没有 SwiGLU：等 FIX 的 GM store 完成，归还 credit，再在下方为该 N-block 发布 Linear2Ready。
            wait_intra_block<PIPE_S, 1>(kEpilogueFull);
            // 消费完FIX完成通知后归还credit，AIC后续可提交新的FIX输出。
            notify_intra_block<PIPE_S, 1>(kEpilogueEmpty);
        } else {
            // SwiGLU 写的是 Linear2 输入 ring；必须等上一代同槽的全部 GMM2 N-block 消费完。
            // 与 dispatch 输入 ring、GMM2 输出 ring 的释放条件不同，不能共用一个 ready 标志。
            // Reuse Linear2 inputs after all output N-blocks finish
            // l2_acts ring 回收点：在新 GMM1 epilogue 的第一行检查 b-ring_blocks 的 H/256 个 GMM2 完成。
            // 槽仍被旧 GMM2 读取时返回 false，不占住 AIV1，主循环可推进旧块 combine/pull。
            if (not is_shared and epi.m_idx_in_block == 0 and pool_block_idx >= kNumRingBlocks and
                    not comm::is_block_ready<workspace>(buffer, layout::kLinear2Ready,
                                                        // 检查旧l2_acts物理槽的上一代逻辑块b-Bring。
                                                        num_shared_m_blocks + pool_block_idx - kNumRingBlocks,
                                                        // 等待旧块的H/256份GMM2输出完成，作为旧激活已不再被读取的保守释放点。
                                                        kNumLinear2NBlocks))
                // 不阻塞等待该旧块，先返回让主循环推进通信/其他任务。
                return false;

            // 一个新GMM1任务开始时取valid_m，处理到中间32行小步时不重复更新。
            if (epi.m_idx_in_block == 0)
                // shared按本地T计算尾块；routed通过统一专家分块规则取M。
                epi.valid_m = is_shared ? min(num_tokens - pool_block_idx * BLOCK_M, BLOCK_M) :
                    epi.expert.advance_to(buffer, pool_block_idx);
            // 行号拆为 M128 stage 和 stage 内偏移；一个 stage 可分多次 32-row VF 处理，但只在 stage 起点等一次 FIX。
            const uint32_t stage_idx = epi.m_idx_in_block / STORE_BLOCK_M;
            // 当前行在其M128 UBstage内的偏移，VF输入/输出指针都从此处开始。
            const uint32_t m_idx_in_stage = epi.m_idx_in_block % STORE_BLOCK_M;
            // 当前UBstage对应逻辑M块的行起点，为0或128。
            const uint32_t stage_m_begin = stage_idx * STORE_BLOCK_M;
            // 当前stage真实行数=min(128,当前块剩余行数)，尾stage不写越界行。
            const uint32_t stage_valid_m = min(STORE_BLOCK_M, epi.valid_m - stage_m_begin);
            // Yield after at most 32 rows
            // 一个 M128 UB stage 分最多4次32行处理；只有 stage 完整处理后才写 GM/归还 FIX credit。
            // VF 每次内部再按2行处理；小步是给通信状态机让出调度机会，不是改变 GEMM M256 切分。
            const uint32_t step_m = min(32u, epi.valid_m - epi.m_idx_in_block);

            // Load route weights and wait for FIX
            // 仅stage首个小步加载weight并等FIX，此后同stage后续32行沿用已就绪数据。
            if (m_idx_in_stage == 0) {
                // shared分支没有route weight，也无需占用唯一weight UB。
                if (not is_shared) {
                    // routed weight按逻辑块+stage行号读取，绝不使用物理ring行号。
                    const auto route_weights = workspace.routed.topk_weights.get_ptr(
                        // 此stage的weight起点等于pool_block*256+stage_m_begin。
                        buffer, static_cast<uint64_t>(pool_block_idx) * BLOCK_M + stage_m_begin);
                    // route weight UB 只有一份，等上一 stage 的 Vector 读完才能加载新 M128 行权重。
                    wait<PIPE_V, PIPE_MTE2>(kTopkEmpty);
                    // 每stage一次性预取最多128行FP32weight，后续每个32行VF只用对应子范围。
                    copy_gm_to_ub(ub_route_weights.ptr(), route_weights, 1, stage_valid_m,
                                  // weight字节stride=stage_valid_m*4B，L2 hint与小metadata访问配套。
                                  stage_valid_m * sizeof(float), asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
                    // 通知Vector整份weight stage的DMA已完成。
                    notify<PIPE_MTE2, PIPE_V>(kTopkFull);
                }
                // 等待当前 stage 的 gate/up 均已由 FIX 填完，保证 vf_swiglu_quant 读取完整配对输出。
                wait_intra_block<PIPE_V, 1>(kEpilogueFull, stage_idx);
                // SwiGLU 需要 FIX gate/up 与 route weight 两条生产流都完成；单等一个 Full 不能证明另一份输入就绪。
                // routed需额外等weight Full；shared只有FIX输入，因此编译期省去此等待。
                if (not is_shared)
                    wait<PIPE_MTE2, PIPE_V>(kTopkFull);
            }

            // 一个物理 N256 输出含 gate128 + up128，产生 N128 激活；M 仍是当前块的有效行数。
            // routed 分支在量化前乘 top-k weight，返回结果已加权，最终 reduce 不再乘一次。
            // Quantize one row group
            // BF16输入从当前stage已处理行开始，每行N256元素，gate/up均在同一行内。
            const auto input = ub_accum[stage_idx].ptr() + m_idx_in_stage * BLOCK_N;
            // FP8 量化输出压紧到同一 BF16 UB 的低地址，避免另开激活输出区；输入/输出 stride 分别为 N256*2B、N128*1B。
            const auto output = reinterpret_cast<__ubuf__ float8_e4m3_t*>(ub_accum[stage_idx].ptr()) +
                                // FP8输出每行只占N128字节，压在同一UBstage起点后的紧凑输出区。
                                m_idx_in_stage * (BLOCK_N / 2);
            // scale指针先移动当前行偏移，VF内部再按STORE_BLOCK_M跨pair列。
            const auto output_sf = ub_output_sf[stage_idx].ptr() + m_idx_in_stage;
            // shared激活不乘路由weight，但仍执行相同clamp、SwiGLU与MX FP8量化。
            if (is_shared) {
                // kApplyRouteWeight=false消除weight读取，step_m仍控制本次32行小步。
                vf_swiglu_quant<false, STORE_BLOCK_M>(input, output, output_sf, nullptr, step_m, activation_clamp);
            } else {
                // routed在量化前乘当前route weight；后续GMM2和最终reduce不再次施加它。
                vf_swiglu_quant<true, STORE_BLOCK_M>(input, output, output_sf, ub_route_weights.ptr() + m_idx_in_stage, step_m, activation_clamp);
            }
            // 累积处理行前缀；stage 未结束时保留 UB data/scale，不提前释放给 FIX。
            epi.m_idx_in_block += step_m;

            // Finish GM stores before FIX reuses this UB stage
            // 只有当前stage的全部有效行完成，才批量写data/scale至GM并归还UB。
            if (m_idx_in_stage + step_m == stage_valid_m) {
                // stage权重Vector已读完，routed可归还唯一weight scratch给后续stage。
                if (not is_shared)
                    // weight区与FIX输出区的credit独立，允许下次weight预取与其他流水并行。
                    notify<PIPE_V, PIPE_MTE2>(kTopkEmpty);
                // 当前stage全部量化结果已经由Vector产生，通知MTE3。
                notify<PIPE_V, PIPE_MTE3>(kQuantFull);
                // MTE3读取压紧输出前等待Vector生产完成。
                wait<PIPE_V, PIPE_MTE3>(kQuantFull);

                // 当前GMM1任务的N块序号，对应SwiGLU输出每次128列的列组。
                const uint32_t n_block_idx = epi.task_idx % num_n_blocks;
                // shared数据不回绕，routed数据先映射ring；二者随后都乘256得到物理行基址。
                const uint64_t output_m_idx = static_cast<uint64_t>(
                    is_shared ? pool_block_idx : layout::get_ring_block_idx<workspace>(pool_block_idx)) * BLOCK_M;
                // shared/routed各自有独立l2_acts空间及对应scale，保持后续GMM2选择一致。
                const auto dst_layout = is_shared ? workspace.shared_l2_acts : workspace.routed.l2_acts;
                // 每个任务产 N128，只写 [M][n_block*128:...+128]；GM data 行 stride 是整层 I（shared 为扩展 I）。
                const uint32_t num_cols = num_n_blocks * (BLOCK_N / 2);
                // 当前层完整I宽每行有I/64个scale pair，用来计算跨M块scale基址。
                const uint32_t num_sf_pairs = num_cols / MX_SF_DIVISOR;
                // 目的data起点包括M-block、stage行和当前128列组偏移。
                const auto dst = dst_layout.data.get_ptr(buffer,
                    // 整体GM row stride=num_cols，stage_m_begin移动行，n_block_idx移动列。
                    (output_m_idx + stage_m_begin) * num_cols + n_block_idx * (BLOCK_N / 2));
                // scale 的物理布局为 [M-block][K-pair][M256]；当前任务两对 scale 的列间 stride 为 M256。
                // stage_m_begin 只移动 M 维，不能按 token-major 的 data 布局计算 scale 地址。
                const auto dst_sf = dst_layout.sf.get_ptr(buffer,
                    // scale基址为块起点+当前N任务两对scale的M256跨度+stage行起点。
                    output_m_idx * num_sf_pairs + n_block_idx * kNumOutputSFPairs * BLOCK_M + stage_m_begin);

                // 将UB压紧FP8视为[M128,N128]，仅复制stage_valid_m行。
                copy_ub_to_gm(dst, ub_ptr<float8_e4m3_t>(STORE_BLOCK_M, BLOCK_N / 2, ub_accum[stage_idx].addr),
                              // GM目的整行stride=num_cols，不是本任务局部N128宽。
                              stage_valid_m, BLOCK_N / 2, num_cols, kWriteThroughShare);
                // 每任务产生两对 scale，每对沿 stage_valid_m 行搬运；GM 目的对间距始终256行，尾块也不压缩 stride。
                copy_ub_to_gm(dst_sf, ub_output_sf[stage_idx],
                              // scale按两pair×stage_valid_m行搬运，pair间GM跨度固定256*sizeof(int16)。
                              kNumOutputSFPairs, stage_valid_m, BLOCK_M * sizeof(int16_t), kWriteThroughShare);
                // 必须让 data/scale 的 MTE3 读取 UB 完成后再释放 stage；FIX 随后可用同一 UB 写下一任务。
                notify_intra_block<PIPE_MTE3, 1>(kEpilogueEmpty, stage_idx);
            }
            // 当前任务还有未处理行时返回true表示本小步有进展，但不能发布整任务Linear1Ready。
            if (epi.m_idx_in_block != epi.valid_m)
                return true;
            // 任务全M处理完后重置行游标，下一task重新从stage0开始。
            epi.m_idx_in_block = 0;
        }

        // 整个任务的有效 M 行都处理完才推进消费序号；GMM1 的 +1 对应一个完整 N128 激活任务。
        ++ epi.num_completed_gemms;

        // routed Linear1Ready/Linear2Ready 按 N-block 计数，分别等 I/128、H/256。
        // shared Linear2 特例：每核处理自己的最后一个 shared task 后只给 slot 0 加一次，早归并等待活跃核数。
        // Shared early reduce counts each active core once in slot 0
        // 每个完整GMM1任务贡献一份量化完成计数，不按32行小步或128行stage重复加。
        if (is_linear1)
            // shared从0编号，routed加shared实际块前缀；目标为各自层的N任务数。
            publish_block_progress(layout::kLinear1Ready, (is_shared ? 0 : num_shared_m_blocks) + pool_block_idx, 1);
        // shared GMM2 的每核最后任务判定为 task_idx+32>=shared task 总数；
        // 只有这一次加 slot0，避免同核多任务重复贡献而过早放行 shared reduce。
        else if (not is_shared or (kOverlapReduce and epi.task_idx + kNumAICores >= num_shared_m_blocks * kNumLinear2NBlocks))
            // routed每N任务加1；shared早归并只在每核最后shared GMM2任务后给slot0加1。
            publish_block_progress(layout::kLinear2Ready, is_shared ? 0 : num_shared_m_blocks + pool_block_idx, 1);
        // 当前task消费完成，置空phase让下一轮调度器选择新任务。
        epi.phase = sched::BlockPhase::kNone;
        // 返回true仅说明状态机推进了一步，整体是否结束由外层三个游标共同判定。
        return true;
    };

    // 当前rank的BF16 GMM2输出ring，本核push只读其owner行。
    const auto routed_output = workspace.routed.output.get_ptr(buffer);
    // push.pool_block_idx 是已排入发送的块前缀；num_completed_m_blocks 是已经等 stores 并发布 ready 的前缀。
    // 两者可以不同：允许下一块发送与此前块的远端写在途重叠。
    struct CombinePushState {
        // push专家游标与pull/epilogue独立，因为回传可以滞后多个逻辑块。
        sched::ExpertCursor<workspace> expert;
        // 待发送逻辑块号从0开始顺序推进，方便形成单调完成prefix。
        uint32_t pool_block_idx = 0;
        // 当前块下一条本核负责行号；按core_idx+n*32分配。
        uint32_t m_idx_in_block;
        // 当前专家块真实行数，决定空owner/尾行处理。
        uint32_t valid_m;
        // 本核已确认store结束并发布CombineSent的右开逻辑块前缀，可能小于已发块数。
        uint32_t num_completed_m_blocks = 0;
    };
    // 与pull相同owner划分，每核仅发送core_idx+n*32行，无重叠远端写。
    CombinePushState push{.m_idx_in_block = core_idx};

    // 等该 M-block 的全部 GMM2 N-block 完成后，按完整 BF16 hidden 行回传。
    // 每条 route 的目的地址为源 rank 的 [topk-slot][token][hidden]，不同 route 不做远端原子求和。
    // Send BF16 rows; return true when this step advances
    const auto advance_combine_push = [&]() __aicore__ -> bool {
        // Batch store waits to overlap remote writes with other work
        // 每 4 个 M256 块（32*32/256）合并一次 store 等待，减少每块的排空开销。
        // 尾块仍强制 flush；完成回传时再一次给这一批中的各块发布本核 +1。
        constexpr uint32_t kNumMBlocksPerWait = 32 * kNumAICores / BLOCK_M;
        // 仅在新块开始时核对全部 H/256 个 GMM2 输出 tile；一旦开始发送，同块数据无需重复等待。
        // 没有剩余块，或新块GMM2未全完成时立即返回false，不把Scalar卡在该ready上。
        if (push.pool_block_idx == num_total_m_blocks or (push.m_idx_in_block == core_idx and
                not comm::is_block_ready<workspace>(buffer, layout::kLinear2Ready,
                                                    // 当前块ready槽加实际shared块前缀，保持与AIC发布位置一致。
                                                    num_shared_m_blocks + push.pool_block_idx,
                                                    // 必须等整个H/256输出N块写完，不能只凭当前核GMM2完成就发送整H行。
                                                    kNumLinear2NBlocks)))
            return false;

        // 固定本次块号供metadata和payload寻址，后续游标更新不影响该局部值。
        const uint32_t pool_block_idx = push.pool_block_idx;
        // 只在当前块第一行确定有效M；块内后续两行小步沿用该值。
        if (push.m_idx_in_block == core_idx)
            // 回传与生产端使用同一个ExpertCursor公式，尾块不会发送padding行。
            push.valid_m = push.expert.advance_to(buffer, pool_block_idx);
        // source metadata按逻辑块号取，旧ring payload回绕不影响该全量表。
        const auto src_metadata_base = src_metadata + static_cast<uint64_t>(pool_block_idx) * BLOCK_M;
        // BF16 payload按ring物理块定位，完整行宽为H。
        const auto src_block = routed_output + layout::get_ring_m_idx<workspace>(pool_block_idx) * config.hidden;

        // Send at most two rows before yielding
        // 每次最多两行才返回外层，减轻combine独占AIV1对激活关键路径的影响。
        for (uint32_t step_idx = 0; step_idx < 2 and push.m_idx_in_block < push.valid_m; ++ step_idx) {
            // 本次待发的真实块内行号。
            const uint32_t m_idx_in_block = push.m_idx_in_block;
            // 本核负责的第n行使用n%4槽；每次调用最多发送2行，再返回调度器，避免 combine 独占 AIV1。
            const uint32_t stage_idx = (m_idx_in_block / kNumAICores) % kNumPushStages;
            // 读取dispatch保存的source rank和combine行号，恢复原top-k槽。
            const uint32_t combine_metadata = asc_load_dev(src_metadata_base + m_idx_in_block);
            // 从全量 src_metadata 取回目标槽，payload 则从 ring 取；两者不可都改用 ring 下标。
            // 取低24位得到topk_slot*Tmax+token，尚未乘H转换成元素偏移。
            const uint32_t combine_m_idx = combine_metadata & 0x00ffffffu;
            // 当前完整BF16行的本地GM起点。
            const auto src = src_block + m_idx_in_block * config.hidden;

            // Load a BF16 output row
            // 这是阻塞式DMA槽等待：若旧远端MTE3慢，当前AIV1在此期间不能返回处理新SwiGLU。
            wait<PIPE_MTE3, PIPE_MTE2>(kPushEmpty, stage_idx);
            // 先从本地GM把完整H宽BF16行读入所选push UB槽。
            copy_gm_to_ub(ub_push_token[stage_idx].ptr(), src, 1, config.hidden,
                          // 数据宽2B，传输字节H*2；该输出不作长期L2驻留。
                          config.hidden * sizeof(bfloat16_t), asc_load_l2_cache_mode::NOTALLOC_KEEP);
            // 本地GM读取完成后允许MTE3用该UB槽回传。
            notify<PIPE_MTE2, PIPE_MTE3>(kPushFull, stage_idx);

            // 取高8位得到原source rank，即该route结果的回传目标。
            const uint32_t dst_rank_idx = combine_metadata >> 24;
            // map 将同一 combine offset 平移到源 rank 的 peer VA；直接 MTE3 写它的独占 route 槽，无结果 AtomicAdd。
            const auto dst = sym_buffer.map(workspace.combine_input.get_ptr(buffer, static_cast<uint64_t>(combine_m_idx) * config.hidden), dst_rank_idx);

            // Send the BF16 row to its source token
            // 远端write前等本行MTE2读取完成，不把尚未填满的UB交给MTE3。
            wait<PIPE_MTE2, PIPE_MTE3>(kPushFull, stage_idx);
            // 直接向source rank的独占route槽写整BF16行，不做结果AtomicAdd。
            copy_ub_to_gm(dst, ub_push_token[stage_idx].ptr(), 1, config.hidden,
                          // 远端存储模式NOTALLOC_CLEAN；完成发布仍由后续事件与DSB保护，不能只靠cache hint。
                          config.hidden * sizeof(bfloat16_t), asc_store_l2_cache_mode::NOTALLOC_CLEAN);
            // 旧 remote write 消费完 UB 后才能复用 push 槽；4 槽只增加在途行数，不代表 4 条独立网络链路。
            notify<PIPE_MTE3, PIPE_MTE2>(kPushEmpty, stage_idx);
            // 下一行属于同核的+32行，而不是相邻行；其他行由其他AIV1发送。
            push.m_idx_in_block += kNumAICores;
        }

        // 本核当前块行已发完，即使core_idx>=valid_m没有实际行也要推进逻辑块。
        if (push.m_idx_in_block >= push.valid_m) {
            // 下一块从本核固定首行core_idx开始，下一轮再取该块有效M。
            push.m_idx_in_block = core_idx;
            // 已发块前缀先前进，store完成前缀稍后通过批量flush更新。
            ++ push.pool_block_idx;

            // Wait for batched stores; publish each last-expert block for early reduce
            // overlap 在最后专家的各块也主动 flush，以更频繁发布尾部前缀；普通路径只按4块/最终尾块 flush。
            if (push.pool_block_idx % kNumMBlocksPerWait == 0 or
                    // 普通路径仅最后块强制flush；overlap在最后专家的每块都flush以推进尾部prefix。
                    (kOverlapReduce ? push.expert.pool_block_end == num_total_m_blocks : push.pool_block_idx == num_total_m_blocks)) {
                // num_completed_m_blocks 为本核已排空回传的右开前缀；发布 [旧前缀, 当前块末尾) 所有逻辑块。
                publish_block_progress(layout::kCombineSent,
                                       // 这批ready范围从旧完成前缀开始，长度为已发前缀减旧完成前缀。
                                       num_shared_m_blocks + push.num_completed_m_blocks, push.pool_block_idx - push.num_completed_m_blocks);
                // publish内部已等MTE3，故本核的已完成store前缀可追到当前已发块末尾。
                push.num_completed_m_blocks = push.pool_block_idx;
            }
        }
        // 即使只推进空owner块或一两行，也算本次有进度。
        return true;
    };

    // 本 rank 的所有发送核都完成此前的块后，才向各源 rank 发布右开完成前缀。
    // 全 1 表示本 rank 全部回传完成；源端同时利用该标记保护下一次调用的 scratch 复用。
    // Publish the exact completed-block prefix shared by all source ranks
    struct CombinePrefixState {
        // 已经向source ranks发布的、全发送核一致完成的右开块前缀。
        uint32_t num_published_m_blocks = 0;
        // 该publisher是否已给它负责的source rank发全1终止标记。
        bool finished;
    };
    // publisher 按 source rank 分工；core_idx>=rank数时无需发布，rank>32时每核负责多个 source。
    CombinePrefixState ready_prefix{.finished = not kOverlapReduce or core_idx >= config.num_ranks};

    const auto publish_combine_prefix = [&]() __aicore__ {
        // 候选prefix只能来自本核已经flush的块数，不能取仅发DMA的push.pool_block_idx。
        const uint32_t pool_block_end = push.num_completed_m_blocks;
        // 完成后不重复写；候选前缀未增加时也不额外产生远端signal流量。
        if (ready_prefix.finished or (num_total_m_blocks != 0 and pool_block_end <= ready_prefix.num_published_m_blocks))
            return;
        // 各核按逻辑块顺序发布；检查最新前缀的末块收到全部 32 核贡献，就代表更早块也已全部完成。
        // 候选末块必须有全32核CombineSent贡献，证明每核之前各块也都顺序完成。
        if (pool_block_end != 0 and not comm::is_block_ready<workspace>(
                buffer, layout::kCombineSent, num_shared_m_blocks + pool_block_end - 1, kNumAICores))
            // 不满足时立即返回，主循环继续推进本核通信，之后再重试发布。
            return;

        // 这是该发布核已确认“全32个发送核完成”的统一前缀，不能仅发送本核 push.pool_block_idx。
        ready_prefix.num_published_m_blocks = pool_block_end;
        // 仅当全rank-local routed逻辑块都完成时，下一步编码全1哨兵。
        ready_prefix.finished = pool_block_end == num_total_m_blocks;
        // 有限值是右开块边界；~0ull 兼作“全部结束”哨兵，即使本 rank 没有任何 routed 块也要发送。
        const uint64_t prefix = ready_prefix.finished ? ~uint64_t{0} : pool_block_end;
        // 32核按source rank分工，每核可负责多个source，避免多publisher写同一个prefix槽。
        for (uint32_t src_rank_idx = core_idx; src_rank_idx < config.num_ranks; src_rank_idx += kNumAICores) {
            // payload完成和metadata先行可见，再发布producer prefix，建立源端读取的控制依赖。
            asc_sync_data_barrier(DSB_DDR);
            // 写入对端源 rank 的 prefix[本 producer rank]，每槽仅由指定 core_idx 的 publisher 写，避免多写者。
            // 写source端prefix[本producer rank]，每个source都能按route的owner找到正确信号。
            asc_store_dev(sym_buffer.map(workspace.metadata.combine_ready_prefix.get_ptr(buffer, rank_idx), src_rank_idx), prefix);
        }
    };

    // 通信轮转的初始偏好是pull，下一轮可切到push；不是固定先跑完所有dispatch再combine。
    bool prefer_push = false;
    // 谓词循环兼作协作调度器，返回true表示所有状态机与prefix均结束。
    comm::wait_until([&]() __aicore__ {
        // Service the reduce window; check token readiness inside
        // 只有接近回传尾段且本核所有已发 GEMM 都已消费时才插入 reduce；优先避免妨碍激活关键路径。
        const bool reduce_window_open = kOverlapReduce and metadata_ready and push.pool_block_idx + num_reduce_window_m_blocks >= num_total_m_blocks and
            // 本核已发GEMM与已完成epilogue数相等时，尾部reduce才不挤占已提交激活的消费机会。
            asc_load_dev(num_issued_gemms_ptr) == epi.num_completed_gemms;

        if (reduce_window_open) {
            // 尾段先尝试把全核完成prefix发给source，source的早归并才可能前进。
            publish_combine_prefix();
            // 只有 routed 块数超出 ring 才在通信循环借剩余 UB 做早归并；小工作量留给最终 offset0 的大 UB reduce。
            if (num_total_m_blocks > kNumRingBlocks)
                // 一次最多做两个channel chunk后回到调度循环；不是无限归并直到所有token齐备。
                advance_combine_reduce<workspace, kOverlapReduce>(buffer, output, num_tokens, reduce, ub_reduce_inputs, 2);
        }

        // Fetch the next epilogue task
        // 只有当前epilogue任务彻底消费完才取下一task，避免丢失其中未完成stage的行进度。
        if (epi.phase == sched::BlockPhase::kNone)
            epi.task_idx = sched::take_linear_task<workspace>(
                // 传入本地实际T允许shared先行；routed部分仍需metadata_ready与块数证书。
                num_tokens, num_total_m_blocks, epi.schedule, core_idx, epi.phase, metadata_ready);

        // Poll metadata arrival
        // AIV0 通过本 block 的 GM flag 通知 AIV1，读取 certificate 前做 DDR barrier；shared GMM1 可先行。
        if (not metadata_ready and epi.phase != sched::BlockPhase::kSharedLinear1 and
            // AIV1以GM flag发现配对AIV0完成metadata，不使用AIC那一路跨核事件。
            asc_load_dev(workspace.progress.metadata_ready.get_ptr(buffer, core_idx)) != 0) {
            // 读取证书前完成DDR顺序，确保看到发布flag之前的元数据结果。
            asc_sync_data_barrier(DSB_DDR);
            // 证书高位为AIV0到达数，仅取低位作为总块数。
            num_total_m_blocks = asc_load_dev(num_m_blocks_certificate) & layout::kCountValueMask;
            // 从此主循环可启用routed pull/push及routed epilogue task分配。
            metadata_ready = true;
        }

        // 只有metadata已确认才允许判定pull完成，否则初始总块数0会被误当空工作。
        const bool pull_finished = metadata_ready and pull.pool_block_idx == num_total_m_blocks;
        // 退出需要 pull、epilogue、push 都结束且 prefix 发布完成；后面的 combine_reduce 负责排空/远端收尾。
        // 三个状态机都结束且本publisher已发终止prefix，才结束AIV1服务循环。
        if (pull_finished and epi.phase == sched::BlockPhase::kNone and push.pool_block_idx == num_total_m_blocks and ready_prefix.finished)
            return true;

        // 优先处理可推进的 epilogue，避免通信占住 AIV1、阻塞 GMM2 所需激活。
        // push/pull 交替尝试；状态机在小步边界切换，已发出的 DMA 仍由各槽 credit 保护。
        // Advance epilogue before choosing push or pull
        // 每轮先推进一份已选epilogue小步；激活优先是循环边界优先，不能抢占内部阻塞wait。
        const bool epi_progress = advance_epilogue();
        // metadata到达后才通信；pull已结束且epilogue有进展时，本轮先让激活继续，减少无用通信探测。
        if (metadata_ready and not (pull_finished and epi_progress)) {
            // epilogue 已尝试后轮转通信：prefer_push 提供公平机会；push 未就绪则可退回 pull。
            // pull 的槽反压检查立即返回，但已发 DMA 仍可能通过 wait 阻塞；这是小步协作，不是任意指令抢占。
            if ((prefer_push or pull_finished) and advance_combine_push()) {
                // 成功push后下一轮优先pull，避免一方长期占用MTE资源。
                prefer_push = false;
            // 没有可push数据但pull未结束，退回推进dispatch。
            } else if (not pull_finished) {
                // Fall back to pull; prefer push again on the next step
                // 拉下一小批route或发布一个块ready；若ring未释放会立即返回。
                advance_dispatch_pull();
                // 本轮已给pull机会，下一轮再尝试push。
                prefer_push = true;
            }
        }
        // 本轮仍有未完成阶段，wait_until继续调用此谓词；全局超时负责诊断不前进状态。
        return false;
    }, [&]() __aicore__ { comm::print_local_timeout(rank_idx); });
    // 存在shared时完成共享结果相关的DDR顺序再进入最终归并。
    if constexpr (kHasSharedExperts)
        asc_sync_data_barrier(DSB_DDR);
}

// 模板参数固定rank/专家/注册容量/top-k/H/I，JIT为每套注册配置实例化相应布局与分支。
template <uint32_t kNumRanks, uint32_t kNumExperts, uint32_t kNumSharedExperts,
          uint32_t kNumMaxTokensPerRank, uint32_t kNumTopk, uint32_t kHidden, uint32_t kIntermediateHidden>
// mixed kernel配置1个AIC配2个AIV；output为本地最终BF16[T,H]，统计输出可空。
__global__ __mix__(1, 2) void mega_moe_impl(__gm__ bfloat16_t* output, __gm__ int32_t* cumulative_local_expert_recv_stats,
                                            // routed GMM1 packed FP4 data与配套int16 scale。
                                            __gm__ const uint8_t* l1_weights, __gm__ const int16_t* l1_weights_sf,
                                            // routed GMM2 packed FP4 data与配套int16 scale。
                                            __gm__ const uint8_t* l2_weights, __gm__ const int16_t* l2_weights_sf,
                                            // shared GMM1为预转换FP8权重，不走routed FP4展开。
                                            __gm__ const uint8_t* shared_l1_weights, __gm__ const int16_t* shared_l1_weights_sf,
                                            // shared GMM2同样为FP8路径，并把拼接的shared intermediate求和回H。
                                            __gm__ const uint8_t* shared_l2_weights, __gm__ const int16_t* shared_l2_weights_sf,
                                            // 实际T与clamp运行时传入；注册容量Tmax已在模板固定，不能用更大T调用。
                                            uint32_t num_tokens, float activation_clamp,
                                            // peer基址及偏移随launch传入，映射到本次存活的HCCL注册allocation。
                                            layout::SymBuffer<> sym_buffer) {
    // 编译期workspace包含所有GM区域offset/容量，与host申请空间用同一结构。
    constexpr auto& workspace = layout::kWorkspace<kNumRanks, kNumExperts, kNumSharedExperts,
                                                   kNumMaxTokensPerRank, kNumTopk, kHidden, kIntermediateHidden>;
    // 后续所有分支引用同一配置，保证发送、接收和索引计算一致。
    constexpr auto& config = workspace.config;

    // 阈值按注册容量而非本次实际 token 数选择；这是实现中的经验策略，不是硬件通用拐点。
    // overlap 主要在尾段利用空闲 UB 推进归并，并非所有阶段都无条件并发执行 reduce。
    // Measured 32 MiB combine-work threshold for tail reduction
    constexpr bool kOverlapReduce = static_cast<uint64_t>(config.num_max_tokens_per_rank) *
        // 按Tmax估计返回区是否超过32MiB，不按本次T或本rank实际专家行数切换。
        (config.num_topk + (config.num_shared_experts > 0)) * config.hidden * sizeof(bfloat16_t) > 32ull * 1024 * 1024;

    // 本rank负责的routed专家数，用于清接收count和专家ID->rank映射。
    constexpr uint32_t kNumExpertsPerRank = config.num_experts / config.num_ranks;

    // Block readiness reads the low byte of each completion counter
    // 完成计数仅轮询uint32低byte，GMM2每块N任务数必须<=255。
    static_assert(config.hidden / BLOCK_N <= UINT8_MAX and
                  // routed GMM1量化任务数I/128同样不得超过255。
                  config.intermediate_hidden / (BLOCK_N / 2) <= UINT8_MAX and
                  // shared GMM1任务数S*I/128也受相同低byte约束。
                  config.shared_intermediate_hidden / (BLOCK_N / 2) <= UINT8_MAX);

    // 初始化当前设备执行上下文/内建状态，必须先于各角色使用设备原语。
    asc_init();
    // 每个mixed block的AIC与两个AIV共享同一core_idx，建立配对事件和任务序列。
    const uint32_t core_idx = static_cast<uint32_t>(block_idx);

    // 本rank工作区真实GM基址，由SymBuffer根地址恢复。
    const auto buffer = sym_buffer.get_base_ptr<__gm__ uint8_t*>();
    // 当前EP rank编号，不等于设备核号。
    const uint32_t rank_idx = sym_buffer.rank_idx;
    // 源rank原始top-k索引视图，后面SIMT读取与最终reduce过滤都使用它。
    const auto topk_idx = workspace.input.topk_idx.get_ptr(buffer);
    // 整个rank routed块数证书地址，输入阶段清零，AIV0汇总后再通知其他角色。
    const auto num_m_blocks_certificate = workspace.metadata.num_m_blocks_certificate.get_ptr(buffer);

    // 将四组权重指针组合为统一描述，data与SF不得来自不同布局版本。
    const layout::Weights weights{{l1_weights, l1_weights_sf}, {l2_weights, l2_weights_sf},
                                  {shared_l1_weights, shared_l1_weights_sf}, {shared_l2_weights, shared_l2_weights_sf}};

    // AIC: GEMMs, then wait for both AIVs
    // AIC与AIV在同一kernel入口分流，AIC只执行矩阵计算与最终block等待。
    if ASCEND_IS_AIC {
        // 提交shared/routed交错GMM任务，FIX与AIV epilogue通过跨核credit协作。
        aic_gemm_pipeline<workspace>(buffer, num_tokens, rank_idx, weights, core_idx);
        // AIC GEMM 提交完后仍等待两份 AIV 的最终输出 MTE3 到达；所有角色结束才可把本次 kernel 视为完成。
        asc_sync_block_wait(PIPE_S, kCombineDoneID);
        // AIC等待两份AIV最终完成后结束本角色，后面token分工只由AIV执行。
        return;
    }

    // Partition tokens across AIVs
    // 同mixed block的AIV子核号0/1，分别对应metadata/权重与epilogue/通信角色。
    const uint32_t subblock_idx = static_cast<uint32_t>(asc_get_sub_block_id());
    // subblock 0 为权重/metadata 核，subblock 1 为 epilogue/通信核；归并 token 使用全局 64-AIV 索引分工。
    const uint32_t aiv_idx = 2 * core_idx + subblock_idx;

    // Overlap mode revisits round-robin tokens; the final-only path uses contiguous ranges
    // overlap采用交错token owner以便轮询；普通最终reduce采用连续token区间。
    const uint32_t first_token_idx = kOverlapReduce ? aiv_idx : min(aiv_idx * ceil_div(num_tokens, kNumAIVs), num_tokens);
    // 计算本AIV拥有多少原始token，尾核可能为0，不能读越界token。
    const uint32_t num_reduce_tokens = kOverlapReduce ?
        // 交错模式统计aiv_idx+n*64尚在T内的数量。
        (num_tokens > aiv_idx ? ceil_div(num_tokens - aiv_idx, kNumAIVs) : 0) :
        // 连续模式从first_token_idx开始最多ceil(T/64)个token，并截到实际T。
        min(ceil_div(num_tokens, kNumAIVs), num_tokens - first_token_idx);
    // 初始化本核归并游标，后面AIV0/AIV1分别保留各自进度直到最终reduce。
    CombineReduceState<workspace> reduce{.token_idx = first_token_idx, .num_remaining_tokens = num_reduce_tokens};

    // Host 已在同一 stream 清零 progress 区；device 在这里清零本次接收计数。
    // 随后的 rank barrier 确保所有接收端清零完成，源端才可发布新一轮 metadata/count。
    // Clear counts before metadata publication
    // 所有64个AIV分担清本地专家arrival/sum；不能让单个AIV假定专家数<=64。
    for (uint32_t expert_idx = aiv_idx; expert_idx < kNumExpertsPerRank; expert_idx += kNumAIVs)
        // 只清总arrival/sum；每源count和metadata后续会由所有source（含零count）覆盖。
        asc_store_dev(workspace.metadata.expert_recv_count_sum.get_ptr(buffer, expert_idx), uint32_t{0});
    // 证书整个rank仅一份，交给全局AIV0清零，避免多核清零覆盖别人新贡献。
    if (aiv_idx == 0)
        asc_store_dev(num_m_blocks_certificate, uint32_t{0});
    // 参与清总count的核先发布其Scalar stores；统一rank_barrier保护清零与后续远端发布。
    if (aiv_idx < kNumExpertsPerRank)
        asc_sync_data_barrier(DSB_DDR);
    // 所有rank都完成接收计数清零之后，才开始本轮metadata/count发送。
    comm::rank_barrier<workspace>(buffer, sym_buffer, aiv_idx);

    // AIV0: metadata and routed weights
    // subblock0负责本rank的路由metadata与其后routed权重供给。
    if (subblock_idx == 0) {
        // 每 worker 512 SIMT 线程；按 route 数启用最多 32 个 worker，T=0 也保留一个 worker 发布零 count。
        const uint32_t num_metadata_workers = min(max(ceil_div(num_tokens * config.num_topk, layout::kNumDispatchThreads), 1u), kNumAICores);
        // 只启用前num_metadata_workers个AIV0做SIMT扫描，其他核仍需参与后续证书汇总。
        if (core_idx < num_metadata_workers) {
            // SIMT reads UB offsets; metadata owns this scratch until the VF completes
            // 为SIMT复制peer地址差，VF无法直接把Scalar局部对象当其共享寻址表使用。
            __ubuf__ int64_t offsets[kNumRanks];
            // 仅写当前通信域的有效rank个数，不需要复制整个最大256rank数组。
            for (uint32_t i = 0; i < kNumRanks; ++ i)
                // 每项是peer base相对本rank base的有符号VA差。
                offsets[i] = sym_buffer.offsets[i];
            // Scalar 先填 UB peer offset 表，DSB_UB 后交给 SIMT VF；VF 执行期间该 scratch 不能改作权重槽。
            asc_sync_data_barrier(DSB_UB);
            // 调用512线程SIMT VF，产生源端专家压紧路由并向owner推送控制metadata。
            asc_vf_call<comm::build_dispatch_metadata<workspace, kOverlapReduce>>(
                // VF launch维度为一维512线程，不代表增加512个物理AIV。
                dim3(layout::kNumDispatchThreads, 1, 1),
                // worker身份用core_idx，活跃worker数显式传入，决定局部count前缀与grid屏障目标。
                buffer, topk_idx, num_tokens, rank_idx, core_idx, num_metadata_workers, offsets);
            // VF调用后发Vector->Scalar通知，Scalar后继不能抢占仍属于SIMT的UBscratch。
            notify<PIPE_V, PIPE_S>(kMetadataDone);
            // 等待metadata VF完成，再进入接收count汇总与权重流水。
            wait<PIPE_V, PIPE_S>(kMetadataDone);
        }

        // 即使当前核不做SIMT扫描，也须贡献证书arrival；返回统一routed总块数。
        const uint32_t num_total_m_blocks = comm::wait_dispatch_metadata<workspace>(buffer, rank_idx, core_idx, cumulative_local_expert_recv_stats);

        // Notify AIV1 via GM and AIC via event
        // 同一份 certificate 分别经 GM flag 通知 AIV1、经跨核事件通知 AIC，两种通知不可互相替代。
        asc_store_dev(workspace.progress.metadata_ready.get_ptr(buffer, core_idx), uint8_t{1});
        // 通知同block的AIC消耗metadata证书；AIV1则用前一行GMflag发现。
        notify_intra_block<PIPE_S, 0>(kMetadataFull);

        // AIV0复用metadata阶段UB为packed/NZ权重槽，必要时插入尾部reduce。
        aiv0_weight_pipeline<workspace, kOverlapReduce>(buffer, output, num_tokens, reduce, num_total_m_blocks, weights, core_idx);
    } else {
        // AIV1: epilogue and communication
        // AIV1小步推进激活、dispatch、combine和尾部归并，直到数据流水与prefix发布完成。
        aiv1_epilogue_comm<workspace, kOverlapReduce>(buffer, output, num_tokens, rank_idx, reduce, activation_clamp, core_idx, sym_buffer);
    }

    // 两份AIV最终各自把未完成原token归并完，并处理远端终止标记/收尾。
    combine_reduce<workspace, kOverlapReduce>(buffer, output, num_tokens, rank_idx, aiv_idx, reduce, sym_buffer);

    // Notify AIC after final output stores
    // 两份 AIV 都在最终 y store 之后到达；AIC 在等待两者之前不能返回，避免输出/通信仍在途。
    // 在本AIV最终MTE3队列尾到达block完成事件，AIC等两份AIV到达才可返回。
    asc_sync_block_arrive(PIPE_MTE3, kCombineDoneID);
}

} // namespace deep_gemm::mega_moe
