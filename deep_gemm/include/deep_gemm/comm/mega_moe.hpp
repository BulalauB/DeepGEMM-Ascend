#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/mega_moe.hpp>

#include <c_api/asc_simd.h>
#include <simt_api/device_functions.h>

namespace deep_gemm::mega_moe::comm {

// The predicate may advance work; true means the wait is complete
// 通用 Scalar 轮询器：pred 可以只检查标志，也可以推进状态机；返回 true 才结束本次等待。
template <typename pred_t, typename print_timeout_t>
__aicore__ __forceinline__ void wait_until(const pred_t& pred, const print_timeout_t& print_timeout) {
    // 60 seconds at 1 GHz on Ascend 950DT
    // Scalar 系统周期按实现假定的 1GHz 换算，超时上限为60秒；与 SIMT clock64 的时钟口径不同。
    constexpr uint64_t kNumTimeoutCycles = 60ull * 1'000'000'000;
    // 记录本次等待起始时刻，后面每轮都相对同一时刻计算耗时，不因谓词推进而重置。
    const auto start_cycle = static_cast<uint64_t>(asc_get_system_cycle());
    // 先调用谓词，再决定是否继续；谓词内部的 pull/push/epilogue 正是在此处被反复推进。
    while (not pred()) {
        // 若一直无法完成，检查是否超时；这只是失效诊断，不将未完成数据当作就绪。
        if (static_cast<uint64_t>(asc_get_system_cycle()) - start_cycle >= kNumTimeoutCycles) {
            // 调用方提供诊断上下文，例如卡号、核号、peer、当前 prefix 或专家到达数。
            print_timeout();
            // 终止异常 kernel，防止无限轮询掩盖跨核/跨 rank 进度故障。
            trap();
        }
    }
}

// 本 rank 内部事件/完成计数超时的统一诊断，core 为混合 block 的索引。
__aicore__ inline void print_local_timeout(uint32_t rank_idx) {
    __asc_aicore::printf("MegaMoE local wait timeout: rank=%u core=%u\n", rank_idx, static_cast<uint32_t>(block_idx));
}

// 跨 rank 回传前缀等待的诊断；wait_done 区分等某 token 可归并与等 peer 整体收尾。
__aicore__ inline void print_peer_timeout(uint32_t rank_idx, uint32_t aiv_idx, uint32_t peer_idx,
                                         uint64_t prefix, bool wait_done) {
    __asc_aicore::printf("MegaMoE peer wait timeout: rank=%u aiv=%u peer=%u prefix=0x%llx wait_done=%u\n",
                        rank_idx, aiv_idx, peer_idx, static_cast<unsigned long long>(prefix), static_cast<uint32_t>(wait_done));
}

// 每个元数据屏障调用点使用独立 counter；模板索引只用于区分超时诊断中的阶段。
template <uint32_t kGridSyncIdx>
__simt_callee__ __forceinline__ void grid_sync(__gm__ uint32_t* counter_ptr, uint32_t rank_idx,
                                               uint32_t worker_idx, uint32_t num_workers) {
    // Worker deltas sum to the phase bit; low bits return to zero
    // 只利用第31位作阶段翻转位，其余位用于尚未抵消的到达贡献；不是 route 数。
    constexpr uint32_t kFinishSumTag = 0x80000000u;
    // 60 seconds of vector cycles at 1.65 GHz on Ascend 950DT
    // SIMT 时钟假定1.65GHz，故同样60秒使用不同周期预算。
    constexpr uint64_t kNumTimeoutCycles = 60ull * 1'650'000'000;

    // 先发布各线程的 GM 写，再汇合本 worker 的线程；跨 worker 屏障不能替代这一步。
    asc_threadfence();
    asc_syncthreads();
    // 只有一个 worker 时无需跨 worker 原子汇合；前面的线程 fence/syncthreads 仍必须执行。
    if (num_workers == 1)
        return;
    // 每个 worker 只让一个线程写全局计数，其余线程停在末尾 syncthreads，避免重复到达。
    if (threadIdx.x == 0) {
        // 只有每个 worker 的 thread 0 更新计数；全部 delta 之和恰为 0x80000000。
        // 因此每次屏障翻转 phase bit、低位恢复为 0，不需要单独清零 counter。
        const uint32_t old_value = asc_atomic_add(
            // worker0 贡献 phase_bit-(N-1)，其余各贡献1；任意到达顺序下总和都是一次 phase 翻转。
            counter_ptr, worker_idx == 0 ? kFinishSumTag - (num_workers - 1) : 1u);
        // 用本次原子更新前的 phase 推导下一 phase；等待所有 worker 到达，而非只等本核写完。
        const uint32_t expected_tag = (old_value ^ kFinishSumTag) & kFinishSumTag;
        // 为本次 SIMT 屏障单独开始计时，不复用 Scalar 的系统时钟。
        const auto start_cycle = static_cast<uint64_t>(clock64());
        // SIMT cannot use the Scalar wait_until lambda
        // volatile GM 读取只检查 phase 高位；必须等其他 worker 的贡献抵消低位后才会翻转。
        while ((*reinterpret_cast<volatile __gm__ uint32_t*>(counter_ptr) & kFinishSumTag) != expected_tag) {
            if (static_cast<uint64_t>(clock64()) - start_cycle >= kNumTimeoutCycles) {
                // 只在超时分支取完整 counter，用于区分“缺到达”和“phase 不匹配”。
                const uint32_t current = *reinterpret_cast<volatile __gm__ uint32_t*>(counter_ptr);
                __asc_simt_vf::printf("MegaMoE grid sync timeout: rank=%u worker=%u grid=%u workers=%u current=0x%x expected_tag=0x%x\n",
                                     rank_idx, worker_idx, kGridSyncIdx, num_workers, current, expected_tag);
                __trap();
            }
        }
    }
    // thread0 确认跨 worker 屏障结束，再释放本 worker 的全部 SIMT 线程进入下一阶段。
    asc_syncthreads();
}

// 输入屏障使用默认索引0，普通最终归并使用索引1；两组 epoch/signal 不互相覆盖。
template <const auto& workspace, uint32_t kBarrierIdx = 0>
__aicore__ __forceinline__ void rank_barrier(__gm__ uint8_t* buffer, const layout::SymBuffer<>& sym_buffer, uint32_t vec_core_idx) {
    // workspace 是编译期注册布局，config 决定所有 rank 使用相同通信尺寸。
    constexpr auto& config = workspace.config;
    // 取本次 kernel 所属 rank，决定诊断信息与 symmetric peer 映射的本地基准。
    const uint32_t rank_idx = sym_buffer.rank_idx;
    // Native event waits still rely on the device watchdog
    // 预留 inter-AIV 事件14作为 rank 屏障的本地汇合点，与算子数据流水事件分开。
    constexpr uint32_t kAllAIVSyncID = 14;
    // 先排空各 AIV 的本地流水，再用 inter-core 事件汇合本 rank；只有全局 AIV 索引 0 向远端发信号。
    asc_sync_pipe(PIPE_ALL);
    // 每个 AIV 在本地 PIPE_MTE3 到达，确保此前该核的存储排在到达事件之前。
    asc_sync_inter_arrive(PIPE_MTE3, kAllAIVSyncID);
    // Scalar 等全部本地 AIV 到齐，之后全局 AIV0 才能代表本 rank 发布跨卡到达。
    asc_sync_inter_wait(PIPE_S, kAllAIVSyncID);
    // 单 rank 省略远端信号交换；已有的本地汇合仍然保留。
    if constexpr (config.num_ranks == 1)
        return;
    // 只有一个全局 AIV 执行跨卡 signal 原子加，其他 AIV 等末尾本地汇合。
    if (vec_core_idx == 0) {
        // 找到本调用点的 RankBarrier，所有 rank 使用相同区域偏移。
        const auto barrier = workspace.sync.rank_barriers.get_ptr(buffer, kBarrierIdx);
        // 读取连续调用的 epoch，选 signal 槽与加/减方向；每个 rank 的调用顺序必须一致。
        const uint64_t epoch = asc_load_dev(&barrier->epoch);
        // epoch 0/1 分别向 signals[0/1] 加一，epoch 2/3 分别减一，循环使用两槽。
        // 全 rank 到达目标在 num_ranks 与 0 间切换，避免屏障后另做一次远端清零。
        const int delta = epoch & 2 ? -1 : 1;
        // 加法代等待所有 rank 的总贡献 num_ranks；减法代等待原信号被全部减回0。
        const int target = delta < 0 ? 0 : static_cast<int>(config.num_ranks);
        // 本 rank 提前推进下次 epoch，但本次仍使用保存在局部变量中的旧 epoch 选择槽。
        asc_store_dev(&barrier->epoch, epoch + 1);
        // epoch最低位选两份 signal 中的一份，防止相邻代共用同一信号槽。
        const auto signal_ptr = &barrier->signals[epoch & 1];
        // 包括本 rank 在内逐一向每个 rank 贡献一次；本地等待的信号也会收到自己的贡献。
        for (uint32_t dst_rank_idx = 0; dst_rank_idx < config.num_ranks; ++ dst_rank_idx) {
            // 本地 allocation base 加 peer VA 差值得到远端同布局的基地址。
            const auto remote_base = sym_buffer.map(buffer, dst_rank_idx);
            // 在远端基地址上使用同一个 barrier索引，发送与等待地址公式配对。
            const auto remote_barrier = workspace.sync.rank_barriers.get_ptr(remote_base, kBarrierIdx);
            // 对所有 rank（含本 rank）贡献一次，接收端目标是全部参与 rank 的总和。
            asc_atomic_add(&remote_barrier->signals[epoch & 1], delta);
        }
        // 在开始检查本地总信号前发布当前核的 DDR 访问；不把本地 pipe 事件当远端信号本身。
        asc_sync_data_barrier(DSB_DDR);

        // 等待本 rank 的 signal 达到这一代目标，超时回调包含 epoch/目标/实际信号。
        wait_until([&]() __aicore__ { return asc_load_dev(signal_ptr) == target; }, [&]() __aicore__ {
            __asc_aicore::printf("MegaMoE rank barrier timeout: rank=%u barrier=%u epoch=%llu signal=%d target=%d\n",
                   rank_idx, kBarrierIdx, static_cast<unsigned long long>(epoch), asc_load_dev(signal_ptr), target);
        });
    }
    // 远端屏障由全局 AIV 索引 0 完成后，再汇合本 rank 所有 AIV，保证它们共同离开此阶段。
    asc_sync_pipe(PIPE_ALL);
    // AIV0 的跨卡阶段结束后，所有本地 AIV 再次到达同一 inter-core 汇合点。
    asc_sync_inter_arrive(PIPE_MTE3, kAllAIVSyncID);
    // 末尾 wait 将代表核的远端等待结果传播给本 rank 所有 AIV，统一离开屏障。
    asc_sync_inter_wait(PIPE_S, kAllAIVSyncID);
}

// 完成探测只读当前 logical progress slot，不等待；适合状态机决定是否切换到其他工作。
template <const auto& workspace>
__aicore__ inline bool is_block_ready(__gm__ uint8_t* buffer, layout::ProgressStage stage,
                                      uint32_t progress_slot_idx, uint32_t num_expected_completions) {
    // Counts fit in the low byte of the atomic uint32 counter
    // get_ptr<uint8_t> 只改变读取类型，不改变每槽的 kL2SectorBytes 地址步长。
    // 写端原子更新 uint32_t，读端仅轮询低字节；目标数量必须满足主 kernel 的 UINT8_MAX 约束。
    const auto completion = workspace.progress.blocks.template get_ptr<uint8_t>(buffer, workspace.get_progress_idx(stage, progress_slot_idx));
    // 完成次数单调增加，达到或超过目标均可消费；当前调用内计数不随 ring 回绕清零。
    return asc_load_dev(completion) >= num_expected_completions;
}

// 阻塞式封装：适用于 AIC 必须消费输入的位置；AIV 可选用上面的非阻塞探测。
template <const auto& workspace>
__aicore__ inline void wait_for_block(__gm__ uint8_t* buffer, uint32_t rank_idx, layout::ProgressStage stage,
                                      uint32_t progress_slot_idx, uint32_t num_expected_completions) {
    wait_until([&]() __aicore__ {
        // 由 stage+logical slot 找计数，num_expected_completions 由各阶段的生产者个数决定。
        return is_block_ready<workspace>(buffer, stage, progress_slot_idx, num_expected_completions);
    }, [&]() __aicore__ { print_local_timeout(rank_idx); });
}

// 路由前处理使用 SIMT：按 route 计数 -> worker 前缀 -> 按专家压紧索引 -> 远端写 metadata -> 发布 count。
// 这里只发送控制信息；FP8 token/scale 留在源端，由专家端的 AIV1 在后续 dispatch 拉取。
// 不按 (token, destination rank) 去重，重复专家/同卡多专家路由仍保留各自 top-k 槽。
// Build expert-contiguous dispatch metadata
// kOverlapReduce 决定是否维护 route->producer块映射和回传前缀，不改变专家路由的数学含义。
template <const auto& workspace, bool kOverlapReduce>
__simt_vf__ __aicore__ __launch_bounds__(layout::kNumDispatchThreads)
// buffer 是本 rank 注册空间，topk_idx 为原始 int64路由；offsets 为 SIMT 可读的 UB peer地址差表。
void build_dispatch_metadata(__gm__ uint8_t* buffer, __gm__ const int64_t* topk_idx,
                             // num_tokens 是实际T；worker_idx 是当前 AIV0的metadata worker号，num_workers 是本轮活跃worker数。
                             uint32_t num_tokens, uint32_t rank_idx, uint32_t worker_idx, uint32_t num_workers,
                             __ubuf__ const int64_t* offsets) {
    constexpr auto& config = workspace.config;
    // 专家按 rank 等量连续分配：expert / experts_per_rank 得owner，取余得到owner内编号。
    constexpr uint32_t kNumExpertsPerRank = config.num_experts / config.num_ranks;

    // 最多32核×每核512线程构成一轮route扫描预算，用于overlap连续区间分配。
    constexpr uint32_t kNumThreadsPerRound = kNumAICores * layout::kNumDispatchThreads;

    // 三个独立 grid counter 分别保护计数、索引和远端metadata三个阶段的发布。
    const auto grid_sync_ptr = workspace.sync.grid_sync_count.get_ptr(buffer);
    // GM二维表 [worker][global expert] 保存各worker histogram，供所有worker读取前缀。
    const auto expert_count = workspace.metadata.worker_expert_count.get_ptr(buffer);
    // 三个表的生命周期：worker_expert_count 是跨 worker 汇总 scratch；total_expert_count 决定发送长度；
    // staged_token_topk_idx 保存索引，接收端 src_token_metadata 则是最终的索引/权重二元组。
    const auto total_expert_count = workspace.metadata.total_expert_count.get_ptr(buffer);
    // 源端按专家分区的route索引暂存表；实际写入slot长度由有效route count决定。
    const auto staged_token_topk_idx = workspace.metadata.staged_token_topk_idx.get_ptr(buffer);
    // Metadata owns this UB scratch until the VF completes
    // UB histogram 所有512线程共享，后续原地改成该worker的专家内slot写指针。
    __ubuf__ uint32_t expert_token_count[config.num_experts];
    // UB 同时容纳专家直方图与peer offset表；编译期拒绝过大的布局，不依赖运行时越界。
    static_assert(sizeof(expert_token_count) + config.num_ranks * sizeof(int64_t) <= UBSizeBytes, "Metadata UB overflow");

    // 线程号用于划分专家清零、route扫描、warp内发送；它不是AIC核号。
    const uint32_t thread_idx = threadIdx.x;
    // 路由总条目为实际token数×topk，重复专家槽也算独立条目。
    const uint32_t num_token_topk = num_tokens * config.num_topk;

    // Count expert selections in UB
    // 以512线程步长分工清零全部专家，支持专家数大于单worker线程数的情况。
    for (uint32_t expert_idx = thread_idx; expert_idx < config.num_experts; expert_idx += layout::kNumDispatchThreads)
        expert_token_count[expert_idx] = 0;
    // 全部UB计数清零后才能允许其他线程原子累加。
    asc_syncthreads();

    // With overlap reduce, assign contiguous route batches to each worker
    // 普通路径：worker 交错处理 512-route batch；overlap 路径：worker 处理连续 batch 区间。
    // worker_begin/batch_stride 同时用于计数与索引写入，确保两次扫描覆盖完全一致的 route。
    const uint32_t worker_stride = not kOverlapReduce ? layout::kNumDispatchThreads :
        (num_token_topk + kNumThreadsPerRound - 1) / kNumThreadsPerRound * layout::kNumDispatchThreads;
    // overlap 时 token_topk_end 可能截在该 worker 的连续区间末；普通模式则一直扫描到 num_token_topk。
    // 这组边界贯穿两次 route 扫描，不能只改其中一次的分核规则。
    const uint32_t worker_begin = worker_idx * worker_stride;
    // 普通路径每轮跳过其他worker的batch；overlap路径在本worker连续区间逐batch前进。
    const uint32_t batch_stride = (kOverlapReduce ? 1 : num_workers) * layout::kNumDispatchThreads;
    // 当前worker连续区间的右开边界；普通模式不以此界限停止扫描。
    const uint32_t worker_end = worker_begin + worker_stride;
    // 尾worker区间截在真实route总数，不能对padding条目读topk或写映射。
    const uint32_t token_topk_end = kOverlapReduce and worker_end < num_token_topk ? worker_end : num_token_topk;
    // 每线程每次处理一个route；步长与后面构建索引阶段完全一致。
    for (uint32_t token_topk_idx = worker_begin + thread_idx; token_topk_idx < token_topk_end; token_topk_idx += batch_stride) {
        // Clear route mappings before counts allow any peer to dispatch these inputs
        if constexpr (kOverlapReduce)
            // 先将本源route映射标成无效；专家dispatch后才会回写其真实逻辑producer块号。
            *workspace.route_m_blocks.get_ptr(buffer, token_topk_idx) = ~uint32_t{0};
        // 转为无符号后，负 ID 也会超出专家范围而被跳过；每个有效 top-k 槽独立加一。
        // 这里没有 unique 判断：同 token 重复选择同专家会占用多个 route slot。
        const auto expert_idx = static_cast<uint64_t>(topk_idx[token_topk_idx]);
        // 仅合法专家ID参加count；负ID转无符号后越界，因此也自然过滤。
        if (expert_idx < config.num_experts)
            // UB原子加避免同一worker内多个线程命中同专家时丢失计数。
            asc_atomic_add(expert_token_count + expert_idx, 1u);
    }
    // 等待本worker所有route计数完成，再由分工线程读UB并保存GM。
    asc_syncthreads();

    // worker_expert_count 保存各 worker 计数；grid_sync 后可计算互不重叠的专家内 slot 区间。
    // UB histogram 随后复用为写指针，不能在保存计数前提前覆盖。
    // Save counts, then reuse UB counters as write offsets
    for (uint32_t expert_idx = thread_idx; expert_idx < config.num_experts; expert_idx += layout::kNumDispatchThreads)
        // 每worker只写自己的GM行，其他worker稍后只读它，不存在多写者覆盖。
        expert_count[worker_idx * config.num_experts + expert_idx] = expert_token_count[expert_idx];
    // 屏障0：所有 worker 的专家计数已落 GM，才可读前序 worker 数计算 exclusive prefix。
    grid_sync<0>(grid_sync_ptr, rank_idx, worker_idx, num_workers);

    for (uint32_t expert_idx = thread_idx; expert_idx < config.num_experts; expert_idx += layout::kNumDispatchThreads) {
        // 当前专家的exclusive prefix从0开始，累计本worker之前的计数。
        uint32_t slot_begin = 0;
        // 当前 worker 的起始 slot = 此专家所有前序 worker 的 count 之和。
        // 最后一个 worker 同时得到总 count；每个 worker 使用自己的不重叠区间。
        for (uint32_t prev_worker_idx = 0; prev_worker_idx < worker_idx; ++ prev_worker_idx)
            // 按相同expert列累加前序worker，得到当前worker拥有区间的起始slot。
            slot_begin += expert_count[prev_worker_idx * config.num_experts + expert_idx];
        // 只有最后worker写总count，避免不同worker保存不同的局部前缀值。
        if (worker_idx == num_workers - 1)
            total_expert_count[expert_idx] = slot_begin + expert_token_count[expert_idx];
        // UB计数由“本worker条目数”切换成“下一可分配slot”，原始计数已在GM备份。
        expert_token_count[expert_idx] = slot_begin;
    }
    // 所有线程完成UB写指针初始化后，再进行route原子slot分配。
    asc_syncthreads();

    // Stage one route per thread in kNumDispatchThreads-route batches
    for (uint32_t batch_base = worker_begin; batch_base < token_topk_end; batch_base += batch_stride) {
        // 当前batch起点加线程号恢复原token/topk扁平编号，完整身份保存在该32位索引里。
        const uint32_t token_topk_idx = batch_base + thread_idx;
        // 尾batch不满512条时，不读取越界topk。
        if (token_topk_idx < token_topk_end) {
            const auto expert_idx = static_cast<uint64_t>(topk_idx[token_topk_idx]);
            if (expert_idx < config.num_experts) {
                // UB 原子自增为命中同一专家的线程分配唯一 slot；batch 内顺序不保证按 token 排序。
                // 写入的仍是 token_topk_idx，后续不依赖 slot 顺序恢复 token/slot 身份。
                const uint32_t dst_slot_idx = asc_atomic_add(expert_token_count + expert_idx, 1u);
                // 专家基址采用最坏src_slots stride，dst_slot_idx只在该专家分区内移动。
                staged_token_topk_idx[static_cast<uint64_t>(expert_idx) * workspace.num_src_slots_per_expert + dst_slot_idx] = token_topk_idx;
            }
        }

        // Keep route batches ordered across iterations
        // overlap模式在连续batch间汇合线程，维持batch先后关系；batch内部原子次序仍可变化。
        if (kOverlapReduce and batch_base + batch_stride < token_topk_end)
            asc_syncthreads();
    }
    // 屏障1：total count 与所有 staged route 都已写好；后面 warp 可能发送其他 worker 写入的 slot。
    grid_sync<1>(grid_sync_ptr + 1, rank_idx, worker_idx, num_workers);

    // Distribute expert ranges across warps
    // SIMT warp固定32线程；同warp按lane分工复制某个专家的metadata。
    constexpr uint32_t kNumThreadsPerWarp = 32;
    // worker内部warp编号，范围0..15。
    const uint32_t warp_idx = thread_idx / kNumThreadsPerWarp;
    // warp内部lane编号，范围0..31，决定从哪个专家slot开始发。
    const uint32_t lane_idx = thread_idx % kNumThreadsPerWarp;
    // 以 warp 为单位分配专家，同 warp 的 lane 按步长 32 复制 route，避免多个 warp 重复发送。
    const uint32_t global_warp_idx = warp_idx * num_workers + worker_idx;

    // 以所有活跃worker的warp总数为步长，每个专家只由一个warp发送。
    for (uint32_t expert_idx = global_warp_idx; expert_idx < config.num_experts; expert_idx += num_workers * (layout::kNumDispatchThreads / kNumThreadsPerWarp)) {
        // 发送长度为源rank对该专家的总route数，不是T或topk固定长度。
        const uint32_t count = total_expert_count[expert_idx];
        // 全局专家编号整除本卡专家数，定位专家owner rank。
        const uint32_t dst_rank_idx = expert_idx / kNumExpertsPerRank;
        // 全局专家编号取余，定位owner卡上的本地专家。
        const uint32_t local_expert_idx = expert_idx % kNumExpertsPerRank;
        // SIMT 使用UB offsets 将本地workspace基址映射到专家owner的注册空间。
        const auto remote_base = layout::SymBuffer<>::map(buffer, dst_rank_idx, offsets);
        // 源索引scratch按全局专家编号分区，读取本专家实际slot前缀。
        const uint64_t src_idx = static_cast<uint64_t>(expert_idx) * workspace.num_src_slots_per_expert;
        // 接收端布局为 [本地专家][源 rank][该源的专家内 slot]；每个源 rank 只写自己的区间。
        const uint64_t dst_idx = workspace.get_src_metadata_idx(local_expert_idx, rank_idx, 0);
        // 目的metadata只写[local expert][本source rank]分区；其他source写自己的区间。
        const auto remote_metadata = workspace.metadata.src_token_metadata.get_ptr(remote_base, dst_idx);
        // 取当前全局专家的staged索引前缀，warp lane将以32为步长读。
        const auto staged_indices = workspace.metadata.staged_token_topk_idx.get_ptr(buffer, src_idx);

        // Ship route weights with indices to avoid per-route remote Scalar reads.
        for (uint32_t slot_idx = lane_idx; slot_idx < count; slot_idx += kNumThreadsPerWarp) {
            // 读回原始token*topk+j；不在接收端依赖原子分配slot的顺序推断token身份。
            const uint32_t token_topk_idx = staged_indices[slot_idx];
            // 按 uint32 原样搬运 float 位模式，高位不是数值转换；接收端再将这些位写入 float 存储。
            // 索引和权重合成一次 64-bit 写，避免 dispatch 时逐 route 远端读取权重。
            const uint32_t weight_bits = *workspace.input.topk_weights.template get_ptr<uint32_t>(buffer, token_topk_idx);
            // 例 token=7、topk=6、槽j=2，低位保存44；接收端 44/6=7、44%6=2。
            // 高位保留 weight 的 float 位模式，即使本 token 的其他槽也命中该卡，仍再发送独立 metadata。
            remote_metadata[slot_idx] = (static_cast<uint64_t>(weight_bits) << 32) | token_topk_idx;
        }
    }
    // 屏障2：所有 metadata 远端写完成发布后，worker0 才向专家 owner 发 count；
    // 接收方以 arrival 为就绪证明，不能出现 count 先发布但 metadata 仍在写的顺序。
    grid_sync<2>(grid_sync_ptr + 2, rank_idx, worker_idx, num_workers);

    // 发布顺序是 metadata 数据 -> grid_sync/threadfence -> 每源 count -> arrival 原子计数。
    // count_sum 高 8 位编码到达源 rank 数，低 24 位编码 route 总数；零 route 也必须报告到达。
    // 接收端看到全部 arrival 后才能依赖相应 metadata，不能仅检查总 count 是否非零。
    // Publish counts after the metadata copy barrier
    // 所有metadata已发布后只保留worker0发送count，其余worker无需重复贡献arrival。
    if (worker_idx == 0) {
        // worker0内部按线程分配专家，即使count=0也会发布一次arrival。
        for (uint32_t expert_idx = thread_idx; expert_idx < config.num_experts; expert_idx += layout::kNumDispatchThreads) {
            const uint32_t count = total_expert_count[expert_idx];
            const uint32_t dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const uint32_t local_expert_idx = expert_idx % kNumExpertsPerRank;

            // Reset the peer prefix before publishing local expert 0's count
            if (kOverlapReduce and local_expert_idx == 0) {
                // 先清本 rank 所见 dst_rank 的旧回传前缀，再发布送往它的路由 count。
                // 对方收到所有 count 后才可能 dispatch/回传，避免新前缀先到、却被本轮清零覆盖。
                *workspace.metadata.combine_ready_prefix.get_ptr(buffer, dst_rank_idx) = 0;
                asc_threadfence();
            }

            const auto remote_base = layout::SymBuffer<>::map(buffer, dst_rank_idx, offsets);
            // 计数表布局[source rank][local expert]，与接收端dispatch取数公式一致。
            const uint64_t recv_count_idx = workspace.get_expert_recv_idx(rank_idx, local_expert_idx);
            // 先写当前source的独立count，接收端用于把route拼成source区间。
            *workspace.metadata.expert_recv_count.get_ptr(remote_base, recv_count_idx) = count;
            // 总计数器按本地专家独占512B stride，避免不同专家的高频atomic混用同一槽。
            const auto remote_count = workspace.metadata.expert_recv_count_sum.get_ptr(remote_base, local_expert_idx);
            // Release the count store before publishing its arrival
            asc_threadfence();
            // kCountArrivalBit | count：低 24 位相加累积行数，高位每源贡献一次 arrival。
            // API 限制最坏 route 数不超过低位范围，防止 carry 污染 arrival 位。
            asc_atomic_add(remote_count, layout::kCountArrivalBit | count);
        }
        // 本worker完成全部专家count发布后再做threadfence，后继阶段才可复用其scratch。
        asc_threadfence();
    }
}

template <const auto& workspace>
__aicore__ __forceinline__ uint32_t wait_dispatch_metadata(__gm__ uint8_t* buffer, uint32_t rank_idx, uint32_t core_idx,
                                                           __gm__ int32_t* cumulative_local_expert_recv_stats) {
    constexpr auto& config = workspace.config;
    // certificate 既编码本rank AIV0到达数，也编码各专家逻辑块数之和。
    const auto num_m_blocks_certificate = workspace.metadata.num_m_blocks_certificate.get_ptr(buffer);

    // Wait for source counts and sum local M-blocks
    // 此函数计算整个 rank 的静态调度边界，不做 payload 传输；count 完整后才允许专家游标读取。
    // 当前核只累加自己负责专家的M-block数；后面再跨AIV0汇总。
    uint32_t num_contributed_m_blocks = 0;
    for (uint32_t expert_idx = core_idx; expert_idx < config.num_experts / config.num_ranks; expert_idx += kNumAICores) {
        // 每个专家各有独立arrival/sum计数，避免空专家因低位0被误判未到达。
        const auto count_ptr = workspace.metadata.expert_recv_count_sum.get_ptr(buffer, expert_idx);
        // 轮询时保存整份packed count；退出后需屏蔽高位才可作为route数使用。
        uint32_t count;
        // 每个 AIV0 负责 expert_idx = core_idx + n*核数；等该专家的全部源 rank 到达。
        // 只有 arrival 数达到 num_ranks，低位总行数和每源 metadata 才可用于分块。
        wait_until([&]() __aicore__ {
            return ((count = asc_load_dev(count_ptr)) & ~layout::kCountValueMask) == config.num_ranks * layout::kCountArrivalBit;
        }, [&]() __aicore__ {
            __asc_aicore::printf("MegaMoE metadata count timeout: rank=%u local_expert=%u arrivals=%u expected=%u\n",
                   rank_idx, expert_idx, count / layout::kCountArrivalBit, config.num_ranks);
        });
        // 去掉高位arrival，只保留本专家有效route行数，不含M块padding。
        count &= layout::kCountValueMask;
        // Accumulate received routes before padding; shared experts are excluded
        // 统计tensor可为空；启用时累加真实route数，支持多次调用累计统计。
        if (cumulative_local_expert_recv_stats != nullptr)
            // 当前专家由唯一AIV0处理，但使用atomic是为了累计统计的更新语义。
            asc_atomic_add(cumulative_local_expert_recv_stats + expert_idx, static_cast<int32_t>(count));
        // 空专家贡献 0 块；非空专家的尾块不能与下一专家共用，分别向上取整。
        num_contributed_m_blocks += ceil_div(count, layout::BLOCK_M);
    }
    // 发布本核贡献前完成DDR顺序，确保相关专家计数的读取/处理位于证书更新之前。
    asc_sync_data_barrier(DSB_DDR);

    // 每个专家单独向上取整到 M256，因此不能对总 route 数只做一次 ceil_div。
    // certificate 汇总本 rank 的逻辑块总数和 AIV0 到达数，供所有 AIC/AIV1 使用相同任务边界。
    // All AIV0 cores contribute, including idle cores
    // 每个 AIV0（含不负责专家的空闲核）贡献一次到达，以及自己负责专家的块数。
    // 消费者等 kNumAICores 次贡献后取得统一 num_total_m_blocks，避免各核任务序列分歧。
    asc_atomic_add(num_m_blocks_certificate, layout::kCountArrivalBit + num_contributed_m_blocks);
    // 临时保存certificate整字；下一轮轮询会更新它直到所有32个AIV0贡献到达。
    uint32_t num_total_m_blocks;
    wait_until([&]() __aicore__ {
        return ((num_total_m_blocks = asc_load_dev(num_m_blocks_certificate)) & ~layout::kCountValueMask) == kNumAICores * layout::kCountArrivalBit;
    }, [&]() __aicore__ { print_local_timeout(rank_idx); });
    // 读取完整certificate后完成DDR屏障，调用方再向AIC/AIV1发布metadata_ready。
    asc_sync_data_barrier(DSB_DDR);
    // 返回纯逻辑M-block数，shared块不在此计数内，由调用方单独加入进度索引前缀。
    return num_total_m_blocks & layout::kCountValueMask;
}

} // namespace deep_gemm::mega_moe::comm
