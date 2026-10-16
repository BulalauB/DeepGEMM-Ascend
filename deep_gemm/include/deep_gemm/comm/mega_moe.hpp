#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/mega_moe.hpp>

#include <c_api/asc_simd.h>
#include <simt_api/device_functions.h>

namespace deep_gemm::mega_moe::comm {

// The predicate may advance work; true means the wait is complete
template <typename pred_t, typename print_timeout_t>
// @param pred 返回true表示完成；可在探测中推进其他流水，不要求是纯条件读取。
// @param print_timeout 超时诊断回调，执行后trap；不能在回调里继续正常流水。
__aicore__ __forceinline__ void wait_until(const pred_t& pred, const print_timeout_t& print_timeout) {
    // 60 seconds at 1 GHz on Ascend 950DT
    constexpr uint64_t kNumTimeoutCycles = 60ull * 1'000'000'000;
    const auto start_cycle = static_cast<uint64_t>(asc_get_system_cycle());
    while (not pred()) {
        if (static_cast<uint64_t>(asc_get_system_cycle()) - start_cycle >= kNumTimeoutCycles) {
            print_timeout();
            trap();
        }
    }
}

__aicore__ inline void print_local_timeout(uint32_t rank_idx) {
    __asc_aicore::printf("MegaMoE local wait timeout: rank=%u core=%u\n", rank_idx, static_cast<uint32_t>(block_idx));
}

__aicore__ inline void print_peer_timeout(uint32_t rank_idx, uint32_t aiv_idx, uint32_t peer_idx,
                                         uint64_t prefix, bool wait_done) {
    __asc_aicore::printf("MegaMoE peer wait timeout: rank=%u aiv=%u peer=%u prefix=0x%llx wait_done=%u\n",
                        rank_idx, aiv_idx, peer_idx, static_cast<unsigned long long>(prefix), static_cast<uint32_t>(wait_done));
}

// 复用同一计数槽的翻转屏障：所有 worker 的原子增量之和恰为最高位，完成时低位归零。
// 旧值决定本轮目标相位；不能在某些 worker 已进入下一轮后由单个 worker 清零计数。
template <uint32_t kGridSyncIdx>
// @param counter_ptr 当前grid同步阶段的单个GM相位计数槽。
// @param rank_idx 本rank编号，仅用于诊断；worker_idx为SIMT worker编号[0,num_workers)。
// @param num_workers 本轮实际启动metadata VF的AIV0数，最多32；不是SIMT线程数512。
__simt_callee__ __forceinline__ void grid_sync(__gm__ uint32_t* counter_ptr, uint32_t rank_idx,
                                               uint32_t worker_idx, uint32_t num_workers) {
    // Worker deltas sum to the phase bit; low bits return to zero
    constexpr uint32_t kFinishSumTag = 0x80000000u;
    // 60 seconds of vector cycles at 1.65 GHz on Ascend 950DT
    constexpr uint64_t kNumTimeoutCycles = 60ull * 1'650'000'000;

    asc_threadfence();
    asc_syncthreads();
    if (num_workers == 1)
        return;
    if (threadIdx.x == 0) {
        const uint32_t old_value = asc_atomic_add(
            counter_ptr, worker_idx == 0 ? kFinishSumTag - (num_workers - 1) : 1u);
        const uint32_t expected_tag = (old_value ^ kFinishSumTag) & kFinishSumTag;
        const auto start_cycle = static_cast<uint64_t>(clock64());
        // SIMT cannot use the Scalar wait_until lambda
        while ((*reinterpret_cast<volatile __gm__ uint32_t*>(counter_ptr) & kFinishSumTag) != expected_tag) {
            if (static_cast<uint64_t>(clock64()) - start_cycle >= kNumTimeoutCycles) {
                const uint32_t current = *reinterpret_cast<volatile __gm__ uint32_t*>(counter_ptr);
                __asc_simt_vf::printf("MegaMoE grid sync timeout: rank=%u worker=%u grid=%u workers=%u current=0x%x expected_tag=0x%x\n",
                                     rank_idx, worker_idx, kGridSyncIdx, num_workers, current, expected_tag);
                __trap();
            }
        }
    }
    asc_syncthreads();
}

// 先汇合本 rank 全部 AIV，再由 AIV 0 向所有 peer 原子发信号，最后将放行结果扩散到本地。
// epoch 交替选择两个信号槽，每个槽轮流累加到 num_ranks、减回 0，避免每轮清零的竞争。
template <const auto& workspace, uint32_t kBarrierIdx = 0>
// @param buffer 本rank对称GM基址；sym_buffer保存本rank及所有peer的基址偏移。
// @param vec_core_idx 全局AIV编号2*core_idx+subblock，范围[0,64)，只有0号负责跨rank发信号。
// @tparam kBarrierIdx 屏障状态槽：0用于入口清理，1用于非overlap收尾；各rank调用顺序须一致。
__aicore__ __forceinline__ void rank_barrier(__gm__ uint8_t* buffer, const layout::SymBuffer<>& sym_buffer, uint32_t vec_core_idx) {
    constexpr auto& config = workspace.config;
    const uint32_t rank_idx = sym_buffer.rank_idx;
    // Native event waits still rely on the device watchdog
    // 本地inter-core事件14先汇合全部64个AIV，再由aiv_idx=0执行跨rank原子信号；AIC不参加此屏障。
    // epoch 0/1在槽0/1各加到P，epoch 2/3在槽0/1各减到0；下一轮重复，避免手工清零信号槽。
    // barrier 0用于入口计数清理，barrier 1用于非overlap归并前等待所有peer的结果写完。
    constexpr uint32_t kAllAIVSyncID = 14;
    asc_sync_pipe(PIPE_ALL);
    asc_sync_inter_arrive(PIPE_MTE3, kAllAIVSyncID);
    asc_sync_inter_wait(PIPE_S, kAllAIVSyncID);
    if constexpr (config.num_ranks == 1)
        return;
    if (vec_core_idx == 0) {
        const auto barrier = workspace.sync.rank_barriers.get_ptr(buffer, kBarrierIdx);
        const uint64_t epoch = asc_load_dev(&barrier->epoch);
        const int delta = epoch & 2 ? -1 : 1;
        const int target = delta < 0 ? 0 : static_cast<int>(config.num_ranks);
        asc_store_dev(&barrier->epoch, epoch + 1);
        const auto signal_ptr = &barrier->signals[epoch & 1];
        for (uint32_t dst_rank_idx = 0; dst_rank_idx < config.num_ranks; ++ dst_rank_idx) {
            const auto remote_base = sym_buffer.map(buffer, dst_rank_idx);
            const auto remote_barrier = workspace.sync.rank_barriers.get_ptr(remote_base, kBarrierIdx);
            asc_atomic_add(&remote_barrier->signals[epoch & 1], delta);
        }
        asc_sync_data_barrier(DSB_DDR);

        wait_until([&]() __aicore__ { return asc_load_dev(signal_ptr) == target; }, [&]() __aicore__ {
            __asc_aicore::printf("MegaMoE rank barrier timeout: rank=%u barrier=%u epoch=%llu signal=%d target=%d\n",
                   rank_idx, kBarrierIdx, static_cast<unsigned long long>(epoch), asc_load_dev(signal_ptr), target);
        });
    }
    asc_sync_pipe(PIPE_ALL);
    asc_sync_inter_arrive(PIPE_MTE3, kAllAIVSyncID);
    asc_sync_inter_wait(PIPE_S, kAllAIVSyncID);
}

template <const auto& workspace>
// @param buffer 本rank工作区基址；stage选择Dispatched/Linear1Ready/Linear2Ready/CombineSent。
// @param progress_slot_idx shared用本轮shared块号，routed用本轮shared块数+逻辑pool块号；不取模。
// @param num_expected_completions 预期贡献数，单位由stage决定：核数或N-task数，不能当作token数。
__aicore__ inline bool is_block_ready(__gm__ uint8_t* buffer, layout::ProgressStage stage,
                                      uint32_t progress_slot_idx, uint32_t num_expected_completions) {
    // Counts fit in the low byte of the atomic uint32 counter
    const auto completion = workspace.progress.blocks.template get_ptr<uint8_t>(buffer, workspace.get_progress_idx(stage, progress_slot_idx));
    return asc_load_dev(completion) >= num_expected_completions;
}

template <const auto& workspace>
__aicore__ inline void wait_for_block(__gm__ uint8_t* buffer, uint32_t rank_idx, layout::ProgressStage stage,
                                      uint32_t progress_slot_idx, uint32_t num_expected_completions) {
    wait_until([&]() __aicore__ {
        return is_block_ready<workspace>(buffer, stage, progress_slot_idx, num_expected_completions);
    }, [&]() __aicore__ { print_local_timeout(rank_idx); });
}

// Build expert-contiguous dispatch metadata
template <const auto& workspace, bool kOverlapReduce>
__simt_vf__ __aicore__ __launch_bounds__(layout::kNumDispatchThreads)
// 将 token/top-k 选择压紧为按专家连续的源路由表，并把索引与 FP32 权重位型打包发送。
// 阶段化处理：整个本rank直方图->worker前缀->全部staged索引->远端metadata写出->count/arrival发布。
// 不是扫描到一条top-k就立即向peer发送；三个grid_sync仅汇合本rank参与metadata的AIV0 worker。
// 先计数、计算 worker 前缀、写路由，再发布 count；接收方以 count 到达作为元数据可读条件。
// 无效专家 ID 不产生路由；零计数仍需发布到达贡献，防止 peer 永久等待空专家。
// @param buffer 本rank对称GM基址；topk_idx是有效[T,Ktop]的展平int64全局专家ID表。
// @param num_tokens 本rank本轮有效输入T；rank_idx是源rank编号[0,P)。
// @param worker_idx 当前AIV0 metadata worker号；num_workers是参与VF的worker数。
// @param offsets UB中的peer字节偏移表，offsets[p]=peer_base[p]-local_base，VF完成前必须有效。
// 输出写到buffer及peer对应字段：专家counts、连续路由项；overlap时还初始化route_m_blocks。
void build_dispatch_metadata(__gm__ uint8_t* buffer, __gm__ const int64_t* topk_idx,
                             uint32_t num_tokens, uint32_t rank_idx, uint32_t worker_idx, uint32_t num_workers,
                             __ubuf__ const int64_t* offsets) {
    constexpr auto& config = workspace.config;
    constexpr uint32_t kNumExpertsPerRank = config.num_experts / config.num_ranks;

    constexpr uint32_t kNumThreadsPerRound = kNumAICores * layout::kNumDispatchThreads;

    const auto grid_sync_ptr = workspace.sync.grid_sync_count.get_ptr(buffer);
    const auto expert_count = workspace.metadata.worker_expert_count.get_ptr(buffer);
    const auto total_expert_count = workspace.metadata.total_expert_count.get_ptr(buffer);
    const auto staged_token_topk_idx = workspace.metadata.staged_token_topk_idx.get_ptr(buffer);
    // Metadata owns this UB scratch until the VF completes
    __ubuf__ uint32_t expert_token_count[config.num_experts];
    static_assert(sizeof(expert_token_count) + config.num_ranks * sizeof(int64_t) <= UBSizeBytes, "Metadata UB overflow");

    const uint32_t thread_idx = threadIdx.x;
    const uint32_t num_token_topk = num_tokens * config.num_topk;

    // 前处理只发送路由描述，不搬运H维激活。每个worker为一个AIV0上的512线程SIMT VF，
    // worker数=min(max(ceil(T*topk/512),1),32)；空rank仍保留一个worker发布零计数。
    // UB直方图先统计本worker的各专家路由数，随后原地变成该worker在专家连续表中的写游标。
    // Count expert selections in UB
    for (uint32_t expert_idx = thread_idx; expert_idx < config.num_experts; expert_idx += layout::kNumDispatchThreads)
        expert_token_count[expert_idx] = 0;
    asc_syncthreads();

    // 关闭overlap时以worker数为步长轮转512-route批；开启时每worker取连续512倍数区间，
    // 使早期路由更集中。批内通过原子取槽，不能据此承诺token严格有序；归并依赖实际route_m_blocks。
    // With overlap reduce, assign contiguous route batches to each worker
    const uint32_t worker_stride = not kOverlapReduce ? layout::kNumDispatchThreads :
        (num_token_topk + kNumThreadsPerRound - 1) / kNumThreadsPerRound * layout::kNumDispatchThreads;
    const uint32_t worker_begin = worker_idx * worker_stride;
    const uint32_t batch_stride = (kOverlapReduce ? 1 : num_workers) * layout::kNumDispatchThreads;
    const uint32_t worker_end = worker_begin + worker_stride;
    const uint32_t token_topk_end = kOverlapReduce and worker_end < num_token_topk ? worker_end : num_token_topk;
    for (uint32_t token_topk_idx = worker_begin + thread_idx; token_topk_idx < token_topk_end; token_topk_idx += batch_stride) {
        // Clear route mappings before counts allow any peer to dispatch these inputs
        if constexpr (kOverlapReduce)
            *workspace.route_m_blocks.get_ptr(buffer, token_topk_idx) = ~uint32_t{0};
        const auto expert_idx = static_cast<uint64_t>(topk_idx[token_topk_idx]);
        if (expert_idx < config.num_experts)
            asc_atomic_add(expert_token_count + expert_idx, 1u);
    }
    asc_syncthreads();

    // Save counts, then reuse UB counters as write offsets
    for (uint32_t expert_idx = thread_idx; expert_idx < config.num_experts; expert_idx += layout::kNumDispatchThreads)
        expert_count[worker_idx * config.num_experts + expert_idx] = expert_token_count[expert_idx];
    // 第1次grid同步：所有worker直方图可读，之后各worker才能计算自己的exclusive前缀。
    grid_sync<0>(grid_sync_ptr, rank_idx, worker_idx, num_workers);

    for (uint32_t expert_idx = thread_idx; expert_idx < config.num_experts; expert_idx += layout::kNumDispatchThreads) {
        uint32_t slot_begin = 0;
        for (uint32_t prev_worker_idx = 0; prev_worker_idx < worker_idx; ++ prev_worker_idx)
            slot_begin += expert_count[prev_worker_idx * config.num_experts + expert_idx];
        if (worker_idx == num_workers - 1)
            total_expert_count[expert_idx] = slot_begin + expert_token_count[expert_idx];
        expert_token_count[expert_idx] = slot_begin;
    }
    asc_syncthreads();

    // Stage one route per thread in kNumDispatchThreads-route batches
    for (uint32_t batch_base = worker_begin; batch_base < token_topk_end; batch_base += batch_stride) {
        const uint32_t token_topk_idx = batch_base + thread_idx;
        if (token_topk_idx < token_topk_end) {
            const auto expert_idx = static_cast<uint64_t>(topk_idx[token_topk_idx]);
            if (expert_idx < config.num_experts) {
                const uint32_t dst_slot_idx = asc_atomic_add(expert_token_count + expert_idx, 1u);
                staged_token_topk_idx[static_cast<uint64_t>(expert_idx) * workspace.num_src_slots_per_expert + dst_slot_idx] = token_topk_idx;
            }
        }

        // Keep route batches ordered across iterations
        if (kOverlapReduce and batch_base + batch_stride < token_topk_end)
            asc_syncthreads();
    }
    // 第2次grid同步：专家连续的staged索引表写完，后面的warp可以读取任意worker产生的段。
    grid_sync<1>(grid_sync_ptr + 1, rank_idx, worker_idx, num_workers);

    // Distribute expert ranges across warps
    constexpr uint32_t kNumThreadsPerWarp = 32;
    const uint32_t warp_idx = thread_idx / kNumThreadsPerWarp;
    const uint32_t lane_idx = thread_idx % kNumThreadsPerWarp;
    const uint32_t global_warp_idx = warp_idx * num_workers + worker_idx;

    for (uint32_t expert_idx = global_warp_idx; expert_idx < config.num_experts; expert_idx += num_workers * (layout::kNumDispatchThreads / kNumThreadsPerWarp)) {
        const uint32_t count = total_expert_count[expert_idx];
        const uint32_t dst_rank_idx = expert_idx / kNumExpertsPerRank;
        const uint32_t local_expert_idx = expert_idx % kNumExpertsPerRank;
        const auto remote_base = layout::SymBuffer<>::map(buffer, dst_rank_idx, offsets);
        const uint64_t src_idx = static_cast<uint64_t>(expert_idx) * workspace.num_src_slots_per_expert;
        const uint64_t dst_idx = workspace.get_src_metadata_idx(local_expert_idx, rank_idx, 0);
        const auto remote_metadata = workspace.metadata.src_token_metadata.get_ptr(remote_base, dst_idx);
        const auto staged_indices = workspace.metadata.staged_token_topk_idx.get_ptr(buffer, src_idx);

        // Ship route weights with indices to avoid per-route remote Scalar reads.
        // 每路由8字节：高32位为FP32 top-k权重的原始bits，低32位为token*Ktop+slot；不含H维激活/scale。
        // expert与source rank由远端[local_expert,source_rank,slot]地址区间表达，不重复塞进每条uint64。
        for (uint32_t slot_idx = lane_idx; slot_idx < count; slot_idx += kNumThreadsPerWarp) {
            const uint32_t token_topk_idx = staged_indices[slot_idx];
            const uint32_t weight_bits = *workspace.input.topk_weights.template get_ptr<uint32_t>(buffer, token_topk_idx);
            remote_metadata[slot_idx] = (static_cast<uint64_t>(weight_bits) << 32) | token_topk_idx;
        }
    }
    // 第3次grid同步：各worker向peer写出的64位路由项完成并经过threadfence；
    // 之后worker0发布count/arrival，消费者才可按count去读取这些路由项。
    grid_sync<2>(grid_sync_ptr + 2, rank_idx, worker_idx, num_workers);

    // Publish counts after the metadata copy barrier
    // 本rank全部路由项经grid_sync<2>确认后才发布；不同源rank可处于不同阶段，接收方用arrival判断本地专家完整。
    if (worker_idx == 0) {
        for (uint32_t expert_idx = thread_idx; expert_idx < config.num_experts; expert_idx += layout::kNumDispatchThreads) {
            const uint32_t count = total_expert_count[expert_idx];
            const uint32_t dst_rank_idx = expert_idx / kNumExpertsPerRank;
            const uint32_t local_expert_idx = expert_idx % kNumExpertsPerRank;

            // Reset the peer prefix before publishing local expert 0's count
            if (kOverlapReduce and local_expert_idx == 0) {
                *workspace.metadata.combine_ready_prefix.get_ptr(buffer, dst_rank_idx) = 0;
                asc_threadfence();
            }

            const auto remote_base = layout::SymBuffer<>::map(buffer, dst_rank_idx, offsets);
            const uint64_t recv_count_idx = workspace.get_expert_recv_idx(rank_idx, local_expert_idx);
            *workspace.metadata.expert_recv_count.get_ptr(remote_base, recv_count_idx) = count;
            const auto remote_count = workspace.metadata.expert_recv_count_sum.get_ptr(remote_base, local_expert_idx);
            // Release the count store before publishing its arrival
            asc_threadfence();
            // 高8位累加1个来源到达，低24位累加该来源行数；count=0也必须加到达位。
            // host约束P*T_max*topk<2^24，保证行数不会进位污染到达字段；这是本轮证书而非epoch。
            asc_atomic_add(remote_count, layout::kCountArrivalBit | count);
        }
        asc_threadfence();
    }
}

// 返回本 rank 所有非空专家所需的逻辑 M-block 总数，不包含共享专家。
// 两层证书分别等待所有源 rank 的计数、所有 AIV0 的 M-block 贡献；空闲 core 也必须贡献到达位。
template <const auto& workspace>
// @param buffer 本rank工作区；rank_idx用于等待超时诊断；core_idx为AIV0对应AIC号[0,32)。
// @param[in,out] cumulative_local_expert_recv_stats 可空的int32[E/P]累计计数，只加有效路由行，不含padding/shared。
// @return 本rank本轮routed逻辑块总数B，各专家ceil(rows/256)之和。
__aicore__ __forceinline__ uint32_t wait_dispatch_metadata(__gm__ uint8_t* buffer, uint32_t rank_idx, uint32_t core_idx,
                                                           __gm__ int32_t* cumulative_local_expert_recv_stats) {
    constexpr auto& config = workspace.config;
    const auto num_m_blocks_certificate = workspace.metadata.num_m_blocks_certificate.get_ptr(buffer);

    // Wait for source counts and sum local M-blocks
    uint32_t num_contributed_m_blocks = 0;
    for (uint32_t expert_idx = core_idx; expert_idx < config.num_experts / config.num_ranks; expert_idx += kNumAICores) {
        const auto count_ptr = workspace.metadata.expert_recv_count_sum.get_ptr(buffer, expert_idx);
        uint32_t count;
        wait_until([&]() __aicore__ {
            return ((count = asc_load_dev(count_ptr)) & ~layout::kCountValueMask) == config.num_ranks * layout::kCountArrivalBit;
        }, [&]() __aicore__ {
            __asc_aicore::printf("MegaMoE metadata count timeout: rank=%u local_expert=%u arrivals=%u expected=%u\n",
                   rank_idx, expert_idx, count / layout::kCountArrivalBit, config.num_ranks);
        });
        count &= layout::kCountValueMask;
        // Accumulate received routes before padding; shared experts are excluded
        if (cumulative_local_expert_recv_stats != nullptr)
            asc_atomic_add(cumulative_local_expert_recv_stats + expert_idx, static_cast<int32_t>(count));
        num_contributed_m_blocks += ceil_div(count, layout::BLOCK_M);
    }
    asc_sync_data_barrier(DSB_DDR);

    // All AIV0 cores contribute, including idle cores
    // 这层只汇合本rank32个AIV0的块数；跨rank完整性已由各专家arrival==num_ranks证明，无发送后rank_barrier。
    asc_atomic_add(num_m_blocks_certificate, layout::kCountArrivalBit + num_contributed_m_blocks);
    uint32_t num_total_m_blocks;
    wait_until([&]() __aicore__ {
        return ((num_total_m_blocks = asc_load_dev(num_m_blocks_certificate)) & ~layout::kCountValueMask) == kNumAICores * layout::kCountArrivalBit;
    }, [&]() __aicore__ { print_local_timeout(rank_idx); });
    asc_sync_data_barrier(DSB_DDR);
    return num_total_m_blocks & layout::kCountValueMask;
}

} // namespace deep_gemm::mega_moe::comm
