#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/mega_moe.hpp>

namespace deep_gemm::mega_moe::sched {

// kNone means no current task
enum class BlockPhase : uint32_t {kNone, kLinear1, kLinear2, kSharedLinear1, kSharedLinear2};

// 四个游标分别记录 shared/routed 两层下一批任务的起点；不是同一公共游标。
struct LinearTaskState {
    uint32_t shared_l1_task_base = 0;
    uint32_t l1_task_base = 0;
    uint32_t l2_task_base = 0;
    uint32_t shared_l2_task_base = 0;
};

// 调度只决定任务次序和归属，输入是否可消费由调用方的完成计数判断。
// AIC、AIV0 权重生产者和 AIV1 epilogue 重放同一 routed 序列，避免任务/权重错配。
// Map logical M-blocks to nonempty local experts
template <const auto& workspace>
struct ExpertCursor {
    uint32_t pool_block_end = 0;  // Exclusive M-block end
    uint32_t local_expert_idx = ~uint32_t{0};  // First increment wraps to expert 0
    uint32_t num_expert_tokens = 0;

    __aicore__ void next_expert(uint32_t num_tokens) {
        ++ local_expert_idx;
        num_expert_tokens = num_tokens;
        // 专家块在逻辑 pool 内连续拼接；零 token 专家不移动 end，advance_to 会继续跳过它。
        pool_block_end += ceil_div(num_tokens, layout::BLOCK_M);
    }

    // 将专家行数按 FRAC_MN 行组均分到 ceil(M/256) 个块，减少末块过小的浪费。
    // 实际有效行数可小于 256，但 GM 每个逻辑块始终保留 256 行 stride。
    // Balance FRAC_MN groups; expert M is compact, pool blocks keep the BLOCK_M stride
    // Return valid_m and write the expert-relative M start
    __aicore__ uint32_t get_m_block_range(uint32_t pool_block_idx, uint32_t& expert_m_begin) const {
        const uint32_t num_expert_m_blocks = ceil_div(num_expert_tokens, layout::BLOCK_M);
        // pool_block_end - num_expert_m_blocks 是当前专家的起始块号；由此转成专家内块号。
        const uint32_t expert_m_block_idx = pool_block_idx + num_expert_m_blocks - pool_block_end;
        // 具体分块例：M=300，FRAC_MN=16，共 19 个行组、2 个 M-block。
        // 前块拿 10 组=160 行，后块拿 9 组但按真实剩余截为 140 行；不是简单 256+44。
        // expert_m_begin 是专家内紧凑行号，而实际 GM 块仍各占 256 行，两者不可混用作 stride。
        const uint32_t num_m_groups = ceil_div(num_expert_tokens, FRAC_MN);
        const uint32_t num_groups_per_block = num_m_groups / num_expert_m_blocks;
        // 行组不能均分时，前 remainder 个块各多分一个 FRAC_MN 组；最后用真实行数截断 padding。
        const uint32_t num_larger_m_blocks = num_m_groups % num_expert_m_blocks;

        expert_m_begin = (expert_m_block_idx * num_groups_per_block + min(expert_m_block_idx, num_larger_m_blocks)) * FRAC_MN;
        const uint32_t aligned_valid_m = (num_groups_per_block + (expert_m_block_idx < num_larger_m_blocks)) * FRAC_MN;
        return min(aligned_valid_m, num_expert_tokens - expert_m_begin);
    }

    // Advance the expert cursor and return valid rows
    __aicore__ uint32_t advance_to(__gm__ uint8_t* buffer, uint32_t pool_block_idx) {
        // 计数来自已汇总的 expert_recv_count_sum，只取低 24 位；跳过空专家直到逻辑块落入区间。
        while (pool_block_idx >= pool_block_end)
            next_expert(asc_load_dev(workspace.metadata.expert_recv_count_sum.get_ptr(buffer, local_expert_idx + 1)) &
                        layout::kCountValueMask);

        uint32_t expert_m_begin;
        return get_m_block_range(pool_block_idx, expert_m_begin);
    }
};

// Replay the same routed task order on AIC and AIVs
// With kNone, metadata_ready distinguishes a pending schedule from an exhausted one
template <const auto& workspace>
__aicore__ inline uint32_t take_linear_task(uint32_t num_shared_tokens, uint32_t num_total_m_blocks,
                                            LinearTaskState& state, uint32_t core_idx,
                                            BlockPhase& block_phase, bool metadata_ready = true) {
    constexpr auto& config = workspace.config;
    // GMM1 每任务投影 N256，但激活输出 N128，因此 task 数为 I/128；GMM2 每任务输出 N256。
    constexpr uint32_t kNumLinear1NBlocks = config.intermediate_hidden / (layout::BLOCK_N / 2);
    constexpr uint32_t kNumLinear2NBlocks = config.hidden / layout::BLOCK_N;
    constexpr uint32_t kNumSharedLinear1NBlocks = config.shared_intermediate_hidden / (layout::BLOCK_N / 2);
    constexpr bool kHasSharedExperts = config.num_shared_experts > 0;

    // Schedule shared Linear1 before shared Linear2
    if constexpr (kHasSharedExperts) {
        const uint32_t num_shared_m_blocks = ceil_div(num_shared_tokens, layout::BLOCK_M);
        #pragma unroll
        for (uint32_t layer_idx = 0; layer_idx < 2; ++ layer_idx) {
            auto& task_base = layer_idx == 0 ? state.shared_l1_task_base : state.shared_l2_task_base;
            const uint32_t num_tasks = num_shared_m_blocks * (layer_idx == 0 ? kNumSharedLinear1NBlocks : kNumLinear2NBlocks);
            if (task_base >= num_tasks)
                continue;

            // Advance past waves with no task for this core
            // 本核取 wave 起点加 core_idx；所有核游标统一前进 32，尾 wave 无任务核也跳过该 wave。
            const uint32_t task_idx = task_base + core_idx;
            task_base += kNumAICores;
            if (task_idx < num_tasks) {
                block_phase = layer_idx == 0 ? BlockPhase::kSharedLinear1 : BlockPhase::kSharedLinear2;
                return task_idx;
            }
        }
    }

    // Routed tasks require metadata
    // kNone 在 metadata 尚未到达时只表示暂时无 routed 任务，调用方必须继续等待而非退出。
    if (not metadata_ready) {
        block_phase = BlockPhase::kNone;
        return ~uint32_t{0};
    }

    // wave 指 kNumAICores 个线性 GEMM 任务，不是固定 256 个 token。
    // 先补齐下一 wave GMM2 所需 M-block 的 GMM1，再多做一 wave GMM1 作为前瞻，
    // 随后切到 GMM2；两层任务各自推进 task_base，不是跑完全部 GMM1 才执行 GMM2。
    // Prepare the next Linear2 wave with one Linear1 wave of lookahead
    const uint32_t num_routed_tasks[2]{num_total_m_blocks * kNumLinear1NBlocks, num_total_m_blocks * kNumLinear2NBlocks};

    while (state.l1_task_base < num_routed_tasks[0] or state.l2_task_base < num_routed_tasks[1]) {
        bool is_linear1 = state.l2_task_base >= num_routed_tasks[1];
        if (not is_linear1 and state.l1_task_base < num_routed_tasks[0]) {
            // 下一 GMM2 wave 的右开 task 边界 -> ceil_div 得到它覆盖的 M-block 右边界。
            // 乘 GMM1 的 N-block 数，得到必须完成的 GMM1 task 前缀；再加 32 个任务作一 wave 前瞻。
            const uint32_t required_linear1_task_end =
                ceil_div(min(state.l2_task_base + kNumAICores, num_routed_tasks[1]), kNumLinear2NBlocks) *
                kNumLinear1NBlocks;
            // 例 C=32、GMM1 每块24任务、GMM2 每块28任务：GMM2 base=0 时下一 wave 覆盖块0/1，
            // required GMM1 end=ceil(32/28)*24=48；判定界限再加32成为80，GMM1 base 按0/32/64推进。
            // 所以启动阶段先分配3个 GMM1 wave，再选 GMM2；“一 wave 前瞻”是任务数预算，并非恰好一个 M-block。
            // 该选择只分配任务，GMM2 真正读取仍要等覆盖块全部 GMM1 epilogue 的 I/128 完成计数。
            is_linear1 = state.l1_task_base < required_linear1_task_end + kNumAICores;
        }

        const uint32_t layer_idx = is_linear1 ? 0 : 1;
        auto& task_base = is_linear1 ? state.l1_task_base : state.l2_task_base;
        // 两层分别保持 task_idx 的 M-major 线性编号：task / Nblocks 是 M-block，task % Nblocks 是 N-block。
        // 选择层后，只推进该层游标；AIC/AIV0/AIV1 使用相同函数重放相同 routed 次序。
        const uint32_t task_idx = task_base + core_idx;
        task_base += kNumAICores;
        // 尾 wave 可能某核 task_idx 越界，此核在 while 中继续推进到另一层/下一 wave；
        // 各角色须用相同 core_idx 重放，不能自行跳过合法任务，否则权重与 FIX 输出的消费次序会错位。
        if (task_idx < num_routed_tasks[layer_idx]) {
            block_phase = is_linear1 ? BlockPhase::kLinear1 : BlockPhase::kLinear2;
            return task_idx;
        }
    }

    block_phase = BlockPhase::kNone;
    return ~uint32_t{0};
}

} // namespace deep_gemm::mega_moe::sched
