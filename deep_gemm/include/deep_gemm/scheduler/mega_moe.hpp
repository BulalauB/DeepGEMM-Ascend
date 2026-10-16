#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/mega_moe.hpp>

namespace deep_gemm::mega_moe::sched {

// kNone means no current task
enum class BlockPhase : uint32_t {kNone, kLinear1, kLinear2, kSharedLinear1, kSharedLinear2};

// 调度 wave = 最多 kNumAICores=32 个 (M-block,N-block) GEMM task；base 每次加32，
// task_idx=base+core_idx，M=task_idx/Nblocks、N=task_idx%Nblocks。没有 wave 末尾全核 barrier。
// GMM1 一个 N256 task 包含 gate128/up128，故 N1=I/128；GMM2 N2=H/256。
// 同一 wave 可能跨 M-block/专家；不能将32 task解释为32 token或固定数量的专家。
// 此处 wave 与 GM ring 的32/128个 M-block、K-loop 的多级 stage 是三种不同单位。
struct LinearTaskState {
    uint32_t shared_l1_task_base = 0;
    uint32_t l1_task_base = 0;
    uint32_t l2_task_base = 0;
    uint32_t shared_l2_task_base = 0;
};

// Map logical M-blocks to nonempty local experts
template <const auto& workspace>
struct ExpertCursor {
    uint32_t pool_block_end = 0;  // Exclusive M-block end
    uint32_t local_expert_idx = ~uint32_t{0};  // First increment wraps to expert 0
    uint32_t num_expert_tokens = 0;

    // @param num_tokens 下一个本地专家从全部rank接收的有效路由行数，区别于kernel入口的本rank输入T。
    __aicore__ void next_expert(uint32_t num_tokens) {
        ++ local_expert_idx;
        num_expert_tokens = num_tokens;
        pool_block_end += ceil_div(num_tokens, layout::BLOCK_M);
    }

    // Balance FRAC_MN groups; expert M is compact, pool blocks keep the BLOCK_M stride
    // Return valid_m and write the expert-relative M start
    // 先按 FRAC_MN 将专家行数分组，再均分到逻辑 M-block，避免只有最后一块承担很小的尾部。
    // expert_m_begin 是专家内紧凑行偏移；pool_block_idx 仍用固定 BLOCK_M 存储槽编号。
    // @param pool_block_idx 当前专家所属的全局逻辑pool块号，未做ring取模。
    // @param[out] expert_m_begin 此块在当前专家有效行序列中的起点，以行为单位。
    // @return 当前块有效行数；GM仍按BLOCK_M=256行分配，实际专家行区间按16行组均分。
    __aicore__ uint32_t get_m_block_range(uint32_t pool_block_idx, uint32_t& expert_m_begin) const {
        const uint32_t num_expert_m_blocks = ceil_div(num_expert_tokens, layout::BLOCK_M);
        const uint32_t expert_m_block_idx = pool_block_idx + num_expert_m_blocks - pool_block_end;
        const uint32_t num_m_groups = ceil_div(num_expert_tokens, FRAC_MN);
        const uint32_t num_groups_per_block = num_m_groups / num_expert_m_blocks;
        const uint32_t num_larger_m_blocks = num_m_groups % num_expert_m_blocks;

        // 例如专家257行：2个逻辑块、17个16行组，实际分为144和113行，而非256和1。
        // 两个块仍各保留256行 GM stride；valid_m<=128 的块可选 M128 MAD，减轻尾部浪费。
        expert_m_begin = (expert_m_block_idx * num_groups_per_block + min(expert_m_block_idx, num_larger_m_blocks)) * FRAC_MN;
        const uint32_t aligned_valid_m = (num_groups_per_block + (expert_m_block_idx < num_larger_m_blocks)) * FRAC_MN;
        return min(aligned_valid_m, num_expert_tokens - expert_m_begin);
    }

    // Advance the expert cursor and return valid rows
    // @param buffer 本rank对称GM基址，读取已就绪的expert_recv_count_sum。
    // @param pool_block_idx 单调前进的逻辑routed块号；本函数更新专家游标并返回该块有效M。
    __aicore__ uint32_t advance_to(__gm__ uint8_t* buffer, uint32_t pool_block_idx) {
        while (pool_block_idx >= pool_block_end)
            next_expert(asc_load_dev(workspace.metadata.expert_recv_count_sum.get_ptr(buffer, local_expert_idx + 1)) &
                        layout::kCountValueMask);

        uint32_t expert_m_begin;
        return get_m_block_range(pool_block_idx, expert_m_begin);
    }
};

// Replay the same routed task order on AIC and AIVs
// With kNone, metadata_ready distinguishes a pending schedule from an exhausted one
// 每次返回当前 core 的一个 M/N task，并推进整波 task_base；AIC 与 AIV 独立重放同一顺序。
// kNone 可能表示元数据尚未到达，也可能表示全部任务耗尽，调用方需结合 metadata_ready 判断。
// Linear1 的 lookahead 覆盖下一波 Linear2 所需的 M-block，实际输入可读仍由 progress 保护。
template <const auto& workspace>
// @param num_shared_tokens 本rank实际输入T；AIV0传0以跳过由AIC自行供权的shared任务。
// @param num_total_m_blocks 本rank全部routed专家的逻辑M-block总数，不含shared及ring容量。
// @param[in,out] state 当前角色独立维护的四阶段task base；按32 task推进，跨调用保留。
// @param core_idx AIC及其配对AIV使用的核号[0,32)，不是64个AIV中的全局编号。
// @param[out] block_phase 返回任务所属GMM阶段；metadata_ready表示routed计数证书已完整。
// @return 该阶段展平的M/N task号；无任务返回UINT32_MAX，需结合metadata_ready区分等待和结束。
__aicore__ inline uint32_t take_linear_task(uint32_t num_shared_tokens, uint32_t num_total_m_blocks,
                                            LinearTaskState& state, uint32_t core_idx,
                                            BlockPhase& block_phase, bool metadata_ready = true) {
    constexpr auto& config = workspace.config;
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
            const uint32_t task_idx = task_base + core_idx;
            task_base += kNumAICores;
            if (task_idx < num_tasks) {
                block_phase = layer_idx == 0 ? BlockPhase::kSharedLinear1 : BlockPhase::kSharedLinear2;
                return task_idx;
            }
        }
    }

    // Routed tasks require metadata
    if (not metadata_ready) {
        block_phase = BlockPhase::kNone;
        return ~uint32_t{0};
    }

    // 例 H=4096,I=3072：N2=16,N1=24；首个GMM2 wave覆盖2个M-block，需要48个GMM1 task。
    // 再领先32 task，GMM1 base须达到80；按32向上推进到96后才发GMM2 [0,32)。
    // 因此启动为G1 [0,32)/[32,64)/[64,96)，随后G2 [0,32)；并非固定G1/G2一比一交替。
    // 这里只确定任务顺序；Dispatched/Linear1Ready 决定实际数据何时能读，ring credit限制领先距离。
    // Prepare the next Linear2 wave with one Linear1 wave of lookahead
    // DeepSeek-V4-Flash H4096/I2048: N1=N2=16，一波32 task覆盖两块；core c的N=c%16、M=2*wave+c/16。
    // routed块足够时先G1 wave0/1，再G2 wave0，稳态G1 wave(j+1)->G2 wave(j)；没有wave末尾全核barrier。
    // 这不是按ready动态选任务；Flash core0先G1块0、再G1块2、再G2块0，块2未Dispatched可在AIC内挡住G2块0。
    // 只剩一个M-block时尾波仅16核有任务，其余核跳过；本轮总块数已由metadata确定，不等下一次launch来补满。
    // M256是每块行容量；实际M由expert接收数决定，EP8/T96严格均衡时每专家18行，MAD的M按16对齐为32。
    const uint32_t num_routed_tasks[2]{num_total_m_blocks * kNumLinear1NBlocks, num_total_m_blocks * kNumLinear2NBlocks};

    while (state.l1_task_base < num_routed_tasks[0] or state.l2_task_base < num_routed_tasks[1]) {
        bool is_linear1 = state.l2_task_base >= num_routed_tasks[1];
        if (not is_linear1 and state.l1_task_base < num_routed_tasks[0]) {
            // 将下一波 Linear2 task 的末尾除以 Linear2 的 N-block 数并向上取整，得到它涉及的 M-block 前缀。
            // 再乘 Linear1 的 N-block 数，得到必须先生产的完整激活列任务范围，最后额外领先一波 AIC数 的 Linear1任务。
            // lookahead 给激活尾部留时间，不是取消 GMM2 wait：progress 仍须等对应 M-block 的所有 N 列激活写回。
            // 任务 wave 由 task_base 每次加 AIC数推进，即使某个 core 在尾波没有 task，也不应停住该阶段游标。
            const uint32_t required_linear1_task_end =
                ceil_div(min(state.l2_task_base + kNumAICores, num_routed_tasks[1]), kNumLinear2NBlocks) *
                kNumLinear1NBlocks;
            is_linear1 = state.l1_task_base < required_linear1_task_end + kNumAICores;
        }

        const uint32_t layer_idx = is_linear1 ? 0 : 1;
        auto& task_base = is_linear1 ? state.l1_task_base : state.l2_task_base;
        const uint32_t task_idx = task_base + core_idx;
        task_base += kNumAICores;
        if (task_idx < num_routed_tasks[layer_idx]) {
            block_phase = is_linear1 ? BlockPhase::kLinear1 : BlockPhase::kLinear2;
            return task_idx;
        }
    }

    block_phase = BlockPhase::kNone;
    return ~uint32_t{0};
}

} // namespace deep_gemm::mega_moe::sched
