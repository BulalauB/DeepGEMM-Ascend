#pragma once

#include <vector>

#include <deep_gemm/layout/mega_moe.hpp>

#include "../jit/jit.hpp"

namespace deep_gemm {

// workspace给出注册维度和字节布局；buffer是本rank对称GM基址，sym_buffer_ptrs[p]是peer p的基址。
// y为最终BF16[num_tokens,H]；weights持有两层routed/shared权重及scale的设备指针。
// rank_idx为当前rank，num_tokens为本轮有效T；activation_clamp控制SwiGLU裁剪。
// cumulative_local_expert_recv_stats可空，否则指向int32[E/P]累计计数；所有内存须活到异步launch结束。
struct MegaMoEArgs {
    const mega_moe::layout::Workspace& workspace;

    void* buffer;
    const std::vector<int64_t>& sym_buffer_ptrs;
    void* y;
    int32_t* cumulative_local_expert_recv_stats;
    mega_moe::layout::Weights weights;
    uint32_t rank_idx;
    uint32_t num_tokens;
    float activation_clamp;
};

// 将注册布局维度嵌入模板特化，实际 token 数和 peer 地址作为运行时参数传入。
// 同一 stream 先清零本轮 progress 再 launch，避免旧轮次 ready 放行本轮消费者。
// 持久化 barrier epoch 不在该清零范围内，它负责重复 launch 的跨 rank 同步。
static void launch_mega_moe(const MegaMoEArgs& args) {
    const auto& workspace = args.workspace;
    const auto& config = workspace.config;
    const auto code = std::format(R"(
#include <deep_gemm/mega_moe.hpp>

static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&deep_gemm::mega_moe::mega_moe_impl<{}, {}, {}, {}, {}, {}, {}>);
}}

)",
        config.num_ranks, config.num_experts, config.num_shared_experts, config.num_max_tokens_per_rank,
        config.num_topk, config.hidden, config.intermediate_hidden);
    if (runtime->get_dry_run()) {
        jit->compile_without_load("mega_moe", code);
        return;
    }
    const auto kernel = jit->compile("mega_moe", code);

    const auto stream = get_current_npu_stream();
    const auto progress = workspace.progress.blocks.get_ptr(args.buffer);
    DJ_ACL_CHECK(aclrtMemsetAsync(progress, workspace.num_progress_bytes, 0,
                                 workspace.num_progress_bytes, stream));

    jit->launch(kernel, {.stream = stream, .num_blocks = kNumAICores},
                args.y, args.cumulative_local_expert_recv_stats, args.weights, args.num_tokens, args.activation_clamp,
                mega_moe::layout::SymBuffer<>(args.sym_buffer_ptrs, args.rank_idx));
}

} // namespace deep_gemm
