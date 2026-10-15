#pragma once

#include <vector>

#include <deep_gemm/layout/mega_moe.hpp>

#include "../jit/jit.hpp"

namespace deep_gemm {

// host 启动描述符：容量配置参与 JIT 特化，实际 T 与指针作为运行时参数。
struct MegaMoEArgs {
    // 布局以引用传入，本函数同步完成代码生成和 launch 参数封装。
    const mega_moe::layout::Workspace& workspace;

    // 整个对称注册空间的本地基址。
    void* buffer;
    // 各 rank 基址表，用来在 launch 时构造按值传入的 SymBuffer。
    const std::vector<int64_t>& sym_buffer_ptrs;
    // 最终本地 BF16 输出地址。
    void* y;
    // 可选本地专家累计统计，空指针表示不记录。
    int32_t* cumulative_local_expert_recv_stats;
    // 路由/共享的两层 transformed 权重及其 scale 地址。
    mega_moe::layout::Weights weights;
    // 本进程在通信组内的编号。
    uint32_t rank_idx;
    // 实际有效 token 数，区别于模板中的最大 token 容量。
    uint32_t num_tokens;
    // 已经按 BF16 舍入的激活截断阈值。
    float activation_clamp;
};

static void launch_mega_moe(const MegaMoEArgs& args) {
    // 容量和偏移来自同一 Workspace，避免 host 分配与 device 模板不匹配。
    const auto& workspace = args.workspace;
    // 取出编译期特化所需的 rank、专家、容量和维度配置。
    const auto& config = workspace.config;
    // 生成只负责实例化模板的源码；七个模板参数固定布局和循环常量。
    const auto code = std::format(R"(
#include <deep_gemm/mega_moe.hpp>

static void __instantiate() {{
    auto p = reinterpret_cast<void*>(&deep_gemm::mega_moe::mega_moe_impl<{}, {}, {}, {}, {}, {}, {}>);
}}

)",
        // 依模板声明顺序填入 R、E、共享专家数、Tmax，再填 top-k、H、I。
        config.num_ranks, config.num_experts, config.num_shared_experts, config.num_max_tokens_per_rank,
        config.num_topk, config.hidden, config.intermediate_hidden);
    // dry-run 仅预编译，不清零进度、不启动通信或写输出。
    if (runtime->get_dry_run()) {
        // 生成缓存中的可编译产物，但不加载到设备运行。
        jit->compile_without_load("mega_moe", code);
        // 提前退出，避免 dry-run 误触发后面的运行时内存操作。
        return;
    }
    // 按生成源码查 JIT 缓存或编译并加载对应特化内核。
    const auto kernel = jit->compile("mega_moe", code);

    // 使用调用方当前 NPU stream，使输入准备、计数清零和 kernel 保持流内顺序。
    const auto stream = get_current_npu_stream();
    // 完成计数、已发 GEMM 序号和 metadata_ready 共用连续 progress 区，每次调用整体清零。
    // memset 与 kernel 在同一 stream 顺序执行；接收 count 的清零和跨 rank 握手由 kernel 负责。
    // progress 是逻辑任务计数区，不是三个激活/输出 ring 的 payload。
    const auto progress = workspace.progress.blocks.get_ptr(args.buffer);
    // 每次调用重置完成计数与提交序号，防止读到上次调用的 ready 状态。
    DJ_ACL_CHECK(aclrtMemsetAsync(progress, workspace.num_progress_bytes, 0,
                                 workspace.num_progress_bytes, stream));

    // 以32个混合 block 启动；device 内通过 AIC/AIV 身份选择各自角色。
    jit->launch(kernel, {.stream = stream, .num_blocks = kNumAICores},
                // 运行时参数与 mega_moe_impl 签名配对，实际 T 不会改变注册空间 stride。
                args.y, args.cumulative_local_expert_recv_stats, args.weights, args.num_tokens, args.activation_clamp,
                // 把 peer 表转为 device 可用的对称地址映射，包含当前 rank 身份。
                mega_moe::layout::SymBuffer<>(args.sym_buffer_ptrs, args.rank_idx));
}

} // namespace deep_gemm
