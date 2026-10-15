#pragma once

#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <c10/core/DeviceGuard.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

#include <deep_gemm/layout/mega_moe.hpp>

#include "../backends/hccl.hpp"
#include "../jit/jit.hpp"
#include "../jit_kernels/mega_moe.hpp"

namespace deep_gemm::mega_moe {

namespace py = pybind11;

// 统一建立 host/device 共用的布局描述；下面的限制也是模板内核地址计算和任务切块的前提。
static layout::Workspace get_workspace(uint32_t num_ranks, uint32_t num_experts,
                                       uint32_t num_max_tokens_per_rank, uint32_t num_topk,
                                       uint32_t hidden, uint32_t intermediate_hidden, uint32_t num_shared_experts) {
    // rank 数参与到达计数及元数据编码，必须非零且小于256。
    DJ_HOST_ASSERT(num_ranks > 0 and num_ranks < 256);
    // 专家按连续编号均分给各 rank，因此专家总数必须能被 rank 数整除。
    DJ_HOST_ASSERT(num_experts > 0 and num_experts % num_ranks == 0);
    // Tmax 是注册容量，不是本次有效 token 数；空间按它预留。
    DJ_HOST_ASSERT(num_max_tokens_per_rank > 0);
    // top-k 至多32，匹配路由扫描、归并和位域的实现上限。
    DJ_HOST_ASSERT(num_topk > 0 and num_topk <= 32);
    // shared 分支只支持0、1或2个共享专家，其隐藏维度合并处理。
    DJ_HOST_ASSERT(num_shared_experts <= 2);
    // 输出 H 必须按物理 N256 对齐，GMM2 的 N 任务数为 H/256。
    DJ_HOST_ASSERT(hidden > 0 and hidden % layout::BLOCK_N == 0);
    // 中间维度至少512且按256对齐，满足 gate/up 分块和矩阵指令归约粒度。
    DJ_HOST_ASSERT(intermediate_hidden >= 512 and
                   intermediate_hidden % layout::BLOCK_N == 0);
    // count 低24位与 combine row 低24位都依赖容量界限；限制全rank最坏route数防止 packed 编码溢出。
    DJ_HOST_ASSERT(static_cast<uint64_t>(num_ranks) * num_max_tokens_per_rank * num_topk <
                   layout::kCountArrivalBit);
    // 只计算各区域偏移和容量；这里还没有申请或清零 device 内存。
    return layout::Workspace{{num_ranks, num_experts, num_shared_experts, num_max_tokens_per_rank,
                              num_topk, hidden, intermediate_hidden}};
}

// The root Tensor owns the HCCL lease; data views borrow its memory
struct SymmBuffer {
    // 根 tensor 持有整块注册空间；所有输入和中间结果视图都借用这块内存。
    torch::Tensor buffer;
    // 每个 rank 的对称空间基地址，device 用地址差把本地布局映射到 peer。
    std::vector<int64_t> buffer_ptrs;

    // 由进程组名和 rank 身份取得对称内存 lease，并包装成 PyTorch 张量。
    SymmBuffer(uint32_t rank_idx, uint32_t num_ranks, const std::string& hccl_group_name, int64_t num_bytes) {
        // 禁止零长度分配或越界 rank，避免创建无效通信窗口。
        DJ_HOST_ASSERT(num_bytes > 0 and num_ranks > 0 and rank_idx < num_ranks);
        // 底层 HCCL 管理注册与 peer 地址，本层保留 allocation 的共享所有权。
        auto allocation = HCCLSymmetricBuffer::acquire(hccl_group_name, rank_idx, num_ranks, num_bytes);
        int32_t device_idx;
        // 读取当前 NPU 设备编号，供 tensor 的 device 标记和析构时切换设备使用。
        DJ_ACL_CHECK(aclrtGetDevice(&device_idx));
        // PrivateUse1 是这里 PyTorch NPU 后端使用的设备类型。
        const c10::Device device(c10::DeviceType::PrivateUse1, device_idx);
        // from_blob 不复制数据；捕获 allocation 的 deleter 延长 lease 生命周期。
        buffer = torch::from_blob(allocation->buffer, {num_bytes}, [allocation, device](void*) {
            // Wait for device work before the last owner returns the lease to the pool
            // 最后一个根 tensor 引用释放时，先切回该分配所属设备。
            const c10::DeviceGuard guard(device);
            // 等待设备工作结束后才允许 lease 回池，防止 kernel 仍在访问被复用的空间。
            DJ_ACL_CHECK(aclrtSynchronizeDevice());
        // 根空间按字节暴露，具体区域在 slice_buffer 中重新解释 dtype。
        }, torch::TensorOptions().dtype(torch::kInt8).device(device));
        // 保存 peer 指针数值；kernel 启动时据此构造 SymBuffer。
        buffer_ptrs.assign(allocation->peer_ptrs.begin(), allocation->peer_ptrs.end());
    }
};

// from_blob 仅建立外部内存视图，不拥有 allocation；调用方必须保留根 SymmBuffer，不能让输入 view 独立延长 lease 生命周期。
static torch::Tensor slice_buffer(const torch::Tensor& buffer, uint64_t offset,
                                  torch::IntArrayRef shape, torch::ScalarType dtype) {
    // 切片的根必须是连续一维字节张量，offset 才能直接按字节相加。
    DJ_HOST_ASSERT(buffer.dim() == 1 and buffer.scalar_type() == torch::kInt8 and buffer.is_contiguous());
    // 用逻辑元素数乘 dtype 字节数，计算该视图真实覆盖范围。
    const auto num_bytes = c10::multiply_integers(shape) * c10::elementSize(dtype);
    // 先验证 offset，再用减法检查长度，避免 offset+长度的溢出风险。
    DJ_HOST_ASSERT(offset <= buffer.nbytes() and num_bytes <= buffer.nbytes() - offset);
    // 仅生成带 shape/dtype 的视图，不分配、不搬运，也不自动持有根 lease。
    return torch::from_blob(static_cast<uint8_t*>(buffer.data_ptr()) + offset, shape, buffer.options().dtype(dtype));
}

static auto get_symm_buffer_size_for_mega_moe(
    uint32_t num_ranks, uint32_t num_experts, uint32_t num_max_tokens_per_rank, uint32_t num_topk,
    uint32_t hidden, uint32_t intermediate_hidden, const std::string& mma_type,
    const std::string& activation, uint32_t num_shared_experts = 0) {
    // 本入口仅提供 FP8 激活乘 FP4 路由权重以及 SwiGLU 的实现。
    DJ_HOST_ASSERT(mma_type == "fp8xfp4" and activation == "swiglu");
    // 容量查询和实际 launch 使用同一 Workspace 公式，防止两边区域偏移不一致。
    const auto workspace = get_workspace(num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
                                         hidden, intermediate_hidden, num_shared_experts);
    // 这些输入视图直接借用已注册的 symmetric allocation；调用方应在 launch 前填好 FP8 data、
    // scale、top-k ID 和 weight。本 kernel 不包含 BF16 输入量化，计时须明确是否计入输入准备。
    auto slice_input_buffers = [workspace](const torch::Tensor& buffer) {
        // 要求用户传入的注册空间至少覆盖全部布局区域。
        DJ_HOST_ASSERT(buffer.nbytes() >= workspace.get_num_bytes());
        // 读取布局的已校验配置，后续视图尺寸全部由该配置导出。
        const auto& config = workspace.config;
        // 返回的是最大容量视图；实际有效行数在 launch 时从 y.shape[0] 获取。
        const int64_t num_tokens = config.num_max_tokens_per_rank;
        // H 为输入/输出宽度，I 为路由专家激活后的中间宽度。
        const int64_t hidden = config.hidden, intermediate_hidden = config.intermediate_hidden;
        // 路由中间激活只预留 ring 行数，区别于完整逻辑任务的 metadata/progress 区。
        const int64_t num_ring_tokens = workspace.num_ring_rows;
        // 共享专家使用合并后的中间宽度，其空间与 routed ring 分开。
        const int64_t shared_intermediate_hidden = config.shared_intermediate_hidden;
        // 源输入是已经量化好的 FP8，shape 为 Tmax×H；dispatch 从 peer 的此区域取行。
        auto x = slice_buffer(buffer, workspace.input.acts.data.offset, {num_tokens, hidden}, torch::kFloat8_e4m3fn);
        // 每个 int16 打包两个 E8M0 scale 字节，覆盖两个连续 K32 组。
        auto x_sf = slice_buffer(buffer, workspace.input.acts.sf.offset, {num_tokens, config.num_input_sf_pairs}, torch::kInt16);
        // 每个源 token 保存 top-k 专家编号，device 前处理按 route 扫描。
        auto topk_idx = slice_buffer(buffer, workspace.input.topk_idx.offset, {num_tokens, config.num_topk}, torch::kInt64);
        // 每个 route 保存 FP32 权重，后面传到专家端并在 SwiGLU 后量化前乘入。
        auto topk_weights = slice_buffer(buffer, workspace.input.topk_weights.offset, {num_tokens, config.num_topk}, torch::kFloat32);

        // 共享 GMM1 直接使用本地输入，不另外做 routed dispatch 的激活复制。
        auto shared_l1_acts = x;
        // 无共享专家时返回空 tensor，避免暴露不存在的共享计算输入。
        auto shared_l1_acts_sf = config.num_shared_experts > 0 ? x_sf : torch::Tensor();
        // 共享 SwiGLU 的 FP8 输出保留完整本地 token 容量，不使用 routed M-block ring。
        auto shared_l2_acts = config.num_shared_experts > 0 ? slice_buffer(buffer, workspace.shared_l2_acts.data.offset,
            {num_tokens, shared_intermediate_hidden}, torch::kFloat8_e4m3fn) : torch::Tensor();
        // Linear2 SFs are physical storage: [M-block, K-pair, M] with fixed BLOCK_M
        // scale 是物理布局的扁平视图，不能简单按 token-major 解释。
        auto shared_l2_acts_sf = config.num_shared_experts > 0 ? slice_buffer(buffer, workspace.shared_l2_acts.sf.offset,
            // shared scale 按补齐到 M256 的行数分配，保证最后块的固定 stride。
            {workspace.num_padded_shared_rows * (shared_intermediate_hidden / MX_SF_DIVISOR)}, torch::kInt16) : torch::Tensor();
        // dispatch 接收 FP8 数据的 ring，逻辑 M-block 通过取模定位物理槽。
        auto l1_acts = slice_buffer(buffer, workspace.routed.l1_acts.data.offset, {num_ring_tokens, hidden}, torch::kFloat8_e4m3fn);
        // 接收 scale 与 data 使用对应 ring 行，GMM1 按 token-major pair 读取。
        auto l1_acts_sf = slice_buffer(buffer, workspace.routed.l1_acts.sf.offset, {num_ring_tokens, config.num_input_sf_pairs}, torch::kInt16);
        // SwiGLU 输出到第二个 ring，GMM2 消费；复用条件独立于 dispatch ring。
        auto l2_acts = slice_buffer(buffer, workspace.routed.l2_acts.data.offset, {num_ring_tokens, intermediate_hidden}, torch::kFloat8_e4m3fn);
        // GMM2 scale 按块内 K-pair-major 写入，这里只暴露连续物理存储。
        auto l2_acts_sf = slice_buffer(buffer, workspace.routed.l2_acts.sf.offset,
            {num_ring_tokens * (intermediate_hidden / MX_SF_DIVISOR)}, torch::kInt16);
        // 按来源 rank 与本地专家保存到达计数，供前处理汇总本地专家 token 数。
        auto expert_recv_count = slice_buffer(buffer, workspace.metadata.expert_recv_count.offset,
            {config.num_ranks, config.num_experts / config.num_ranks}, torch::kInt32);
        // 返回顺序是 Python 调用方接口的一部分，包含输入、调试视图和专家计数。
        return std::make_tuple(x, x_sf, topk_idx, topk_weights,
                               shared_l1_acts, shared_l1_acts_sf, shared_l2_acts, shared_l2_acts_sf,
                               l1_acts, l1_acts_sf, l2_acts, l2_acts_sf, expert_recv_count);
    };
    // 同时返回总字节数和切片函数，让调用方先分配注册空间再构造各区域视图。
    return std::make_tuple(workspace.get_num_bytes(), std::function(slice_input_buffers));
}

static layout::LinearWeights check_weights(const std::tuple<torch::Tensor, torch::Tensor>& weights_tuple, uint32_t num_n_blocks,
                                           uint32_t num_k, uint32_t num_experts_per_rank = 0) {
    // 权重参数由 transformed data 和对应 scale 两个张量组成。
    const auto& [weights, weights_sf] = weights_tuple;
    // 专家数参数0表示共享权重，其张量不带最外层 expert 轴。
    const bool is_shared = num_experts_per_rank == 0;
    // 共享权重直接存 FP8；路由权重用 int8 容器打包每字节两个 FP4。
    DJ_HOST_ASSERT(weights.scalar_type() == (is_shared ? torch::kFloat8_e4m3fn : torch::kInt8) and weights.is_contiguous());
    // scale 使用 int16 容器打包两个 E8M0 字节，并要求物理连续。
    DJ_HOST_ASSERT(weights_sf.scalar_type() == torch::kInt16 and weights_sf.is_contiguous());
    // 数据和 scale 必须位于 NPU，不能把 host 地址直接传给内核。
    DJ_HOST_ASSERT(weights.device().type() == c10::DeviceType::PrivateUse1 and
                   weights_sf.device().type() == c10::DeviceType::PrivateUse1);
    // Shared weights omit the expert dimension and store unpacked FP8 data.
    // 权重按 expert/N256/K 布局；路由 FP4 的最后一轴字节数为 K/2。
    const int64_t weight_shape[]{num_experts_per_rank, num_n_blocks, layout::BLOCK_N, num_k / (is_shared ? 1 : 2)};
    // 权重 scale 每 K64 一个 int16 pair，N256 是物理内层轴。
    const int64_t sf_shape[]{num_experts_per_rank, num_n_blocks, num_k / MX_SF_DIVISOR, layout::BLOCK_N};
    // 共享模式跳过 weight_shape 的 expert 轴，验证三维 transformed 布局。
    DJ_HOST_ASSERT(weights.sizes() == torch::IntArrayRef(weight_shape + is_shared, 4 - is_shared));
    // scale 与 data 同步省略共享专家轴，避免 device 端按错误 stride 读数。
    DJ_HOST_ASSERT(weights_sf.sizes() == torch::IntArrayRef(sf_shape + is_shared, 4 - is_shared));
    // 只取 device 指针形成轻量描述符，权重张量生命周期仍由调用方负责。
    return {static_cast<const uint8_t*>(weights.data_ptr()), static_cast<const int16_t*>(weights_sf.data_ptr())};
}

static void fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l1_weights_tuple_opt,
    const std::optional<std::tuple<torch::Tensor, torch::Tensor>>& shared_l2_weights_tuple_opt,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const int& hidden, const int& intermediate_hidden, const int& num_shared_experts,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
) {
    // Config checks
    // 解包量化 recipe；这里不根据参数动态选择其他精度路径。
    const auto [rm, rn, rk] = recipe;
    // 只接受逐 M/N 元素、K32 一组的 MX scale 配置。
    DJ_HOST_ASSERT(rm == 1 and rn == 1 and rk == 32);
    // 当前只实现使用 fast_math 的 SwiGLU，其他激活组合直接拒绝。
    DJ_HOST_ASSERT(activation == "swiglu" and fast_math);
    // 检查正容量，防止有符号负数传入后转换成巨大的无符号布局尺寸。
    DJ_HOST_ASSERT(num_max_tokens_per_rank > 0 and num_experts > 0 and num_topk > 0);
    // 隐藏维度必须为正，共享专家数不能为负；更细的对齐检查在 get_workspace。
    DJ_HOST_ASSERT(hidden > 0 and intermediate_hidden > 0 and num_shared_experts >= 0);
    // 通信规模以 peer 指针表长度为准，必须与所有参与 rank 的配置一致。
    const auto num_ranks = static_cast<uint32_t>(sym_buffer_ptrs.size());
    // Use the registered layout dimensions, then validate tensors against them.
    const auto workspace = get_workspace(num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
                                         hidden, intermediate_hidden, num_shared_experts);
    // y 的行数决定本次有效 T；注册容量仍固定为 Tmax。
    DJ_HOST_ASSERT(y.dim() == 2 and y.size(0) <= num_max_tokens_per_rank and y.size(1) == hidden);
    // 开启共享专家时，两层共享权重必须同时存在。
    DJ_HOST_ASSERT(shared_l1_weights_tuple_opt.has_value() == (num_shared_experts > 0) and
                   shared_l2_weights_tuple_opt.has_value() == (num_shared_experts > 0));
    // 本次实际 token 数传入 device，用来屏蔽容量视图中尚未使用的尾行。
    const auto num_tokens = static_cast<uint32_t>(y.size(0));
    // 该实现按32个 AIC 设计任务 wave 与分工，此处拒绝不同核数的运行环境。
    DJ_HOST_ASSERT(runtime->get_num_sms() == kNumAICores);

    // Registered allocations must remain valid until completion
    // 当前 rank 必须能索引 peer 地址表。
    DJ_HOST_ASSERT(rank_idx >= 0 and rank_idx < num_ranks);
    // 注册空间根张量必须保持原始连续字节布局。
    DJ_HOST_ASSERT(sym_buffer.dim() == 1 and sym_buffer.scalar_type() == torch::kInt8 and sym_buffer.is_contiguous());
    // 再次核对 launch 布局没有超过申请容量。
    DJ_HOST_ASSERT(sym_buffer.nbytes() >= workspace.get_num_bytes());
    // 输出和通信空间必须在同一设备。
    DJ_HOST_ASSERT(sym_buffer.device() == y.device());
    // 本 rank 的 peer 表项必须指向这个根空间，否则本地与远端布局会错位。
    DJ_HOST_ASSERT(sym_buffer_ptrs[rank_idx] == reinterpret_cast<int64_t>(sym_buffer.data_ptr()));
    // 归并输出固定为连续 BF16，device 最后按此类型和行跨度写回。
    DJ_HOST_ASSERT(y.scalar_type() == torch::kBFloat16 and y.is_contiguous());
    // 保证 y.data_ptr 是 NPU 地址。
    DJ_HOST_ASSERT(y.device().type() == c10::DeviceType::PrivateUse1);

    // Check stats counter
    // 每 rank 专家数也决定统计张量长度及权重最外层维度。
    const uint32_t num_experts_per_rank = num_experts / num_ranks;
    // 统计是可选累加项，不提供时 device 不更新该计数。
    if (cumulative_local_expert_recv_stats.has_value()) {
        // device 使用 int32 计数，host 必须保证 dtype 匹配。
        DJ_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt32);
        // 统计只覆盖本 rank 持有的路由专家。
        DJ_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() == num_experts_per_rank);
        // 累加地址按连续 expert 下标计算，不支持 strided 统计视图。
        DJ_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
        // 统计写回必须与输出在同一设备。
        DJ_HOST_ASSERT(cumulative_local_expert_recv_stats->device() == y.device());
    }

    // Check transformed weights and scales
    // GMM1 的物理 N256 同时含 gate128 与 up128，因此 N 任务数为 I/128。
    const auto l1 = check_weights(l1_weights_tuple, intermediate_hidden / (layout::BLOCK_N / 2), hidden, num_experts_per_rank);
    // GMM2 每任务输出 N256，归约维度为 SwiGLU 输出宽度 I。
    const auto l2 = check_weights(l2_weights_tuple, hidden / layout::BLOCK_N, intermediate_hidden, num_experts_per_rank);
    // 默认共享描述符置空；关闭共享专家时模板分支不访问这些指针。
    layout::LinearWeights shared_l1{}, shared_l2{};
    // 仅在启用共享分支时验证并填充其两层权重。
    if (num_shared_experts > 0) {
        // 共享 GMM1 同样按 gate/up 成对的 N256 tile 检查权重。
        shared_l1 = check_weights(shared_l1_weights_tuple_opt.value(),
                                  workspace.config.shared_intermediate_hidden / (layout::BLOCK_N / 2), hidden);
        // 共享 GMM2 使用合并后的共享中间宽度进行归约。
        shared_l2 = check_weights(shared_l2_weights_tuple_opt.value(),
                                  hidden / layout::BLOCK_N, workspace.config.shared_intermediate_hidden);
    }
    // NOTES: All weights/SFs must be on the current device, like y and sym_buffer
    // launch 的 clamp 先舍入为 BF16 再传给 kernel，Vector 激活使用同一阈值；默认无穷表示不截断。
    // 未提供 clamp 时使用正无穷，保留未截断激活路径。
    const auto activation_clamp = activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    // 禁止负阈值，保证 gate 上界和 up 对称界限有效。
    DJ_HOST_ASSERT(activation_clamp >= 0);
    // 把所有 host 视图整理成 kernel 所需描述符；没有在这里额外做输入量化。
    launch_mega_moe({
        // Workspace 同时决定 JIT 模板配置及 device 内存偏移。
        .workspace = workspace,
        // 传入注册空间根指针，device 通过固定偏移定位每个区域。
        .buffer = sym_buffer.data_ptr(),
        // 传入全部 peer 基地址，构造跨 rank 的地址映射。
        .sym_buffer_ptrs = sym_buffer_ptrs,
        // 最终 BF16 输出可以位于对称空间之外，只由本地归并写入。
        .y = y.data_ptr(),
        // 可选累计统计存在才取 device 指针，否则传空。
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats.has_value() ?
            cumulative_local_expert_recv_stats->data_ptr<int32_t>() : nullptr,
        // 四个描述符依次为 routed 两层和 shared 两层。
        .weights = {l1, l2, shared_l1, shared_l2},
        // 完成范围检查后将 rank 转成 device 使用的无符号编号。
        .rank_idx = static_cast<uint32_t>(rank_idx),
        // T 控制本次有效计算量，Tmax 继续控制注册布局的容量与 stride。
        .num_tokens = num_tokens,
        // 阈值先按 BF16 舍入再转回 float，保持 device 激活阈值一致。
        .activation_clamp = static_cast<float>(c10::BFloat16(activation_clamp)),
    });
}

static void register_apis(py::module_& m) {
    // 向 Python 暴露物理 N 分块大小，供权重变换等准备步骤使用。
    m.attr("_mega_moe_block_n") = layout::BLOCK_N;
    // 导出注册空间的所有权包装，而不是让普通视图各自管理 lease。
    py::class_<SymmBuffer>(m, "SymmBuffer")
        .def(py::init<uint32_t, uint32_t, const std::string&, int64_t>(),
             py::arg("rank_idx"), py::arg("num_ranks"), py::arg("hccl_group_name"), py::arg("num_bytes"))
        // 根 tensor 和 peer 指针表只读暴露，数据内容仍可由调用方填充。
        .def_readonly("buffer", &SymmBuffer::buffer)
        .def_readonly("buffer_ptrs", &SymmBuffer::buffer_ptrs);
    // 导出容量查询与切片接口；不会触发 MegaMoE 计算。
    m.def("get_symm_buffer_size_for_mega_moe", &get_symm_buffer_size_for_mega_moe,
          py::arg("num_ranks"), py::arg("num_experts"), py::arg("num_max_tokens_per_rank"), py::arg("num_topk"),
          py::arg("hidden"), py::arg("intermediate_hidden"), py::arg("mma_type"), py::arg("activation"),
          py::arg("num_shared_experts") = 0);
    // 导出实际算子入口，参数检查完成后在当前 NPU stream 启动。
    m.def("fp8_fp4_mega_moe", &fp8_fp4_mega_moe,
          py::arg("y"), py::arg("l1_weights_tuple"), py::arg("l2_weights_tuple"),
          py::arg("shared_l1_weights_tuple_opt"), py::arg("shared_l2_weights_tuple_opt"),
          py::arg("cumulative_local_expert_recv_stats"), py::arg("sym_buffer"),
          py::arg("sym_buffer_ptrs"), py::arg("rank_idx"), py::arg("num_max_tokens_per_rank"),
          py::arg("num_experts"), py::arg("num_topk"), py::arg("hidden"),
          py::arg("intermediate_hidden"), py::arg("num_shared_experts"), py::arg("recipe"),
          py::arg("activation"), py::arg("activation_clamp_opt"), py::arg("fast_math"));
}

} // namespace deep_gemm::mega_moe
