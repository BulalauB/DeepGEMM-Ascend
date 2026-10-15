import torch
import torch.distributed as dist

from .. import _C

# 物理 N 任务宽度从 C++ 导出，Python 权重分块与 device 保持一致。
BLOCK_N = _C._mega_moe_block_n


class SymmBuffer:
    """HCCL-registered symmetric buffer for the fused MegaMoE pipeline.

    Shape and capacity attributes describe the C++ layout and must not be modified.
    Registered allocations stay in the pool until process exit; destroy() does not free HBM.
    Allocation order and buffer/handle owner lifetimes must match across ranks.
    Use base to share one allocation across sequential configurations.
    Data views borrow memory: keep a SymmBuffer owner alive until all accesses complete.
    Views become invalid when the last owner releases the allocation, even if retained.
    """

    def __init__(self, group: dist.ProcessGroup,
                 num_experts: int,
                 num_max_tokens_per_rank: int, num_topk: int,
                 hidden: int, intermediate_hidden: int,
                 num_shared_experts: int = 0,
                 mma_type: str = 'fp8xfp4',
                 activation: str = 'swiglu',
                 base: 'SymmBuffer | None' = None):
        # 当前绑定只支持 FP8 激活、FP4 路由权重这一计算组合。
        assert mma_type == 'fp8xfp4', 'Ascend MegaMoE currently supports only fp8xfp4'
        # 后处理固定为 SwiGLU，不支持在同一内核中动态换激活。
        assert activation == 'swiglu', 'Ascend MegaMoE currently supports only swiglu'
        # 保存通信组；同一注册空间的各 owner 必须属于同一个组。
        self.group = group
        # 全局路由专家总数，C++ 会验证能被 rank 数整除。
        self.num_experts = num_experts
        # 注册 token 上限 Tmax，决定空间布局，实际执行 T 从输出张量获取。
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        # 每 token 的路由槽数，决定源元数据和完整 combine 返回区容量。
        self.num_topk = num_topk
        # 输入和最终输出宽度 H。
        self.hidden = hidden
        # 单路由专家 SwiGLU 后宽度 I。
        self.intermediate_hidden = intermediate_hidden
        # 共享专家合并中间宽度为此数量乘 I。
        self.num_shared_experts = num_shared_experts
        # 保存计算类型供调用者检查配置，不在这里执行量化。
        self.mma_type = mma_type
        # 保存激活类型，实际启动仍会校验传入的 activation。
        self.activation = activation

        # 先用统一的 C++ Workspace 公式查询容量，同时取得对应布局的切片闭包。
        num_bytes, slice_input_buffers = _C.get_symm_buffer_size_for_mega_moe(
            group.size(), num_experts, num_max_tokens_per_rank, num_topk,
            hidden, intermediate_hidden, mma_type, activation, num_shared_experts)
        # 没有复用根空间时，建立新的 HCCL 对称注册空间 owner。
        if base is None:
            # 获取组内 rank，不能用全局进程号替代子通信组编号。
            rank_idx = group.rank()
            # 通过当前组的 HCCL comm 名取得注册空间和 peer 地址表。
            self.handle = _C.SymmBuffer(rank_idx, group.size(),
                                        group._get_backend(torch.device('npu')).get_hccl_comm_name(rank_idx), num_bytes)
            # 保留根 tensor，它的 deleter 持有底层 allocation lease。
            self.buffer = self.handle.buffer
            # 等待本地分配/初始化设备工作完成。
            torch.npu.synchronize()
            # 所有组内 rank 完成构造后才返回，便于后续按统一布局通信。
            dist.barrier(group)
        else:
            # 复用前验证 base 仍有效且是同一个通信组。
            assert base.buffer is not None and base.handle is not None and base.group is group, 'Cannot reuse an invalid symmetric buffer'
            # 允许较小配置借用较大分配，但不能越过底层容量。
            assert num_bytes <= base.buffer.nbytes, f'Requires {num_bytes} bytes, but the base buffer has {base.buffer.nbytes}'
            # Shared scratch must be used in order, with inputs refilled for each layout
            # 两个配置共享同一根空间，因此只能顺序使用并按当前布局重新填输入。
            self.buffer = base.buffer
            # 共享 handle 所有权，确保释放一个包装对象不会提前归还其他 owner 的 lease。
            self.handle = base.handle

        # 在同一根空间上构造有 dtype/shape 的视图，顺序与 C++ 返回 tuple 一致。
        (self.x, self.x_sf, self.topk_idx, self.topk_weights,
         self.shared_l1_acts, self.shared_l1_acts_sf, self.shared_l2_acts, self.shared_l2_acts_sf,
         self.l1_acts, self.l1_acts_sf, self.l2_acts, self.l2_acts_sf,
         self._expert_recv_count) = slice_input_buffers(self.buffer)

    def destroy(self) -> None:
        """Drop this owner's references; views require another live owner of the allocation."""
        # 放弃本对象的 handle 引用；其他 owner 存在时 allocation 仍有效。
        self.handle = None
        # 放弃根张量引用；这里只是释放引用，不保证将 HBM 返还系统。
        self.buffer = None
        # 清理通信组引用，已销毁对象不能继续启动算子。
        self.group = None
        # 清除源 FP8 输入和 scale 视图。
        self.x = self.x_sf = None
        # 清除源专家编号与路由权重视图。
        self.topk_idx = self.topk_weights = None
        # 共享 GMM1 输入是源输入别名，也同步清理。
        self.shared_l1_acts = self.shared_l1_acts_sf = None
        # 清除共享 SwiGLU 中间结果视图。
        self.shared_l2_acts = self.shared_l2_acts_sf = None
        # 清除 dispatch 接收 ring 的数据和 scale 视图。
        self.l1_acts = self.l1_acts_sf = None
        # 清除 routed SwiGLU 输出 ring 的数据和 scale 视图。
        self.l2_acts = self.l2_acts_sf = None
        # 清除路由前处理计数区视图。
        self._expert_recv_count = None

    @property
    def expert_recv_count(self) -> torch.Tensor:
        # 这是来源 rank×本地专家的接收计数视图，不是累计统计张量。
        return self._expert_recv_count

    @property
    def num_bytes(self) -> int:
        # 返回底层分配实际字节数；base 复用时可能大于当前配置的最低需求。
        return self.buffer.nbytes


def _transform_fp8_nz(t: torch.Tensor) -> torch.Tensor:
    # Shared FP8 weights use the NZ order consumed directly by AIC.
    # 保留外层 expert/N-block 等维度，最后两维解释为 N×K。
    shape = t.shape
    # 用字节视图规避 FP8 的通用张量操作限制，并把 K 拆成 K/32 与32。
    blocks = t.view(torch.uint8).reshape(*shape[:-2], shape[-2], shape[-1] // 32, 32)
    # 交换 N 与 K/32 得到 NZ 顺序；contiguous 真正重排字节，再恢复外观 shape/dtype。
    return blocks.transpose(-3, -2).contiguous().view(shape).view(t.dtype)


def _interleave_weights(t: torch.Tensor) -> torch.Tensor:
    # Gate/up pairs occupy one BLOCK_N block in both weight data and scale factors.
    # 允许共享二维权重或带专家轴的三维权重。
    assert t.dim() in (2, 3)
    # 保存原始 dtype，避免 FP8 转字节后丢失恢复所需类型。
    dtype = t.dtype
    # FP8 需要在布局重排时借用 uint8 类型；int8 packed FP4 可直接操作。
    if dtype == torch.float8_e4m3fn:
        # 只改变 dtype 视图，FP8 与 uint8 都是1字节，不改变数据内容。
        t = t.view(torch.uint8)
    # n 是 gate 与 up 拼接的总行数，k 是物理存储列数，FP4 时为逻辑 K/2。
    *batch, n, k = t.shape
    # 前一半是 gate，后一半是 up。
    half = n // 2
    # 每个物理 N256 任务各取 gate128 和 up128。
    gran = BLOCK_N // 2
    # 两半必须等长，并能按128行完整切块。
    assert n % 2 == 0 and half % gran == 0
    # GMM1 的 N 任务数为中间宽度 I/128。
    num_n_blocks = half // gran
    # 把全部 gate 行分成连续128行子块。
    gate = t[..., :half, :].reshape(*batch, num_n_blocks, gran, k)
    # 用相同分块方式拆 up，保持同一输出通道 gate/up 一一配对。
    up = t[..., half:, :].reshape(*batch, num_n_blocks, gran, k)
    # 在每个 N-block 内依次放 gate128、up128，满足 device SwiGLU 的行内布局。
    result = torch.stack((gate, up), dim=-3).reshape(*batch, num_n_blocks, BLOCK_N, k)
    # 共享 FP8 权重在 host 侧进一步变 NZ；routed FP4 保持打包布局供 AIV0 展开。
    return _transform_fp8_nz(result.view(dtype)) if dtype == torch.float8_e4m3fn else result


def _reshape_weights(t: torch.Tensor) -> torch.Tensor:
    # GMM2 同样允许共享二维或带 expert 轴的三维输入。
    assert t.dim() in (2, 3)
    # 解包外层 batch/expert 轴与物理 N/K。
    *batch, n, k = t.shape
    # GMM2 每个输出任务必须包含完整 N256。
    assert n % BLOCK_N == 0
    # 按输出列 N256 分块，GMM2 不再需要 gate/up 交织。
    result = t.reshape(*batch, n // BLOCK_N, BLOCK_N, k)
    # 共享 FP8 直接转换 NZ；路由 FP4 留给 kernel 的向量展开路径。
    return _transform_fp8_nz(result) if t.dtype == torch.float8_e4m3fn else result


def _scale_fp4_weight_sf(sf: torch.Tensor) -> torch.Tensor:
    # Internal dequant emits FP4 values divided by 64, so compensate once in the weight layout.
    # AIV0 的 FP4 位展开产生原数值的 2^-6，因此预处理 scale 指数加6补偿。
    exponent_bias = 6
    # UE8M0 的254是最大有限指数，255保留为特殊编码。
    max_finite_exponent = 254
    # 逐字节操作两个打包在 int16 中的 scale exponent。
    exponent = sf.view(torch.uint8)
    # 有限指数加6后必须仍不超过254；特殊值255允许原样保留。
    valid = (exponent <= max_finite_exponent - exponent_bias) | (exponent == 255)
    # 同步检查是否存在不可补偿的 scale，避免 uint8 加法回绕。
    if not valid.all().item():
        # 拒绝249..254这些补偿后会溢出的原有限指数。
        raise ValueError('FP4 weight scale exponent must be 0..248 or 255')
    # Exponent 0 represents 2^-127; only 255 is a special value
    # 保留255，其他指数加6，抵消 device 展开导致的64倍缩小。
    exponent = torch.where(exponent == 255, exponent, exponent + exponent_bias)
    # 按连续字节存回 int16 pair，供 C++/device 直接使用。
    return exponent.contiguous().view(torch.int16)


def transform_weights_for_mega_moe(
    l1_weights: tuple[torch.Tensor, torch.Tensor],
    l2_weights: tuple[torch.Tensor, torch.Tensor],
    activation: str = 'swiglu'
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Transform routed FP4 or shared FP8 (weight, scale) pairs."""
    # 权重布局与 gate/up 顺序依赖 SwiGLU，因此拒绝其他激活。
    assert activation == 'swiglu', f'Only `swiglu` activation is supported, got `{activation}`'
    # 两层都必须传入 (data, scale)，不接受未量化的单张量权重。
    assert isinstance(l1_weights, tuple) and isinstance(l2_weights, tuple), 'Ascend MegaMoE requires FP4/FP8 weight-scale pairs'
    # 每层严格包含两个张量。
    assert len(l1_weights) == len(l2_weights) == 2
    # 解包 GMM1 权重与 scale。
    l1_w, l1_sf = l1_weights
    # 解包 GMM2 权重与 scale。
    l2_w, l2_sf = l2_weights
    # 第一层数据与 scale 必须具有一致的 expert/batch 轴和 N 行数。
    assert l1_w.shape[:-2] == l1_sf.shape[:-2] and l1_w.size(-2) == l1_sf.size(-2)
    # 第二层执行同样的轴匹配检查。
    assert l2_w.shape[:-2] == l2_sf.shape[:-2] and l2_w.size(-2) == l2_sf.size(-2)
    # 两层必须使用相同权重精度。
    assert l1_w.dtype == l2_w.dtype
    # int8 表示 packed FP4 容器，FP8 表示共享权重路径。
    assert l1_w.dtype in (torch.int8, torch.float8_e4m3fn)
    # 两层 scale 均用 int16 打包两个 E8M0 字节。
    assert l1_sf.dtype == l2_sf.dtype == torch.int16
    # 检查数据的逻辑 K 与 scale pair 数一致：FP4 每字节2元素，每 pair 覆盖64元素。
    assert l1_w.size(-1) * (2 if l1_w.dtype == torch.int8 else 1) == l1_sf.size(-1) * 64
    # GMM2 使用同一 K/scale 容量对应关系。
    assert l2_w.size(-1) * (2 if l2_w.dtype == torch.int8 else 1) == l2_sf.size(-1) * 64
    # 第一层 data 将 gate/up 改为每任务128+128行。
    l1_w = _interleave_weights(l1_w)
    # scale 先与 data 做相同 N 交织，再转为 [K-pair,N256] 的物理顺序。
    l1_sf = _interleave_weights(l1_sf).transpose(-1, -2).contiguous()
    # 第二层 data 只需 N256 分块。
    l2_w = _reshape_weights(l2_w)
    # 第二层 scale 也转为块内 K-pair-major，匹配 AIC 加载。
    l2_sf = _reshape_weights(l2_sf).transpose(-1, -2).contiguous()
    # 只有 routed FP4 需要指数补偿，shared FP8 不经过缩小64倍的位展开。
    if l1_w.dtype == torch.int8:
        # 补偿第一层 FP4 对应的 scale。
        l1_sf = _scale_fp4_weight_sf(l1_sf)
        # 补偿第二层 FP4 对应的 scale。
        l2_sf = _scale_fp4_weight_sf(l2_sf)
    # 输出第一层已转换 data/scale 元组。
    l1_transformed = (l1_w, l1_sf)
    # 输出第二层已转换 data/scale 元组。
    l2_transformed = (l2_w, l2_sf)
    # 此预处理应在权重准备阶段执行，不属于 fused kernel 的运行时流水。
    return l1_transformed, l2_transformed


def fp8_fp4_mega_moe(y: torch.Tensor,
                     l1_weights: tuple[torch.Tensor, torch.Tensor],
                     l2_weights: tuple[torch.Tensor, torch.Tensor],
                     sym_buffer: SymmBuffer,
                     shared_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
                     shared_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
                     cumulative_local_expert_recv_stats: torch.Tensor | None = None,
                     recipe: tuple[int, int, int] = (1, 1, 32),
                     activation: str = 'swiglu',
                     activation_clamp: float | None = None,
                     fast_math: bool = True) -> None:
    # 启动前检查 owner 仍有效；输入视图本身不能代替注册空间所有权。
    assert sym_buffer.buffer is not None and sym_buffer.handle is not None
    # 进入 C++ 参数校验与 JIT launch，Python 层没有另做 dispatch/combine kernel。
    _C.fp8_fp4_mega_moe(
        # y.shape[0] 决定本次有效 token 数，输出 dtype 要求为 BF16。
        y,
        # 传入已经变换过的 routed 两层权重。
        l1_weights, l2_weights,
        # 可选共享两层权重需成对提供。
        shared_l1_weights, shared_l2_weights,
        # 可选专家累计计数在 device 中更新。
        cumulative_local_expert_recv_stats,
        # 注册空间包含源输入、metadata、ring 和完整 combine 返回区。
        sym_buffer.buffer,
        # 传入 peer 基址与通信组内 rank，不能替换为其他进程组的编号。
        sym_buffer.handle.buffer_ptrs, sym_buffer.group.rank(),
        # Tmax 固定空间 stride，与本次有效 token 数分开。
        sym_buffer.num_max_tokens_per_rank,
        # 专家总数和 top-k 控制路由与元数据大小。
        sym_buffer.num_experts, sym_buffer.num_topk,
        # H、I、共享数量参与 JIT 特化，必须与申请空间的配置匹配。
        sym_buffer.hidden, sym_buffer.intermediate_hidden, sym_buffer.num_shared_experts,
        # 当前只接受 (1,1,32)，表示 K32 的 MX 量化分组。
        recipe,
        # 激活固定 SwiGLU，clamp 未指定时 C++ 使用无穷上界。
        activation, activation_clamp,
        # 当前实现要求启用 fast_math。
        fast_math
    )
