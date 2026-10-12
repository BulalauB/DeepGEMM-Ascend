#pragma once

#include <cstdint>
#include <type_traits>

#include <deep_gemm/common.hpp>
#include <deep_gemm/layout/sym_buffer.hpp>

namespace deep_gemm::mega_moe::layout {

// M256 是通信/存储/依赖的逻辑块上限；实际 MAD 可以选择 M128 或 M256。
// GMM1 的物理 N256 为 gate128 + up128，SwiGLU 后逻辑输出宽度为 128；GMM2 输出 N256。
// Logical blocks keep the GM stride and task count independent of the MAD shape
constexpr uint32_t BLOCK_M = 256, BLOCK_N = 256;

using deep_gemm::layout::SymBuffer;

// GM buffer alignment/count stride, not the 128-byte L2 cache sector
constexpr uint64_t kNumBufferAlignmentBytes = 512;

// SIMT threads per metadata worker and routes per batch
constexpr uint32_t kNumDispatchThreads = 512;

// Packed count: arrivals [31:24], value [23:0]
constexpr uint32_t kCountArrivalBit = 1u << 24;
constexpr uint32_t kCountValueMask = kCountArrivalBit - 1;

// 每个逻辑块的状态链及计数含义：
// Dispatched：所有 AIV1 搬运核完成（含空 owner）；Linear1Ready：全部 GMM1 N-block 完成量化；
// Linear2Ready：全部 GMM2 N-block 完成写回；CombineSent：所有 AIV1 完成该块回传。
// flags 使用不回绕的逻辑块索引，payload 使用 ring 索引，从而区分同槽的不同代数据。
// Routed progress counts workers or GEMM N blocks
// routed 完成计数的写者、等待目标和释放对象（C=32，I=intermediate_hidden，H=hidden）：
//   kDispatched   ：每个 AIV1 每块 +1；AIC 等 C，读取 l1_acts。
//   kLinear1Ready ：每个完整 GMM1/量化任务 +1；AIC 等 I/128，读取 l2_acts；pull 同时可回收旧 l1_acts。
//   kLinear2Ready ：每个 GMM2 FIX 完成任务 +1；push 等 H/256，读取 output；epilogue 可回收旧 l2_acts。
//   kCombineSent  ：每个 AIV1 排空一批回传后，对批中每块 +1；AIC 等 C，回收旧 output。
// payload 槽 b % ring_blocks 的下一代 b，分别等待以上三种消费者完成中的 b-ring_blocks；
// progress 的逻辑 b 不取模，本次调用开始统一清零。shared 的特殊计数在 epilogue 中单独处理。
enum ProgressStage : uint32_t {
    kDispatched,    // Linear1 inputs ready
    kLinear1Ready,  // Quantized Linear2 inputs ready
    kLinear2Ready,  // BF16 expert outputs ready
    kCombineSent,   // Sends complete; output ring reusable
    kNumProgressStages
};

// Fixed shape and registered token capacity
struct Config {
    uint32_t num_ranks;
    uint32_t num_experts;
    uint32_t num_shared_experts;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_topk;
    uint32_t hidden;
    uint32_t intermediate_hidden;

    const uint32_t shared_intermediate_hidden = num_shared_experts * intermediate_hidden;
    const uint32_t num_input_sf_pairs = hidden / MX_SF_DIVISOR;
};

// Typed GM range with a fixed byte stride
template <typename dtype_t, uint64_t kStrideBytes = sizeof(dtype_t)>
struct Buffer {
    static_assert(kStrideBytes >= sizeof(dtype_t) and kStrideBytes % alignof(dtype_t) == 0);
    uint64_t offset;

    constexpr __aicore__ Buffer(uint64_t& next_offset, uint64_t num_elements, uint64_t alignment = kStrideBytes):
        // 按 alignment 对齐起点，再为固定 stride 的 num_elements 个条目推进偏移；host/device 共用公式。
        offset(aligned(next_offset, alignment)) { next_offset = offset + num_elements * kStrideBytes; }

    // entry_idx uses kStrideBytes even when view_t changes
    // Buffer<uint32_t>::get_ptr<uint64_t>(base, i) advances by i * 4 bytes
    template <typename view_t = dtype_t, typename base_t>
    constexpr __aicore__ __attribute__((always_inline)) __gm__ view_t* get_ptr(base_t base, uint64_t entry_idx = 0) const {
        static_assert(std::is_pointer_v<base_t> or std::is_same_v<base_t, uint64_t>,
                      "Buffer base must be a pointer or a uint64_t address");

        // Preserve the base type through CANN lowering
        // 地址 = allocation base + 区域 byte offset + entry_idx * 固定 byte stride；view_t 不影响步长。
        const auto address = __builtin_bit_cast(uintptr_t, base) + offset + entry_idx * kStrideBytes;
        return __builtin_bit_cast(__gm__ view_t*, address);
    }
};

struct LinearWeights {
    __gm__ const uint8_t* data;
    __gm__ const int16_t* sf;
};

struct Weights {
    LinearWeights l1;
    LinearWeights l2;
    LinearWeights shared_l1;
    LinearWeights shared_l2;
};
static_assert(sizeof(Weights) == 8 * sizeof(uint64_t));

// GM workspace layout; all offsets are relative to the allocation base
struct Workspace {
private:
    // Host and device use the same storage layout
#ifdef __CCE__
    using fp8_t = float8_e4m3_t;
    using bf16_t = bfloat16_t;
#else
    using fp8_t = uint8_t;
    using bf16_t = uint16_t;
#endif

    // Keep capacity calculations independent of include order
    template <typename lhs_t, typename rhs_t>
    static constexpr __aicore__ auto ceil_div(lhs_t value, rhs_t divisor) -> decltype(value + divisor) {
        return (value + divisor - 1) / divisor;
    }

public:
    // Rank-barrier ABI; alternate signal slots by epoch
    struct RankBarrier {
        uint64_t epoch;
        int32_t signals[2];
    };

    // data 每元素 1B；sf 每元素 int16，含两个 E8M0 scale，覆盖 64 个通道（每 32 通道一个 scale）。
    struct FP8Buffer {
        Buffer<fp8_t> data;
        Buffer<int16_t> sf;

        constexpr __aicore__ FP8Buffer(uint64_t& offset, uint64_t num_rows, uint32_t num_cols):
            data{offset, num_rows * num_cols, kNumBufferAlignmentBytes},
            sf{offset, num_rows * ceil_div(num_cols, MX_SF_DIVISOR), kNumBufferAlignmentBytes} {}
    };

    Config config;

    // 这里按最坏偏斜预留：每个源 rank 的全部 Tmax*topk route 都可能命中同一专家。
    // 元数据空间按专家乘该容量；实际发送仅使用有效 count 前缀，不能将预留量当通信量。
    // Reserve one packed index/weight entry per source route.
    const uint32_t num_src_slots_per_expert = config.num_max_tokens_per_rank * config.num_topk;
    const uint64_t num_src_indices = static_cast<uint64_t>(config.num_experts) * num_src_slots_per_expert;
    const uint32_t num_padded_shared_rows = aligned(config.num_max_tokens_per_rank, BLOCK_M);

    const uint32_t num_max_shared_m_blocks = config.num_shared_experts > 0 ? ceil_div(config.num_max_tokens_per_rank, BLOCK_M) : 0;
    // 每阶段保留 shared 最大块数与 routed 最大块数；运行时 routed slot 从实际 shared 块数之后开始。
    const uint32_t num_progress_counters_per_stage = num_max_shared_m_blocks + get_num_max_m_blocks();
    // metadata 行数 Bmax*256；l1/l2/output 的 payload 只占 Bring*256，因此 ring 降低激活空间但不压缩 metadata。
    const uint64_t num_full_pool_rows = get_num_max_m_blocks() * BLOCK_M;

    // Size from the registered limit, not active T
    const uint64_t num_ring_rows = static_cast<uint64_t>(get_num_ring_blocks()) * BLOCK_M;

    // Field order fixes the symmetric GM placement
    uint64_t next_offset = 0;
    struct {
        Buffer<uint32_t> grid_sync_count;
        Buffer<RankBarrier> rank_barriers;
    } sync{
        .grid_sync_count = {next_offset, 3, kNumBufferAlignmentBytes},
        .rank_barriers = {next_offset, 2, kNumBufferAlignmentBytes},
    };

    struct {
        Buffer<uint32_t> worker_expert_count;
        Buffer<uint32_t> total_expert_count;
        // Completed producer M-blocks; all ones means this rank has finished its stores
        // 源端按 producer rank 存完成前缀；每条 prefix 独占 sector，避免与其他 rank 的热写互扰。
        Buffer<uint64_t, kL2SectorBytes> combine_ready_prefix;
        // expert_recv_count 为 [src rank][local expert]，expert_recv_count_sum 为每个本地专家的 packed arrival/sum。
        Buffer<uint32_t> expert_recv_count;
        Buffer<uint32_t, kNumBufferAlignmentBytes> expert_recv_count_sum;

        // Packed AIV0 arrivals/M-block count is independent of expert counters
        Buffer<uint32_t> num_m_blocks_certificate;
        // src_token_metadata 是接收端 route 索引/权重；staged_token_topk_idx 是源端压紧索引 scratch。
        // 二者都按专家的最坏容量预留，但本次仅 count 前缀有效。
        Buffer<uint64_t> src_token_metadata;
        Buffer<uint32_t> staged_token_topk_idx;
    } metadata{
        .worker_expert_count = {next_offset, static_cast<uint64_t>(kNumAICores) * config.num_experts, kNumBufferAlignmentBytes},
        .total_expert_count = {next_offset, config.num_experts},
        .combine_ready_prefix = {next_offset, config.num_ranks},
        .expert_recv_count = {next_offset, config.num_experts},
        .expert_recv_count_sum = {next_offset, config.num_experts / config.num_ranks},
        .num_m_blocks_certificate = {next_offset, 1, kL2SectorBytes},
        .src_token_metadata = {next_offset, num_src_indices, kNumBufferAlignmentBytes},
        .staged_token_topk_idx = {next_offset, num_src_indices, kNumBufferAlignmentBytes},
    };

    struct {
        FP8Buffer acts;
        Buffer<int64_t> topk_idx;
        Buffer<float> topk_weights;
    } input{
        .acts = {next_offset, config.num_max_tokens_per_rank, config.hidden},
        .topk_idx = {next_offset, static_cast<uint64_t>(config.num_max_tokens_per_rank) * config.num_topk, kNumBufferAlignmentBytes},
        .topk_weights = {next_offset, static_cast<uint64_t>(config.num_max_tokens_per_rank) * config.num_topk, kNumBufferAlignmentBytes},
    };

    // Clear these progress arrays together on each launch
    struct {
        // 每阶段每逻辑块独立 sector；不同生产核对同一块 atomic_add，不同块不会复用同一计数。
        Buffer<uint32_t, kL2SectorBytes> blocks;
        // 每 AIC 的已发 GEMM 序号供对应 AIV1 检查；这是 uint8 环绕序号，最终完成仍要等 FIX 事件。
        Buffer<uint8_t, kL2SectorBytes> num_issued_gemms;
        // 每 AIV0 通知同 block 的 AIV1：certificate 已完整；AIC 通过另一路跨核事件接收通知。
        Buffer<uint8_t, kL2SectorBytes> metadata_ready;
    } progress{
        .blocks = {next_offset, static_cast<uint64_t>(kNumProgressStages) * num_progress_counters_per_stage},
        .num_issued_gemms = {next_offset, kNumAICores},
        .metadata_ready = {next_offset, kNumAICores},
    };
    const uint64_t num_progress_bytes = next_offset - progress.blocks.offset;

    // 三个独立 payload ring，不互相别名：l1_acts / l2_acts / output。
    // 分别由 Linear1Ready / Linear2Ready / CombineSent 释放旧槽；元数据仍按全量逻辑行保存。
    struct {
        FP8Buffer l1_acts;

        // Metadata uses logical rows and never wraps
        Buffer<float> topk_weights;
        Buffer<uint32_t> src_metadata;

        // Input SF is token-major; quantized Linear2 SF is K-major within each M-block
        FP8Buffer l2_acts;
        Buffer<bf16_t> output;
    } routed{
        .l1_acts = {next_offset, num_ring_rows, config.hidden},
        .topk_weights = {next_offset, num_full_pool_rows, kNumBufferAlignmentBytes},
        .src_metadata = {next_offset, num_full_pool_rows, kNumBufferAlignmentBytes},
        .l2_acts = {next_offset, num_ring_rows, config.intermediate_hidden},
        .output = {next_offset, num_ring_rows * config.hidden, kNumBufferAlignmentBytes},
    };

    // 返回区为全量 [topk-slot][Tmax][hidden]，不随中间激活 ring 回绕。
    // 共享专家（如有）合并为额外一个槽；源端读取有效 route 并在 FP32 中求和。
    // One BF16 slot per route, plus one combined shared slot
    // 这里仍按每 route 全量返回保存：约2B*(topk+shared_slot)*Tmax*H，
    // 所以不能将“中间激活 ring”误称为整个工作区容量与 T 无关。
    Buffer<bf16_t> combine_input{
        next_offset, static_cast<uint64_t>(config.num_topk + (config.num_shared_experts > 0)) * config.num_max_tokens_per_rank * config.hidden,
        kNumBufferAlignmentBytes};

    // Shared Linear2 inputs; Linear1 reads original tokens
    // 所有 shared expert 在 intermediate 维拼成 num_shared_experts*I，Linear2 沿该 K 维求和到同一 shared 返回槽。
    FP8Buffer shared_l2_acts{next_offset, config.num_shared_experts > 0 ? num_padded_shared_rows : 0, config.shared_intermediate_hidden};

    // Producer M-block for each input route; dispatch fills this before publishing readiness
    // 源 rank 上 [Tmax][topk] 的生产块映射；由远端专家 dispatch 写入，归并侧不得根据随机/置换后的 row 顺序猜测。
    Buffer<uint32_t> route_m_blocks{
        next_offset, static_cast<uint64_t>(config.num_max_tokens_per_rank) * config.num_topk, kNumBufferAlignmentBytes};

    // 容量上界：所有 rank 的 route 总量按 M256 向上取整，再为每个本地专家预留额外尾块余量。
    constexpr __aicore__ uint64_t get_num_max_m_blocks() const {
        return ceil_div(static_cast<uint64_t>(config.num_ranks) * config.num_max_tokens_per_rank * config.num_topk, BLOCK_M) + config.num_experts / config.num_ranks;
    }

    // ring 容量按注册上界选择，不随本次专家实际 M 缩小；前瞻深度同时受此容量反压。
    constexpr __aicore__ uint32_t get_num_ring_blocks() const {
        // Measured overlap window; the deadlock-safe minimum can still lose throughput
        // H/256 是每块 GMM2 N 任务数；任务较少时用 4*32 块扩大在途窗口，否则用 32 块。
        // 取 min 后若覆盖全 pool 则不回绕；否则 ring 大小为 32 或 128，均为二次幂。
        const uint32_t num_windows = config.hidden / BLOCK_N <= kNumAICores / 2 ? 4 : 1;
        const uint32_t num_window_blocks = num_windows * kNumAICores;
        return get_num_max_m_blocks() < num_window_blocks ? static_cast<uint32_t>(get_num_max_m_blocks()) : num_window_blocks;
    }

    // Coordinates within the registered GM layout
    constexpr __aicore__ __attribute__((always_inline)) uint32_t get_progress_idx(ProgressStage stage, uint32_t progress_slot_idx) const {
        // 按 [stage][logical progress slot] 展平；绝不取 ring 下标，否则旧代 ready 会误放行新代数据。
        return stage * num_progress_counters_per_stage + progress_slot_idx;
    }

    constexpr __aicore__ __attribute__((always_inline)) uint64_t get_src_metadata_idx(
        uint32_t local_expert_idx, uint32_t src_rank_idx, uint32_t src_slot_idx) const {
        // 按 [local expert][source rank][source slot] 展平，远端发送与本地 dispatch 解码使用同一公式。
        return (static_cast<uint64_t>(local_expert_idx) * config.num_ranks + src_rank_idx) * num_src_slots_per_expert + src_slot_idx;
    }

    constexpr __aicore__ __attribute__((always_inline)) uint64_t get_expert_recv_idx(uint32_t src_rank_idx, uint32_t local_expert_idx) const {
        return static_cast<uint64_t>(src_rank_idx) * (config.num_experts / config.num_ranks) + local_expert_idx;
    }

    constexpr __aicore__ __attribute__((always_inline)) uint64_t get_combine_m_idx(uint64_t slot_idx, uint64_t token_idx) const {
        // combine 行号 = top-k 槽 * 注册 Tmax + 源 token；使用注册容量作 stride，实际 T 变小也不改变地址。
        return slot_idx * config.num_max_tokens_per_rank + token_idx;
    }

    constexpr __aicore__ uint64_t get_num_bytes() const {
        // HCCL registration uses 2 MiB alignment
        return aligned(next_offset, 2 * 1024 * 1024);
    }
};

template <uint32_t kNumRanks, uint32_t kNumExperts, uint32_t kNumSharedExperts,
          uint32_t kNumMaxTokensPerRank, uint32_t kNumTopk, uint32_t kHidden, uint32_t kIntermediateHidden>
inline constexpr Workspace kWorkspace{{
    kNumRanks, kNumExperts, kNumSharedExperts, kNumMaxTokensPerRank, kNumTopk, kHidden, kIntermediateHidden}};

template <const auto& workspace>
constexpr __aicore__ uint32_t get_ring_block_idx(uint32_t pool_block_idx) {
    constexpr uint32_t kNumRingBlocks = workspace.get_num_ring_blocks();
    // 全 pool 模式直接返回逻辑块号；有限 ring 为二次幂，用位与实现取模。
    // 这里只计算地址，安全覆盖由各阶段对 logical_block - ring_blocks 的完成检查保证。
    return kNumRingBlocks == workspace.get_num_max_m_blocks() ? pool_block_idx : pool_block_idx & (kNumRingBlocks - 1);
}

template <const auto& workspace>
constexpr __aicore__ uint64_t get_ring_m_idx(uint32_t pool_block_idx, uint32_t m_idx_in_block = 0) {
    return static_cast<uint64_t>(get_ring_block_idx<workspace>(pool_block_idx)) * BLOCK_M + m_idx_in_block;
}
} // namespace deep_gemm::mega_moe::layout
