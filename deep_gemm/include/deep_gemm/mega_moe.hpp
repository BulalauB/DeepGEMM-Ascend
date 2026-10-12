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
constexpr uint32_t kNumL0Stages = 2;
constexpr uint32_t kNumL1AStages = 3, kNumL1BStages = 4, kNumSFStages = 2;

// Split each logical M-block between two FIX/epilogue buffers
constexpr uint32_t kNumEpilogueStages = 2;
// M256 的 FIX/epilogue 输出按两份 M128 UB 切开；只缩小写回/消费粒度，不改变逻辑 M-block。
constexpr uint32_t STORE_BLOCK_M = BLOCK_M / kNumEpilogueStages;

// FP8 compute shapes
// BLOCK_M  BLOCK_N  BLOCK_K  MAD_M  MAD_N  MAD_K
//     256      256      512    128    128    256
//     256      256      256    256    256    128
template <uint32_t kMadM>
struct MMAConfig {
    static constexpr uint32_t BLOCK_M = layout::BLOCK_M;
    static constexpr uint32_t BLOCK_N = layout::BLOCK_N;
    static constexpr uint32_t BLOCK_K = L0ASizeBytes / kMadM;
    static constexpr uint32_t MAD_M = kMadM;
    static constexpr uint32_t MAD_K = BLOCK_K / kNumL0Stages;
    static constexpr uint32_t MAD_N = L0BSizeBytes / kNumL0Stages / MAD_K;
    static_assert(BLOCK_N % MAD_N == 0);
    static_assert(MAD_M * BLOCK_N * sizeof(float) <= L0CSizeBytes);

    // Shared B reuses routed B's region; events 5-7 permit at most three stages
    static constexpr __aicore__ uint32_t get_num_shared_l1b_stages() {
        constexpr uint32_t num_slots = kNumL1BStages * L0BSizeBytes / (BLOCK_N * BLOCK_K);
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
constexpr PipeEvent<PIPE_MTE2, PIPE_MTE1> kL1AFull{0}, kSFFull{3}, kSharedL1BFull{5};
constexpr PipeEvent<PIPE_MTE1, PIPE_MTE2> kL1AEmpty{0}, kSFEmpty{3}, kSharedL1BEmpty{5};

// AIC: A uses 0–1; split-N B uses 2–3, otherwise readiness shares A's events
constexpr PipeEvent<PIPE_MTE1, PIPE_M> kL0AFull{0}, kL0BFull{2};
constexpr PipeEvent<PIPE_M, PIPE_MTE1> kL0AEmpty{0}, kL0BEmpty{2};

// AIV0: FP4 has 3 stages at 0; NZ has 2 stages at 0
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kFP4Full{0};
constexpr PipeEvent<PIPE_V, PIPE_MTE2> kFP4Empty{0};
constexpr PipeEvent<PIPE_V, PIPE_MTE3> kNZFull{0};
constexpr PipeEvent<PIPE_MTE3, PIPE_V> kNZEmpty{0};
constexpr PipeEvent<PIPE_V, PIPE_S> kMetadataDone{0};

// pull 的两个 UB 槽经历远端 MTE2 读 -> 本地 MTE3 写；push 的四槽经历本地 MTE2 读 -> 远端 MTE3 写。
// 两组槽和事件号分开，因此交替推进两条路径不会覆盖彼此的在途 payload。
// AIV1: pull 0–1 and push 2–5, full forward and empty backward
constexpr uint32_t kNumPullStages = 2, kNumPushStages = 4;
constexpr PipeEvent<PIPE_MTE2, PIPE_MTE3> kPullFull{0}, kPushFull{2};
constexpr PipeEvent<PIPE_MTE3, PIPE_MTE2> kPullEmpty{0}, kPushEmpty{2};

// AIV1: top-k weights, epilogue output, and store completion
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kTopkFull{1};
constexpr PipeEvent<PIPE_V, PIPE_MTE2> kTopkEmpty{0};
constexpr PipeEvent<PIPE_V, PIPE_MTE3> kQuantFull{0};
constexpr PipeEvent<PIPE_MTE3, PIPE_S> kPullDone{0}, kStoresDone{2};

// AIV0/AIV1: reduce stages 6–7 and the final communication drain
constexpr uint32_t kNumReduceStages = 2;
constexpr PipeEvent<PIPE_MTE2, PIPE_V> kReduceFull{6};
constexpr PipeEvent<PIPE_V, PIPE_MTE3> kReduceOutputFull{6};
constexpr PipeEvent<PIPE_MTE3, PIPE_MTE2> kReduceEmpty{6};
constexpr PipeEvent<PIPE_MTE3, PIPE_S> kCommDone{3};

// 这组是同 mixed block 的跨核握手：AIV0 写入 L1B 后通知 AIC，AIC 搬到 L0B 后释放 L1B。
// metadata 阶段先用事件 2 通知 AIC；AIC 消费该通知后再初始化 routed L1B，避免阶段复用冲突。
// AIC/AIV0: MTE3 -> MTE1 full, MTE1 -> MTE3 empty
// Metadata uses flag 2 before routed L1B stages 0–3
constexpr IntraBlockEvent<0> kMetadataFull{2};
constexpr IntraBlockEvent<0> kL1BFull{0}, kL1BEmpty{0};

// GMM1：FIX 将 BF16 写 UB -> AIV1 执行激活/量化并写 GM -> 归还该 UB stage。
// GMM2：FIX 直接写 BF16 GM -> AIV1 等待完成并归还 credit -> 发布 Linear2Ready。
// AIC/AIV1: FIX -> V/S full, MTE3/S -> FIX empty
// Linear1 and Linear2 share the same epilogue credits
constexpr IntraBlockEvent<1> kEpilogueFull{0}, kEpilogueEmpty{0};

// Block sync: both AIVs' MTE3 -> AIC Scalar after final output stores
constexpr uint32_t kCombineDoneID = 4;

// CANN 9.2 forwards mode 12 but does not name write-through-share in its C API enum
constexpr auto kWriteThroughShare = static_cast<asc_store_l2_cache_mode>(12);

template <const auto& workspace>
struct CombineReduceState {
    static constexpr uint32_t kNumBitsPerWord = sizeof(uint32_t) * 8;
    static constexpr uint32_t kUninitializedStage = ~0u;

    // Resume at a channel boundary
    uint32_t token_idx, channel_idx = 0, stage_idx = kUninitializedStage;
    uint32_t num_remaining_tokens;

    // valid_topk_mask 只表示当前被归并 token 的有效 route；completed_blocks 缓存各 producer 单调前缀。
    // reduced_token_mask 则按本 AIV 拥有的 token 集合保留已完成位，两种 mask 生命周期不同。
    uint32_t valid_topk_mask = 0;
    bool shared_output_ready = false;
    uint32_t blocking_rank_idx = 0;
    uint64_t completed_blocks[workspace.config.num_ranks]{};

    // One bit per owned token, strided by the AIV count
    uint32_t reduced_token_mask[ceil_div(workspace.config.num_max_tokens_per_rank, kNumAIVs * kNumBitsPerWord)]{};
};

// Describe staged UB scratch without allocating memory
template <const auto& workspace>
__aicore__ constexpr auto make_reduce_buffer(uint32_t addr = 0) {
    constexpr auto& config = workspace.config;
    constexpr uint32_t kNumInputs = config.num_topk + (config.num_shared_experts > 0);
    constexpr uint32_t kNumChannelsPerVector = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);
    // 剩余 UB 分给两份 reduce stage，每 stage 预留 top-k（加 shared 槽）个输入行片段。
    // channel 宽度向量对齐并不超过 H；stage stride 按预留输入数计算，有效输入少也不改变槽间距。
    const uint32_t num_channels = (UBSizeBytes - addr) / (kNumReduceStages * kNumInputs * sizeof(bfloat16_t)) /
        kNumChannelsPerVector * kNumChannelsPerVector;
    return ub_ptr<bfloat16_t>(kNumInputs, config.hidden < num_channels ? config.hidden : num_channels, addr);
}

// True when all tokens are processed; each step issues one output chunk
template <const auto& workspace, bool kOverlapReduce>
__aicore__ inline bool advance_combine_reduce(__gm__ uint8_t* buffer, __gm__ bfloat16_t* output,
                                      uint32_t num_tokens, CombineReduceState<workspace>& state,
                                      const ub_ptr<bfloat16_t>& ub_reduce_inputs, uint32_t num_steps = ~0u) {
    constexpr auto& config = workspace.config;
    constexpr uint32_t kNumLinear2NBlocks = config.hidden / BLOCK_N;
    constexpr uint32_t kNumExpertsPerRank = config.num_experts / config.num_ranks;
    constexpr bool kHasSharedExperts = config.num_shared_experts > 0;

    const uint32_t num_chunk_channels = ub_reduce_inputs.shape_n;

    const auto combine_input = workspace.combine_input.get_ptr(buffer);
    const auto topk_idx = workspace.input.topk_idx.template get_ptr<uint64_t>(buffer);
    auto& stage_idx = state.stage_idx;

    // Wait for all active shared Linear2 cores
    if (kOverlapReduce and kHasSharedExperts and state.num_remaining_tokens > 0 and not state.shared_output_ready) {
        // shared Linear2 的 slot 0 统计完成全部 shared 任务的活跃核；目标是 min(shared task 总数,32)。
        // 它不是 routed 的 N-block 计数，不能用 H/256 作为此处的等待目标。
        state.shared_output_ready = comm::is_block_ready<workspace>(buffer, layout::kLinear2Ready, 0,
            min(ceil_div(num_tokens, BLOCK_M) * kNumLinear2NBlocks, kNumAICores));
        if (not state.shared_output_ready)
            return false;
    }

    while (state.num_remaining_tokens > 0) {
        const uint32_t token_idx = state.token_idx;
        // overlap 模式每个 AIV 按步长 64 分配 token，绕回该核的首 token 继续探测。
        // 不就绪时换一个 token；bitmap 保证已归并 token 不会被第二次写出。
        const uint32_t next_token_idx = not kOverlapReduce ? token_idx + 1 :
            token_idx + kNumAIVs < num_tokens ? token_idx + kNumAIVs : token_idx % kNumAIVs;
        // 本核拥有 token=aiv_idx+n*64；n/32 定位 bitmap word，n%32 定位 bit，避免为每核分配全 T bitmap。
        const uint32_t word_idx = token_idx / (kNumAIVs * state.kNumBitsPerWord);
        const uint32_t token_bit = 1u << (token_idx / kNumAIVs % state.kNumBitsPerWord);

        // Skip completed tokens
        if (kOverlapReduce and (state.reduced_token_mask[word_idx] & token_bit)) {
            state.token_idx = next_token_idx;
            continue;
        }

        // 早归并必须逐 route 核对实际 producer block：dispatch 做过行置换，不能从 token 顺序推断。
        // prefix 是已完成回传的块数（右开边界）；只有该 token 的所有有效 route 都落在 prefix 内才可读取。
        // Check the actual producer block for each route before the first chunk
        if (kOverlapReduce and state.channel_idx == 0) {
            // 从该 token 的第一个 channel chunk 开始建立完整 valid mask；中间 chunk 恢复时沿用同一 mask。
            state.valid_topk_mask = 0;
            for (uint32_t topk_slot_idx = 0; topk_slot_idx < config.num_topk; ++ topk_slot_idx) {
                const uint32_t token_topk_idx = token_idx * config.num_topk + topk_slot_idx;
                const uint64_t expert_idx = asc_load_dev(topk_idx + token_topk_idx);
                if (expert_idx >= config.num_experts)
                    continue;
                const uint32_t peer_idx = expert_idx / kNumExpertsPerRank;
                // 初值 ~0u 表示专家端尚未公布这条 route 的 producer block；有限 prefix 不会放行它。
                // 只有所有有效 route 的实际 block 都位于对应 producer 的完成前缀内，才允许归并当前 token。
                const uint32_t pool_block_idx = asc_load_dev(workspace.route_m_blocks.get_ptr(buffer, token_topk_idx));
                // Completed prefixes only advance; refresh a peer when its cached value is insufficient
                auto& completed_blocks = state.completed_blocks[peer_idx];
                if (pool_block_idx >= completed_blocks) {
                    // cached prefix 不足才重读远端更新的本地槽；prefix 只前进，不逐 token 重复读取所有 peer 状态。
                    completed_blocks = asc_load_dev(workspace.metadata.combine_ready_prefix.get_ptr(buffer, peer_idx));
                    if (pool_block_idx >= completed_blocks) {
                        state.blocking_rank_idx = peer_idx;
                        state.token_idx = next_token_idx;
                        return false;
                    }
                }
                state.valid_topk_mask |= 1u << topk_slot_idx;
            }
        }

        // Initialize UB credits on first use
        if (stage_idx == state.kUninitializedStage) {
            notify<PIPE_MTE3, PIPE_MTE2, kNumReduceStages>(kReduceEmpty);
            stage_idx = 0;
        }

        for (uint32_t channel_base = state.channel_idx; channel_base < config.hidden; channel_base += num_chunk_channels) {
            const uint32_t num_channels = min(config.hidden - channel_base, num_chunk_channels);
            // 复用 reduce stage 前等上一轮最终输出 MTE3 完成；随后 MTE2 读取各 route 的同一 channel 片段。
            wait<PIPE_MTE3, PIPE_MTE2>(kReduceEmpty, stage_idx);

            // Pack valid routes, then shared output; keep the reserved stage stride
            // 非法专家 route 不读取旧接收槽；有效 route 在 UB 内压紧，shared 输出作为最后一个输入。
            // FP32 sum 输入 stride 是本次 num_channels，UB stage 之间仍保留最大 stride。
            uint32_t num_inputs = 0;
            for (uint32_t topk_slot_idx = 0; topk_slot_idx < config.num_topk + kHasSharedExperts; ++ topk_slot_idx) {
                if (topk_slot_idx < config.num_topk and (kOverlapReduce ? (state.valid_topk_mask & (1u << topk_slot_idx)) == 0 :
                        asc_load_dev(topk_idx + static_cast<uint64_t>(token_idx) * config.num_topk + topk_slot_idx) >= config.num_experts))
                    continue;
                const auto src = combine_input + workspace.get_combine_m_idx(topk_slot_idx, token_idx) * config.hidden + channel_base;
                copy_gm_to_ub(ub_reduce_inputs[stage_idx].ptr() + num_inputs++ * num_channels,
                              src, 1, num_channels, num_channels * sizeof(bfloat16_t), asc_load_l2_cache_mode::NOTALLOC_KEEP);
            }
            notify<PIPE_MTE2, PIPE_V>(kReduceFull, stage_idx);

            // Accumulate in FP32, then round once to BF16
            wait<PIPE_MTE2, PIPE_V>(kReduceFull, stage_idx);
            // 等待本次输入 DMA 后在原 UB 中求和，结果覆盖第一输入行；MTE3 写输出后才能释放该 stage。
            vf_combine_reduce(ub_reduce_inputs[stage_idx].ptr(), num_inputs, num_channels);
            notify<PIPE_V, PIPE_MTE3>(kReduceOutputFull, stage_idx);

            // Store BF16 output
            wait<PIPE_V, PIPE_MTE3>(kReduceOutputFull, stage_idx);
            copy_ub_to_gm(output + static_cast<uint64_t>(token_idx) * config.hidden + channel_base,
                          ub_reduce_inputs[stage_idx].ptr(), 1, num_channels,
                          num_channels * sizeof(bfloat16_t), kWriteThroughShare);
            notify<PIPE_MTE3, PIPE_MTE2>(kReduceEmpty, stage_idx);
            stage_idx = (stage_idx + 1) % kNumReduceStages;

            // Save the resume position
            // token 可能分多个 channel chunk；记录下一 chunk 边界，避免让出流水后重读或重写已完成片段。
            state.channel_idx = channel_base + num_channels;
            if (state.channel_idx == config.hidden) {
                state.channel_idx = 0;
                state.token_idx = next_token_idx;
                if constexpr (kOverlapReduce)
                    state.reduced_token_mask[word_idx] |= token_bit;
                -- state.num_remaining_tokens;
            }

            if (-- num_steps == 0)
                return state.num_remaining_tokens == 0;
        }
    }
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
    if constexpr (kOverlapReduce) {
        notify<PIPE_MTE3, PIPE_S>(kCommDone);
        wait<PIPE_MTE3, PIPE_S>(kCommDone);
    } else {
        comm::rank_barrier<workspace, 1>(buffer, sym_buffer, aiv_idx);
    }

    // Drain reduce stages before changing chunk width
    if (reduce.stage_idx != reduce.kUninitializedStage)
        wait<PIPE_MTE3, PIPE_MTE2, kNumReduceStages>(kReduceEmpty);
    reduce.stage_idx = reduce.kUninitializedStage;

    // Reduce remaining tokens
    // After draining, both AIVs reuse UB from offset 0 for two reduce stages
    constexpr auto ub_reduce_inputs = make_reduce_buffer<workspace>();
    comm::wait_until([&]() __aicore__ {
        return advance_combine_reduce<workspace, kOverlapReduce>(buffer, output, num_tokens, reduce, ub_reduce_inputs);
    }, [&]() __aicore__ {
        if (config.num_shared_experts > 0 and not reduce.shared_output_ready) {
            comm::print_local_timeout(rank_idx);
            return;
        }
        // This is the last observed blocker; token_idx already points to the next probe
        const uint32_t peer_idx = reduce.blocking_rank_idx;
        comm::print_peer_timeout(rank_idx, aiv_idx, peer_idx,
            asc_load_dev(workspace.metadata.combine_ready_prefix.get_ptr(buffer, peer_idx)), false);
    });

    // 本核 token 已全部归并不等于所有 peer 停止写：空 rank 也必须等每个 peer 的 all-ones 完成标记。
    // 该收尾保护下一次调用的输入/metadata/progress 复用，不能因本地无输出而跳过。
    // Empty ranks also wait for peer writes before the next invocation reuses scratch
    if constexpr (kOverlapReduce) {
        for (uint32_t peer_idx = aiv_idx; peer_idx < config.num_ranks; peer_idx += kNumAIVs) {
            const auto prefix_ptr = workspace.metadata.combine_ready_prefix.get_ptr(buffer, peer_idx);
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
    constexpr uint32_t kNumLinear1NBlocks = config.intermediate_hidden / (BLOCK_N / 2);
    constexpr uint32_t kNumLinear2NBlocks = config.hidden / BLOCK_N;
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
    constexpr ub_ptr<uint8_t> ub_fp4(1, L0BSizeBytes / 2);
    constexpr ub_ptr<uint8_t> ub_nz(1, L0BSizeBytes + MMAConfig<128>::BLOCK_K, ub_fp4.offset(kNumFP4Stages));
    constexpr auto ub_reduce_inputs = make_reduce_buffer<workspace>(ub_nz.offset(kNumNZStages));

    static_assert(ub_nz.offset(kNumNZStages) <= UBSizeBytes and ub_reduce_inputs.shape_n > 0, "AIV0 UB overflow");

    // Initialize UB reuse credits; AIC releases the initial L1B credits
    notify<PIPE_V, PIPE_MTE2, kNumFP4Stages>(kFP4Empty);
    notify<PIPE_MTE3, PIPE_V, kNumNZStages>(kNZEmpty);

    // Size the reduce window from local output tokens
    const uint32_t num_reduce_window_m_blocks = ceil_div(num_tokens, BLOCK_M);
    sched::LinearTaskState schedule;
    sched::ExpertCursor<workspace> expert_cursors[2]{};
    uint32_t fp4_stage_idx = 0, nz_stage_idx = 0;
    uint32_t l1b_stage_idx = 0;

    while (true) {
        sched::BlockPhase block_phase;
        // AIC handles shared weights; zero shared tokens skips those tasks
        // 传 num_shared_tokens=0 只重放 routed 任务；shared 权重已经是 FP8，由 AIC 自行装入 L1。
        const auto task_idx = sched::take_linear_task<workspace>(0, num_total_m_blocks, schedule, core_idx, block_phase);
        if (block_phase == sched::BlockPhase::kNone)
            break;

        // Select task weights
        const bool is_linear1 = block_phase == sched::BlockPhase::kLinear1;
        const uint32_t num_n_blocks = is_linear1 ? kNumLinear1NBlocks : kNumLinear2NBlocks;
        const uint32_t num_k = is_linear1 ? config.hidden : config.intermediate_hidden;
        // task 按 M-block 再 N-block 展平，两层 N-block 数不同，用各自 num_n_blocks 解码。
        const uint32_t pool_block_idx = task_idx / num_n_blocks;
        const uint32_t n_block_idx = task_idx % num_n_blocks;
        const uint32_t layer_idx = is_linear1 ? 0 : 1;
        const uint32_t valid_m = expert_cursors[layer_idx].advance_to(buffer, pool_block_idx);
        // 同专家有多个 M-block 时偏向保留权重的 L2 局部性；这里仍逐任务搬运、展开权重，
        // 不能把缓存 hint 理解成跨 M-block 的 L1B 常驻或显式跳过重复加载。
        const auto weight_l2_hint = expert_cursors[layer_idx].num_expert_tokens > BLOCK_M ?
            asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP;
        const uint64_t weight_block_idx =
            static_cast<uint64_t>(expert_cursors[layer_idx].local_expert_idx) * num_n_blocks + n_block_idx;
        // 权重按 [expert][N256][K/2 packed bytes] 存储；每字节两份 FP4，offset 的 K 维以字节计。
        const auto weight_data = (is_linear1 ? weights.l1.data : weights.l2.data) +
                                 weight_block_idx * BLOCK_N * (num_k / 2);

        const auto load_weights = [&]<uint32_t MAD_M>() __aicore__ {
            constexpr uint32_t BLOCK_K = MMAConfig<MAD_M>::BLOCK_K;
            constexpr uint32_t MAD_N = MMAConfig<MAD_M>::MAD_N;
            constexpr uint32_t kNumPackedCols = BLOCK_K / 2;
            constexpr uint32_t kNumColGroups = BLOCK_K / get_frac_k<uint8_t>();
            constexpr l1_ptr<float8_e4m3_t, Major::K> l1b(MAD_N, BLOCK_K, kNumL1AStages * L0ASizeBytes);

            for (uint32_t k_idx = 0; k_idx < num_k; k_idx += BLOCK_K) {
                // valid_k remains a multiple of MAD_K, including the final K block
                const uint32_t valid_k = min(BLOCK_K, num_k - k_idx);
                for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N) {
                    // Load packed FP4
                    // Vector 已读完该 packed FP4 槽后，MTE2 才能填新 K/N 子块；源行 stride 为完整专家 K/2。
                    wait<PIPE_V, PIPE_MTE2>(kFP4Empty, fp4_stage_idx);
                    copy_gm_to_ub(ub_fp4[fp4_stage_idx],
                                  const_cast<__gm__ uint8_t*>(weight_data + static_cast<uint64_t>(n_mad_idx) * (num_k / 2) + k_idx / 2),
                                  MAD_N, valid_k / 2, num_k / 2, kNumPackedCols, weight_l2_hint);
                    notify<PIPE_MTE2, PIPE_V>(kFP4Full, fp4_stage_idx);

                    // Dequantize to FP8 NZ
                    wait<PIPE_MTE2, PIPE_V>(kFP4Full, fp4_stage_idx);
                    // NZ 槽要等上一轮 UB->L1 读完才能覆盖；packed FP4 与 NZ 两组独立轮转，分别回收。
                    wait<PIPE_MTE3, PIPE_V>(kNZEmpty, nz_stage_idx);
                    vf_fp4_to_fp8_nz<MAD_N, BLOCK_K>(ub_fp4[fp4_stage_idx].ptr(), ub_nz[nz_stage_idx].ptr());
                    // 展开已消费完原 FP4 槽，可提早归还给下一轮 MTE2；NZ 输出仍由 MTE3 占用。
                    notify<PIPE_V, PIPE_MTE2>(kFP4Empty, fp4_stage_idx);
                    notify<PIPE_V, PIPE_MTE3>(kNZFull, nz_stage_idx);

                    // Publish L1B to AIC
                    wait<PIPE_V, PIPE_MTE3>(kNZFull, nz_stage_idx);
                    // 等待 AIC MTE1 释放目标 L1B 后才能写入；即使 NZ 已就绪也不能覆盖 AIC 尚未读取的权重。
                    wait_intra_block<PIPE_MTE3, 0>(kL1BEmpty, l1b_stage_idx);
                    // 源 NZ UB 含额外 padding，复制指令按 kNumColGroups/MAD_N 跳过 padding，L1 仅保存真实矩阵块。
                    // 它只改变局部布局，不改变全局 packed FP4 的 expert/N/K 排列。
                    asc_copy_ub2l1(l1b[l1b_stage_idx].template ptr<uint8_t>(), ub_nz[nz_stage_idx].ptr(), kNumColGroups, MAD_N, 1, 0);
                    // 通知保护的是该 L1B N/K 子块；AIC 只在收到 Full 后搬入 L0B。随后本核可以释放 NZ UB。
                    notify_intra_block<PIPE_MTE3, 0>(kL1BFull, l1b_stage_idx);
                    notify<PIPE_MTE3, PIPE_V>(kNZEmpty, nz_stage_idx);

                    // 三个 stage index 分别为 FP4(3槽)、NZ(2槽)、L1B(4槽)；不要要求它们同余或同时回收。
                    fp4_stage_idx = (fp4_stage_idx + 1) % kNumFP4Stages;
                    nz_stage_idx = (nz_stage_idx + 1) % kNumNZStages;
                    l1b_stage_idx = (l1b_stage_idx + 1) % kNumL1BStages;
                }

                // Reduce one chunk in unused UB
                if (kOverlapReduce and num_total_m_blocks > kNumRingBlocks and pool_block_idx + num_reduce_window_m_blocks >= num_total_m_blocks)
                    advance_combine_reduce<workspace, kOverlapReduce>(buffer, output, num_tokens, reduce, ub_reduce_inputs, 1);
            }
        };
        if (valid_m <= 128)
            load_weights.template operator()<128>();
        else
            load_weights.template operator()<256>();
    }
}

// AIC: shared and routed GEMMs
template <const auto& workspace>
__aicore__ inline void aic_gemm_pipeline(__gm__ uint8_t* buffer, uint32_t num_tokens, uint32_t rank_idx,
                                         const layout::Weights& weights, uint32_t core_idx) {
    constexpr auto& config = workspace.config;
    constexpr uint32_t kNumRingBlocks = workspace.get_num_ring_blocks();
    constexpr bool kHasSharedExperts = config.num_shared_experts > 0;
    constexpr uint32_t kNumSFPairsPerChunk = 32;
    constexpr l1_ptr<int16_t> l1_sfa(BLOCK_M, kNumSFPairsPerChunk,
                                   kNumL1AStages * L0ASizeBytes + kNumL1BStages * L0BSizeBytes);
    constexpr l1_ptr<int16_t> l1_sfb(BLOCK_N, kNumSFPairsPerChunk, l1_sfa.offset(kNumSFStages));
    constexpr ub_ptr<bfloat16_t> ub_accum(STORE_BLOCK_M, BLOCK_N);
    static_assert(l1_sfb.offset(kNumSFStages) <= L1SizeBytes, "L1 overflow");

    // issued 序号只属于本 AIC 与它的 AIV1；各 core 的进度可不同，跨 core 的 M-block 输入用 progress 原子计数汇合。
    const auto num_issued_gemms_ptr = workspace.progress.num_issued_gemms.get_ptr(buffer, core_idx);
    const uint32_t num_shared_m_blocks = kHasSharedExperts ? ceil_div(num_tokens, BLOCK_M) : 0;
    const bool reuse_shared_weights = num_shared_m_blocks > 1;
    const auto routed_output = workspace.routed.output.get_ptr(buffer);
    const auto combine_input = workspace.combine_input.get_ptr(buffer);

    // 首次给 L0A/B、L1A 和 SF 槽发 Empty，表示尚无旧消费者；routed L1B 另由 shared 阶段结束后的 AIC 初始化。
    // Initialize reuse credits
    asc_set_mmad_direction_n();
    notify<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0AEmpty);
    notify<PIPE_M, PIPE_MTE1, kNumL0Stages>(kL0BEmpty);
    notify<PIPE_MTE1, PIPE_MTE2, kNumL1AStages>(kL1AEmpty);
    notify<PIPE_MTE1, PIPE_MTE2, kNumSFStages>(kSFEmpty);
    if constexpr (kHasSharedExperts)
        notify<PIPE_MTE1, PIPE_MTE2, 3>(kSharedL1BEmpty);

    sched::LinearTaskState schedule;
    sched::ExpertCursor<workspace> expert_cursors[2]{};
    uint32_t l1a_stage_idx = 0;
    uint32_t l1b_stage_idx = 0;
    uint32_t sf_stage_idx = 0;
    uint8_t num_issued_gemms = 0;
    bool routed_l1b_initialized = false;
    uint32_t num_total_m_blocks = 0;  // Routed pool only; shared blocks are separate
    bool metadata_ready = false;
    const auto num_m_blocks_certificate = workspace.metadata.num_m_blocks_certificate.get_ptr(buffer);

    while (true) {
        sched::BlockPhase block_phase;
        const auto task_idx = sched::take_linear_task<workspace>(
            num_tokens, num_total_m_blocks, schedule, core_idx, block_phase, metadata_ready);
        if (block_phase == sched::BlockPhase::kNone) {
            if (metadata_ready)
                break;

            // shared 任务允许在 routed metadata 未就绪时先执行；无可执行任务时才等待 AIV0 的 metadata 事件。
            // 消费后读取 certificate，后续 kNone 才表示 routed 调度结束。
            // Consume metadata before routed L1B reuses the same intra-block event
            wait_intra_block<PIPE_S, 0>(kMetadataFull);
            num_total_m_blocks = asc_load_dev(num_m_blocks_certificate) & layout::kCountValueMask;
            metadata_ready = true;
            continue;
        }

        // Decode the GEMM task
        const bool is_shared = block_phase == sched::BlockPhase::kSharedLinear1 or block_phase == sched::BlockPhase::kSharedLinear2;
        const bool is_linear1 = block_phase == sched::BlockPhase::kLinear1 or block_phase == sched::BlockPhase::kSharedLinear1;

        // Shared experts widen Linear1 N and Linear2 K
        const uint32_t intermediate_hidden = is_shared ? config.shared_intermediate_hidden : config.intermediate_hidden;
        const uint32_t num_linear1_n_blocks = intermediate_hidden / (BLOCK_N / 2);
        const uint32_t num_n_blocks = is_linear1 ? num_linear1_n_blocks : config.hidden / BLOCK_N;
        const uint32_t num_k = is_linear1 ? config.hidden : intermediate_hidden;
        const uint32_t num_sf_pairs = num_k / MX_SF_DIVISOR;
        const uint32_t pool_block_idx = task_idx / num_n_blocks;
        const uint32_t n_block_idx = task_idx % num_n_blocks;
        const uint32_t layer_idx = is_linear1 ? 0 : 1;
        const uint32_t valid_m = is_shared ? min(num_tokens - pool_block_idx * BLOCK_M, BLOCK_M) :
            expert_cursors[layer_idx].advance_to(buffer, pool_block_idx);
        const auto weight_sf_l2_hint = (is_shared ? reuse_shared_weights : expert_cursors[layer_idx].num_expert_tokens > BLOCK_M) ?
            asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP;
        // MMAD 的 M 对齐到 FRAC_MN；逻辑有效行 valid_m 仍用于 FIX 写回，避免把对齐 padding 当真实 token。
        const uint32_t aligned_valid_m = aligned(valid_m, FRAC_MN);
        const uint64_t weight_block_idx = is_shared ? n_block_idx :
            static_cast<uint64_t>(expert_cursors[layer_idx].local_expert_idx) * num_n_blocks + n_block_idx;
        // B scale 物理布局 [expert][N-block][K-pair][N256]；weight_block_idx 选择 expert/N tile，sf_offset 跨完整 K。
        const uint64_t sf_offset = weight_block_idx * num_sf_pairs * BLOCK_N;
        const auto task_weights = is_linear1 ? (is_shared ? weights.shared_l1 : weights.l1) : (is_shared ? weights.shared_l2 : weights.l2);

        // Hand L1B over to AIV0 after shared tasks
        if (not is_shared and not routed_l1b_initialized) {
            l1b_stage_idx = 0;
            notify_intra_block<PIPE_MTE1, 0, kNumL1BStages>(kL1BEmpty);
            routed_l1b_initialized = true;
        }

        // pool_block_idx 是不回绕的逻辑块号，input_block_idx 才是有限 ring 的物理槽号。
        // shared 输入另有空间；routed Linear1/Linear2 分别读取各自独立的输入 ring。
        // Select input data and scales
        const uint32_t input_block_idx = is_shared ? pool_block_idx : layout::get_ring_block_idx<workspace>(pool_block_idx);
        const auto input_layout = is_linear1 ?
            (is_shared ? workspace.input.acts : workspace.routed.l1_acts) :
            (is_shared ? workspace.shared_l2_acts : workspace.routed.l2_acts);
        const uint64_t input_m_idx = static_cast<uint64_t>(input_block_idx) * BLOCK_M;
        const auto input = input_layout.data.get_ptr(buffer, input_m_idx * num_k);
        const auto input_sf = input_layout.sf.get_ptr(buffer, input_m_idx * num_sf_pairs);

        const auto compute = [&]<uint32_t MAD_M>() __aicore__ {
            // M128 使用较宽 K512 和 N128 MAD，M256 使用 K256 和 N256 MAD；各自的 L0 stage/累加布局配套。
            constexpr uint32_t BLOCK_K = MMAConfig<MAD_M>::BLOCK_K;
            constexpr uint32_t MAD_K = MMAConfig<MAD_M>::MAD_K;
            constexpr uint32_t MAD_N = MMAConfig<MAD_M>::MAD_N;
            constexpr auto l0_full = BLOCK_N == MAD_N ? kL0AFull : kL0BFull;
            constexpr uint32_t kNumSharedL1BStages = MMAConfig<MAD_M>::get_num_shared_l1b_stages();
            constexpr l1_ptr<float8_e4m3_t, Major::K> l1a(MAD_M, BLOCK_K, 0);
            constexpr l1_ptr<float8_e4m3_t, Major::K> l1b(MAD_N, BLOCK_K, l1a.offset(kNumL1AStages));
            constexpr l1_ptr<float8_e4m3_t, Major::K> shared_l1b(BLOCK_N, BLOCK_K, l1b.addr);
            constexpr l0a_ptr<float8_e4m3_t> l0a(MAD_M, MAD_K);
            constexpr l0b_ptr<float8_e4m3_t> l0b(MAD_N, MAD_K);
            constexpr l0c_ptr<float> l0c(MAD_M, MAD_N);

            for (uint32_t k_idx = 0; k_idx < num_k; k_idx += BLOCK_K) {
                // valid_k remains a multiple of MAD_K, including the final K block
                const uint32_t valid_k = min(BLOCK_K, num_k - k_idx);
                // 每 int16 scale pair 覆盖 K64；SF 按 32 对一批驻留，即覆盖 K2048，多个 K-block 可共享同一批 scale。
                const uint32_t sf_pair_idx = k_idx / MX_SF_DIVISOR;
                const uint32_t num_chunk_pairs = min(kNumSFPairsPerChunk, num_sf_pairs - sf_pair_idx);

                // Load shared FP8 weights
                if (is_shared) {
                    wait<PIPE_MTE1, PIPE_MTE2>(kSharedL1BEmpty, l1b_stage_idx);
                    const uint64_t weight_offset = weight_block_idx * BLOCK_N * num_k;
                    asc_set_gm2l1_loop_size(1, 1);
                    asc_copy_gm2l1_align(shared_l1b[l1b_stage_idx].template ptr<uint8_t>(),
                                         const_cast<__gm__ uint8_t*>(task_weights.data + weight_offset + k_idx * BLOCK_N),
                                         1, BLOCK_N * valid_k, 0, 0, false,
                                         reuse_shared_weights ? asc_load_l2_cache_mode::NORMAL_LAST_VICTIM : asc_load_l2_cache_mode::NOTALLOC_KEEP,
                                         BLOCK_N * BLOCK_K, BLOCK_N * BLOCK_K);
                    notify<PIPE_MTE2, PIPE_MTE1>(kSharedL1BFull, l1b_stage_idx);
                }

                // Load weight scales once per SF chunk
                // 只有进入新 SF chunk 才加载权重 scale；在其最后一个 K-block 被搬到 L0 后才归还 SF stage。
                if (sf_pair_idx % kNumSFPairsPerChunk == 0) {
                    wait<PIPE_MTE1, PIPE_MTE2>(kSFEmpty, sf_stage_idx);
                    copy_gm_to_l1_mx(l1_sfb[sf_stage_idx],
                                     gm_ptr<int16_t, Major::MN>(BLOCK_N, reinterpret_cast<uintptr_t>(task_weights.sf + sf_offset + sf_pair_idx * BLOCK_N)),
                                     BLOCK_N, num_chunk_pairs, weight_sf_l2_hint);
                }

                // 等待的是整个 M-block 的输入依赖，而不是当前物理核的上一个 GEMM：
                // GMM1 等所有 AIV1 的 dispatch 完成；GMM2 等该块全部 GMM1 N-block 的量化输出。
                // 权重/scale 与激活就绪独立，先提交它们可隐藏部分输入等待。
                // Prefetch independent weights before waiting for activations
                if (k_idx == 0 and (not is_shared or not is_linear1))
                    comm::wait_for_block<workspace>(buffer, rank_idx,
                        is_linear1 ? layout::kDispatched : layout::kLinear1Ready,
                        (is_shared ? 0 : num_shared_m_blocks) + pool_block_idx,
                        is_linear1 ? kNumAICores : num_linear1_n_blocks);

                // Load activations and scales
                // 旧 L1A 已搬入 L0A 后可覆盖，和 L0A 等 MMAD 完成后才能覆盖是不同的释放点。
                wait<PIPE_MTE1, PIPE_MTE2>(kL1AEmpty, l1a_stage_idx);
                copy_gm_to_l1(l1a[l1a_stage_idx],
                              gm_ptr<float8_e4m3_t, Major::K>(num_k, reinterpret_cast<uintptr_t>(input + k_idx)),
                              valid_m, valid_k, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                notify<PIPE_MTE2, PIPE_MTE1>(kL1AFull, l1a_stage_idx);

                if (sf_pair_idx % kNumSFPairsPerChunk == 0) {
                    // GMM1 输入 scale 是 token-major [M][K-pair]；GMM2 输入是块内 [K-pair][M256]。
                    // 布局来自 dispatch 与 SwiGLU 两种生产路径，读取模板的 Major 和 stride 必须分别匹配。
                    if (is_linear1) {
                        copy_gm_to_l1_mx(l1_sfa[sf_stage_idx],
                                         gm_ptr<int16_t, Major::K>(num_sf_pairs, reinterpret_cast<uintptr_t>(input_sf + sf_pair_idx)),
                                         valid_m, num_chunk_pairs, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    } else {
                        copy_gm_to_l1_mx(l1_sfa[sf_stage_idx],
                                         gm_ptr<int16_t, Major::MN>(BLOCK_M, reinterpret_cast<uintptr_t>(input_sf + sf_pair_idx * BLOCK_M)),
                                         valid_m, num_chunk_pairs, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
                    }

                    // One completion covers both scale loads on the same MTE2 queue
                    notify<PIPE_MTE2, PIPE_MTE1>(kSFFull, sf_stage_idx);
                }

                // Wait for L1 inputs
                wait<PIPE_MTE2, PIPE_MTE1>(kL1AFull, l1a_stage_idx);
                if (is_shared) {
                    wait<PIPE_MTE2, PIPE_MTE1>(kSharedL1BFull, l1b_stage_idx);
                } else {
                    for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N)
                        wait_intra_block<PIPE_MTE1, 0>(kL1BFull, (l1b_stage_idx + n_mad_idx / MAD_N) % kNumL1BStages);
                }

                // M128 的 N256 分成两个 N128 MAD，共用一次装入的 L0A；最后一个 N 子块算完才归还 L0A。
                // L0B 每次 MMAD 后独立归还，不能把 A/B 的复用周期混为一谈。
                // Reuse L0A across the BLOCK_N / MAD_N multiplies
                for (uint32_t k_mad_idx = 0; k_mad_idx < valid_k; k_mad_idx += MAD_K) {
                    const uint32_t l0a_stage_idx = k_mad_idx / MAD_K;
                    // 全局 K-pair 在当前 32-pair SF chunk 内取模；K-block 内的 MAD_K 子块同时推进 data 和 scale。
                    const uint32_t sf_pair_idx_in_chunk = (sf_pair_idx + k_mad_idx / MX_SF_DIVISOR) % kNumSFPairsPerChunk;
                    const auto l0a_mad = l0a[l0a_stage_idx].as_mad_aligned(aligned_valid_m, MAD_K);
                    wait<PIPE_M, PIPE_MTE1>(kL0AEmpty, l0a_stage_idx);
                    copy_l1_to_l0a(l0a_mad, l1a[l1a_stage_idx], 0, k_mad_idx);
                    if (sf_pair_idx % kNumSFPairsPerChunk == 0 and k_mad_idx == 0)
                        wait<PIPE_MTE2, PIPE_MTE1>(kSFFull, sf_stage_idx);
                    // 当前 L0A data 与相同 K 范围的 MX scale 成对装载；scale queue 的 Full 等待在 chunk 起点只执行一次。
                    copy_l1_to_l0a_mx(l0a_mad, l1_sfa[sf_stage_idx], 0, sf_pair_idx_in_chunk, MAD_K);
                    // 整个 K 的最后 MAD 子块才允许 FIX 消费累计结果；不是每个 BLOCK_K 结束都写回一次部分和。
                    const bool is_last_k = k_idx + k_mad_idx + MAD_K == num_k;

                    // Keep B reuse credits independent across MAD shapes
                    for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N) {
                        const uint32_t l0c_idx = n_mad_idx / MAD_N;
                        // 将 K 子块和 N 子块映射到双槽 L0B；M128 的两个 N128 累加区独立，M256 只有一个 N256 区。
                        const uint32_t l0b_stage_idx = (l0a_stage_idx * (BLOCK_N / MAD_N) + l0c_idx) % kNumL0Stages;
                        wait<PIPE_M, PIPE_MTE1>(kL0BEmpty, l0b_stage_idx);
                        const auto l1b_src = is_shared ? shared_l1b[l1b_stage_idx] : l1b[(l1b_stage_idx + n_mad_idx / MAD_N) % kNumL1BStages];
                        copy_l1_to_l0b(l0b[l0b_stage_idx], l1b_src, is_shared ? n_mad_idx : 0, k_mad_idx);
                        // B scale 使用 n_mad_idx 选 N128/N256 子块；与 L0B 的实际权重列和 sf_pair_idx_in_chunk 必须一致。
                        copy_l1_to_l0b_mx(l0b[l0b_stage_idx], l1_sfb[sf_stage_idx], n_mad_idx, sf_pair_idx_in_chunk, MAD_K);
                        notify<PIPE_MTE1, PIPE_M>(l0_full, l0b_stage_idx);

                        // B readiness also covers the earlier A/scales loads on the same MTE1 queue
                        const auto l0c_mad = l0c[l0c_idx].as_mad_aligned(aligned_valid_m, MAD_N);
                        wait<PIPE_MTE1, PIPE_M>(l0_full, l0b_stage_idx);
                        // 只在整个 K 的第一步清零累加器；中间 K 步累加，最后一步更新 unit flag 供 FIX 正确等待计算完成。
                        mad(l0c_mad, l0a_mad, l0b[l0b_stage_idx], aligned_valid_m, MAD_N, MAD_K,
                            is_last_k ? asc_unit_flag_mode::ENABLE_UPDATE : asc_unit_flag_mode::ENABLE_KEEP,
                            k_idx + k_mad_idx == 0);
                        // 同一 L0A 会被该 K 子块的所有 N 子块复用；仅最后一次 MAD 后释放，防止提前填下一 K 覆盖输入。
                        if (n_mad_idx + MAD_N == BLOCK_N)
                            notify<PIPE_M, PIPE_MTE1>(kL0AEmpty, l0a_stage_idx);
                        notify<PIPE_M, PIPE_MTE1>(kL0BEmpty, l0b_stage_idx);
                        if (not is_last_k)
                            continue;

                        // routed GMM2 的 BF16 输出也使用 ring。FIX 覆盖槽之前，必须确认上一代
                        // 同槽输出已被所有发送核消费并完成回传；GMM2 计算完成本身不能释放输出槽。
                        // Reuse output rows after all sends complete
                        if (block_phase == sched::BlockPhase::kLinear2 and n_mad_idx == 0 and pool_block_idx >= kNumRingBlocks)
                            comm::wait_for_block<workspace>(buffer, rank_idx, layout::kCombineSent,
                                                            num_shared_m_blocks + pool_block_idx - kNumRingBlocks, kNumAICores);

                        // One epilogue credit covers BLOCK_N columns
                        // GMM1 每次 FIX 写 M128 的 UB stage，GMM2 一次写当前 valid_m 行到 GM。
                        // M128 MAD 的两个 N128 子块写同一个 N256 stage，待最后一半写完才发布 Full。
                        const uint32_t fix_step_m = is_linear1 ? STORE_BLOCK_M : valid_m;
                        for (uint32_t m_idx_in_block = 0; m_idx_in_block < valid_m; m_idx_in_block += fix_step_m) {
                            const uint32_t ub_stage_idx = m_idx_in_block / STORE_BLOCK_M;
                            // 第一 N 子块等 epilogue credit，后续 N 子块共同填同一 stage；消费者归还之前 FIX 不能重写它。
                            if (n_mad_idx == 0)
                                wait_intra_block<PIPE_FIX, 1>(kEpilogueEmpty, ub_stage_idx);
                            if (is_linear1) {
                                // Store MAD_N gate/up columns with a BLOCK_N destination stride
                                const ub_ptr<bfloat16_t> ub_dst(STORE_BLOCK_M, BLOCK_N,
                                                                ub_accum[ub_stage_idx].addr + n_mad_idx * sizeof(bfloat16_t));
                                copy_l0c_to_ub(ub_dst, l0c_mad.offset_nz_m(m_idx_in_block),
                                               min(STORE_BLOCK_M, valid_m - m_idx_in_block),
                                               MAD_N, asc_dual_dst_mode::DUAL_DST_DISABLE, 1);
                            } else {
                                // Store shared output to combine; routed output to the ring
                                const auto output_base = is_shared ? combine_input : routed_output;
                                // shared Linear2 直接落本 rank 的 combine 额外槽，无需远端 push；routed 输出落 BF16 ring 供 AIV1 回传。
                                const uint64_t output_m_idx = is_shared ?
                                    workspace.get_combine_m_idx(config.num_topk, static_cast<uint64_t>(pool_block_idx) * BLOCK_M) :
                                    layout::get_ring_m_idx<workspace>(pool_block_idx);
                                const auto l2_output = output_base + output_m_idx * config.hidden + n_block_idx * BLOCK_N + n_mad_idx;
                                // 这里只写当前 N 子块列，GM row stride 为完整 H；全部 N-block 完成后才可发送完整 hidden 行。
                                copy_l0c_to_gm(l2_output, l0c_mad, valid_m, MAD_N, config.hidden, kWriteThroughShare);
                            }
                            // 这条跨核 Full 位于最后一个 N 子块的 FIX 后；AIV1 不会在 gate 半块完成但 up 半块缺失时启动激活。
                            if (n_mad_idx + MAD_N == BLOCK_N)
                                notify_intra_block<PIPE_FIX, 1>(kEpilogueFull, ub_stage_idx);
                        }
                    }
                }

                // L1A/B 的消费者是 MTE1，而非 MMAD：最后一次 L1->L0 拷贝完成即可释放 L1，MMAD 继续用 L0。
                // Release L1 after the last L0 load
                if (is_shared) {
                    notify<PIPE_MTE1, PIPE_MTE2>(kSharedL1BEmpty, l1b_stage_idx);
                    l1b_stage_idx = (l1b_stage_idx + 1) % kNumSharedL1BStages;
                } else {
                    for (uint32_t n_mad_idx = 0; n_mad_idx < BLOCK_N; n_mad_idx += MAD_N)
                        notify_intra_block<PIPE_MTE1, 0>(kL1BEmpty, (l1b_stage_idx + n_mad_idx / MAD_N) % kNumL1BStages);
                    l1b_stage_idx = (l1b_stage_idx + (BLOCK_N / MAD_N)) % kNumL1BStages;
                }
                notify<PIPE_MTE1, PIPE_MTE2>(kL1AEmpty, l1a_stage_idx);
                l1a_stage_idx = (l1a_stage_idx + 1) % kNumL1AStages;

                // SF chunk 的最后 K-block（或整个 K 尾块）搬到 L0 后释放；data L1A/B 已可按各自周期提前回收。
                if ((sf_pair_idx + BLOCK_K / MX_SF_DIVISOR) % kNumSFPairsPerChunk == 0 or k_idx + valid_k == num_k) {
                    notify<PIPE_MTE1, PIPE_MTE2>(kSFEmpty, sf_stage_idx);
                    sf_stage_idx = (sf_stage_idx + 1) % kNumSFStages;
                }
            }

        };
        // Shared tasks keep one shape so their L1B stages retain the same physical stride
        // routed 每块按 valid_m 选择 MAD 形状；shared 全部块用相同 num_tokens 判定，避免共享 L1B 物理 stride 随尾块变化。
        if ((is_shared ? num_tokens : valid_m) <= 128)
            compute.template operator()<128>();
        else
            compute.template operator()<256>();

        // 发布本核已提交任务序号，使 AIV1 避免在尚未发起的任务上阻塞；它不替代 FIX Full 完成事件。
        // 序号为 uint8 自然回绕，AIV1 用同宽消费序号配对；每个 UB stage 仍有 credit 限制在途任务。
        asc_store_dev(num_issued_gemms_ptr, ++ num_issued_gemms);
    }
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
    constexpr uint32_t kNumLinear1NBlocks = config.intermediate_hidden / (BLOCK_N / 2);
    constexpr uint32_t kNumLinear2NBlocks = config.hidden / BLOCK_N;
    constexpr uint32_t kNumSharedLinear1NBlocks = config.shared_intermediate_hidden / (BLOCK_N / 2);
    constexpr uint32_t kNumRingBlocks = workspace.get_num_ring_blocks();
    constexpr bool kHasSharedExperts = config.num_shared_experts > 0;

    constexpr uint32_t kNumOutputSFPairs = 2;
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
    constexpr ub_ptr<bfloat16_t> ub_accum(STORE_BLOCK_M, BLOCK_N);
    constexpr ub_ptr<float> ub_route_weights(1, STORE_BLOCK_M, ub_accum.offset(kNumEpilogueStages));
    constexpr ub_ptr<int16_t> ub_output_sf(kNumOutputSFPairs, STORE_BLOCK_M, ub_route_weights.offset(1));

    // Dispatch inputs and source ranges
    constexpr ub_ptr<float8_e4m3_t> ub_pull_token(1, config.hidden, ub_output_sf.offset(kNumEpilogueStages));
    constexpr ub_ptr<int16_t> ub_pull_sf(1, aligned(config.num_input_sf_pairs * sizeof(int16_t), kNumUBAlignmentBytes) / sizeof(int16_t),
                                         ub_pull_token.offset(kNumPullStages));
    constexpr ub_ptr<uint32_t> ub_src_m_begin(1, config.num_ranks, aligned(ub_pull_sf.offset(kNumPullStages), kNumUBAlignmentBytes));
    constexpr ub_ptr<uint32_t> ub_src_m_end(1, config.num_ranks, ub_src_m_begin.offset(1));

    // Combine sends and reduction
    constexpr ub_ptr<bfloat16_t> ub_push_token(1, config.hidden, aligned(ub_src_m_end.offset(1), kNumUBAlignmentBytes));
    constexpr auto ub_reduce_inputs = make_reduce_buffer<workspace>(ub_push_token.offset(kNumPushStages));

    // Size the reduce window from local output tokens
    // 此处窗口按实际本地 T 估算尾段位置，但 ring/overlap 编译策略按 Tmax；active T 与 registered capacity 分别用于不同决策。
    const uint32_t num_reduce_window_m_blocks = max(ceil_div(num_tokens, BLOCK_M), 1u);
    const uint32_t num_shared_m_blocks = kHasSharedExperts ? ceil_div(num_tokens, BLOCK_M) : 0;

    static_assert(ub_push_token.offset(kNumPushStages) <= UBSizeBytes and ub_reduce_inputs.shape_n > 0, "AIV1 UB overflow");

    // 初始 Empty credit 允许 AIC 第一次 FIX 写两份 UB；后续 credit 只能由当前 stage 的消费者归还。
    notify_intra_block<PIPE_MTE3, 1, kNumEpilogueStages>(kEpilogueEmpty);
    notify<PIPE_MTE3, PIPE_MTE2, kNumPushStages>(kPushEmpty);
    notify<PIPE_V, PIPE_MTE2>(kTopkEmpty);
    notify<PIPE_MTE3, PIPE_MTE2, kNumPullStages>(kPullEmpty);

    const auto routed_weights = workspace.routed.topk_weights.get_ptr(buffer);
    const auto src_metadata = workspace.routed.src_metadata.get_ptr(buffer);
    const auto src_token_metadata = workspace.metadata.src_token_metadata.get_ptr(buffer);

    uint32_t num_total_m_blocks = 0;  // Routed pool only; shared blocks are separate
    bool metadata_ready = false;
    const auto num_m_blocks_certificate = workspace.metadata.num_m_blocks_certificate.get_ptr(buffer);

    // 完成计数与 payload 的发布顺序：先等相应 MTE3/FIX，DSB_DDR 发布 Scalar metadata，再 atomic_add。
    // Dispatched 的调用者已等 kPullDone，Linear2Ready 的调用者已等 FIX Full；其余在这里排空 MTE3。
    const auto publish_block_progress = [&](layout::ProgressStage stage,
                                            uint32_t slot_begin, uint32_t num_slots) __aicore__ {
        // Wait for stores; dispatch and FIX callers already waited
        if (stage == layout::kLinear1Ready or stage == layout::kCombineSent) {
            notify<PIPE_MTE3, PIPE_S>(kStoresDone);
            wait<PIPE_MTE3, PIPE_S>(kStoresDone);
        }

        // Publish this core's contribution after its Scalar metadata is visible
        asc_sync_data_barrier(DSB_DDR);
        // 一次对多个连续块贡献 +1 只用于批量回传完成；计数是各逻辑块独立的，不随 ring 槽回绕。
        // DSB 还覆盖 Scalar 写的 route weight/src_metadata/route->block 映射；
        // 只等待 MTE3 不能替代这些 Scalar 控制字段的可见性发布。
        for (uint32_t slot_offset = 0; slot_offset < num_slots; ++ slot_offset)
            asc_atomic_add(workspace.progress.blocks.get_ptr(buffer, workspace.get_progress_idx(stage, slot_begin) + slot_offset), 1u);
    };

    const auto num_issued_gemms_ptr = workspace.progress.num_issued_gemms.get_ptr(buffer, core_idx);

    const auto l1_acts = workspace.routed.l1_acts.data.get_ptr(buffer);
    const auto l1_acts_sf = workspace.routed.l1_acts.sf.get_ptr(buffer);
    // pull 状态同时保留逻辑块、当前 owner 行、专家游标和源行置换；一次推进后可暂停，DMA 槽由事件保护。
    struct DispatchPullState {
        uint32_t pool_block_idx = 0;
        uint32_t m_idx_in_block;  // BLOCK_M marks pending stores
        uint32_t valid_m;
        sched::ExpertCursor<workspace> expert;

        // Permute the concatenated source rows without changing their counts
        uint32_t src_m_idx = 0, src_m_stride = 0, src_m_step = 0;
        uint32_t src_rank_idx = 0;
    };
    DispatchPullState pull{.m_idx_in_block = core_idx};

    // 每次只推进有限行，随后返回主循环给 epilogue/push 执行机会。
    // pull 是专家端远程读取：每条 route 都独立拉取源 token；同卡多专家命中同一 token 不做通信去重。
    // Pull one batch of input rows
    const auto advance_dispatch_pull = [&]() __aicore__ {
        if (pull.pool_block_idx == num_total_m_blocks)
            return;

        if (pull.m_idx_in_block >= BLOCK_M) {
            // Wait for the previous block's stores before publishing readiness
            wait<PIPE_MTE3, PIPE_S>(kPullDone);

            // kDispatched 计的是参与搬运的核数，不是有效行数。空行 owner 也必须发布，
            // 否则尾块/小 M 的 AIC 等待 kNumAICores 永远无法满足。
            // Empty row owners must also contribute
            publish_block_progress(layout::kDispatched, num_shared_m_blocks + pull.pool_block_idx, 1);
            ++ pull.pool_block_idx;
            pull.m_idx_in_block = core_idx;
            return;
        }

        if (pull.m_idx_in_block == core_idx) {
            // 仅回收 Linear1 输入 ring：等旧块所有 GMM1 epilogue 完成后才覆盖。
            // 这里用量化完成作为保守释放点；其他 ring 各有自己的消费完成条件。
            // Reuse input rows after all Linear1 N-blocks finish
            if (pull.pool_block_idx >= kNumRingBlocks and
                    not comm::is_block_ready<workspace>(buffer, layout::kLinear1Ready,
                                                        num_shared_m_blocks + pull.pool_block_idx - kNumRingBlocks,
                                                        kNumLinear1NBlocks))
                return;

            // Load source counts and advance the expert
            while (pull.pool_block_idx >= pull.expert.pool_block_end) {
                const uint32_t next_expert_idx = pull.expert.local_expert_idx + 1;
                uint32_t num_expert_tokens = 0, num_sources = 0;
                for (uint32_t src_rank_idx = 0; src_rank_idx < config.num_ranks; ++ src_rank_idx) {
                    const uint32_t count = asc_load_dev(workspace.metadata.expert_recv_count.get_ptr(
                        buffer, workspace.get_expert_recv_idx(src_rank_idx, next_expert_idx)));
                    // 每个源 count 转为专家内拼接区间 [begin,end)；空 peer 的区间长度为 0，定位时自然跳过。
                    ub_src_m_begin.ptr()[src_rank_idx] = num_expert_tokens;
                    num_expert_tokens += count;
                    ub_src_m_end.ptr()[src_rank_idx] = num_expert_tokens;
                    num_sources += count != 0;
                }
                pull.expert.next_expert(num_expert_tokens);
                pull.src_rank_idx = 0;
                if (num_expert_tokens != 0) {
                    // 互素 stride 使 [0, num_expert_tokens) 的行恰好遍历一次，分散源 rank 访问。
                    // 这是路由行的排列优化，不合并 route，也不减少 token payload 字节数。
                    pull.src_m_stride = next_coprime(ceil_div(num_expert_tokens, num_sources), num_expert_tokens) % num_expert_tokens;
                    // 本核负责目标行 core_idx+n*32，因此下一源行在置换序列中跳 32*stride（mod 专家 route 数）。
                    pull.src_m_step = kNumAICores * pull.src_m_stride % num_expert_tokens;
                }
            }
            uint32_t expert_m_begin;
            pull.valid_m = pull.expert.get_m_block_range(pull.pool_block_idx, expert_m_begin);
            // 专家内原行号 = expert_m_begin + core_idx，再作乘 stride 置换。
            // 各块共用同一置换，从而跨块也不重复/遗漏 route；并非在每块内部重新取模。
            pull.src_m_idx = static_cast<uint64_t>(expert_m_begin + core_idx) * pull.src_m_stride % pull.expert.num_expert_tokens;
            if (pull.src_m_idx < ub_src_m_begin.ptr()[pull.src_rank_idx])
                pull.src_rank_idx = 0;
        }

        for (uint32_t step_idx = 0; step_idx < kNumPullStages and pull.m_idx_in_block < pull.valid_m; ++ step_idx) {
            const uint32_t m_idx_in_block = pull.m_idx_in_block;
            pull.m_idx_in_block += kNumAICores;
            // 每核的第 n 行使用 n%2 pull 槽；pool_m_idx 保存全量 metadata，ring_m_idx 定位可回绕 payload。
            const uint32_t stage_idx = m_idx_in_block / kNumAICores % kNumPullStages;
            // 例 ring_blocks=32，逻辑块b=35：payload 使用物理块3，metadata/ready 仍使用逻辑块35。
            // 覆盖块3前等逻辑块3的旧消费者完成；不能拿 ready[35] 去判断是否可覆盖旧槽。
            const uint64_t pool_m_idx = static_cast<uint64_t>(pull.pool_block_idx) * BLOCK_M + m_idx_in_block;
            const uint64_t ring_m_idx = layout::get_ring_m_idx<workspace>(pull.pool_block_idx, m_idx_in_block);
            // 用源前缀区间定位 source rank，再减 begin 得该源的专家内 metadata slot；置换回绕时 rank 游标归零。
            while (pull.src_m_idx >= ub_src_m_end.ptr()[pull.src_rank_idx])
                ++ pull.src_rank_idx;
            const uint32_t src_rank_idx = pull.src_rank_idx;
            const uint32_t src_slot_idx = pull.src_m_idx - ub_src_m_begin.ptr()[src_rank_idx];

            pull.src_m_idx += pull.src_m_step;
            // src_m_step 已取模，当前 idx 与 step 都小于专家总行数，最多减一次即可回绕；回绕后源 rank 搜索从0重启。
            if (pull.src_m_idx >= pull.expert.num_expert_tokens) {
                pull.src_m_idx -= pull.expert.num_expert_tokens;
                pull.src_rank_idx = 0;
            }
            // metadata 是源端先推给本专家 rank 的本地 GM；此处 map 仅用于读取真正 FP8 token/scale 的源端 peer 地址。
            const auto remote_base = sym_buffer.map(buffer, src_rank_idx);

            // metadata 低 32 位是 token * topk + slot，高 32 位是 FP32 路由权重的原始位。
            // 索引仍保留 top-k 槽身份：即使 token 或专家重复，combine 也写入不同 route 槽。
            // Resolve the source token
            wait<PIPE_MTE3, PIPE_MTE2>(kPullEmpty, stage_idx);
            const uint64_t token_metadata = asc_load_dev(src_token_metadata +
                workspace.get_src_metadata_idx(pull.expert.local_expert_idx, src_rank_idx, src_slot_idx));
            const uint32_t token_topk_idx = static_cast<uint32_t>(token_metadata);
            const uint32_t src_token_idx = token_topk_idx / config.num_topk;
            // 同 token 的 j0/j1 即使同卡也会各拉一遍 token；与 DeepEP 的 (token,dst rank) 去重发送不同。
            // 独立 top-k 槽保证两个 route 权重和返回结果不会互相覆盖。
            const uint32_t src_topk_idx = token_topk_idx % config.num_topk;
            const auto remote_token = workspace.input.acts.data.get_ptr(remote_base, static_cast<uint64_t>(src_token_idx) * config.hidden);
            const auto remote_sf = workspace.input.acts.sf.get_ptr(remote_base, static_cast<uint64_t>(src_token_idx) * config.num_input_sf_pairs);

            // 每 route 发两次 MTE2：H 字节 FP8 token + H/64 个 int16 scale pair（共 H/32 字节）。
            // Full 事件位于同一 MTE2 队列的两次读取之后，等待它才可 MTE3 消费两份 UB 数据。
            // Pull token data and scales
            copy_gm_to_ub(ub_pull_token[stage_idx].ptr(), remote_token,
                          1, config.hidden, config.hidden, asc_load_l2_cache_mode::NORMAL_LAST_VICTIM);
            copy_gm_to_ub(ub_pull_sf[stage_idx].ptr(), remote_sf, 1, config.num_input_sf_pairs,
                          config.num_input_sf_pairs * sizeof(int16_t), asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
            notify<PIPE_MTE2, PIPE_MTE3>(kPullFull, stage_idx);

            // 落本地 GMM1 ring 后发 Empty，表示 MTE3 已不再读取该 UB 槽；跨块 ready 还要等 kPullDone。
            // Stage routed inputs
            wait<PIPE_MTE2, PIPE_MTE3>(kPullFull, stage_idx);
            copy_ub_to_gm(l1_acts + ring_m_idx * config.hidden,
                          ub_pull_token[stage_idx].ptr(), 1, config.hidden,
                          config.hidden, kWriteThroughShare);
            copy_ub_to_gm(l1_acts_sf + ring_m_idx * config.num_input_sf_pairs,
                          ub_pull_sf[stage_idx].ptr(), 1, config.num_input_sf_pairs,
                          config.num_input_sf_pairs * sizeof(int16_t), kWriteThroughShare);
            notify<PIPE_MTE3, PIPE_MTE2>(kPullEmpty, stage_idx);

            // Copy route weights as FP32 bits while DMA runs
            // Scalar 写 weight 与 DMA token 搬运重叠；weight 按 pool_m_idx 全量保存，避免输入 ring 回绕后元数据丢失。
            const uint32_t route_weight_bits = static_cast<uint32_t>(token_metadata >> 32);
            asc_store_dev(reinterpret_cast<__gm__ uint32_t*>(routed_weights + pool_m_idx), route_weight_bits);
            // Record the actual row assignment; combine readiness never infers source/token order
            if constexpr (kOverlapReduce)
                // 专家端向源 rank 写回 route 对应的真实逻辑 M-block，供源端早归并核对 producer prefix。
                // 该映射按原 route 存，不受 ring 回绕及源行置换影响。
                asc_store_dev(workspace.route_m_blocks.get_ptr(remote_base, token_topk_idx), pull.pool_block_idx);
            // Source metadata: rank [31:24], combine M [23:0]
            // 高 8 位存 source rank，低 24 位存 topk_slot*Tmax+token；combine 从此恢复目标 rank/槽。
            asc_store_dev(src_metadata + pool_m_idx,
                static_cast<uint32_t>(workspace.get_combine_m_idx(src_topk_idx, src_token_idx)) | (src_rank_idx << 24));
        }

        if (pull.m_idx_in_block >= pull.valid_m) {
            // BLOCK_M marks pending stores; publish on the next step
            // BLOCK_M 是“本核行已发完、尚待 stores 完成”的哨兵；下一次进入先等 kPullDone，再贡献块计数。
            // 没有有效行的 owner 也走该分支，保证尾块的完成参与数固定为 32。
            pull.m_idx_in_block = BLOCK_M;
            notify<PIPE_MTE3, PIPE_S>(kPullDone);
        }
    };

    // phase/task_idx 是下一份消费任务，m_idx_in_block 是已经处理的行前缀；
    // num_completed_gemms 与 AIC issued 同宽配对，只有完成当前任务全部有效行才推进。
    struct EpilogueState {
        sched::LinearTaskState schedule;
        sched::ExpertCursor<workspace> expert;
        uint32_t task_idx;
        uint32_t valid_m;
        sched::BlockPhase phase = sched::BlockPhase::kNone;
        uint32_t m_idx_in_block = 0;
        uint8_t num_completed_gemms = 0;  // Advanced by epilogue, paired with num_issued_gemms
    };
    EpilogueState epi{};

    // Process Linear1 rows or wait for Linear2 stores; return true on progress
    const auto advance_epilogue = [&]() __aicore__ -> bool {
        if (epi.phase == sched::BlockPhase::kNone)
            return false;

        // Issued tasks still need their FIX completion event
        // 只在新任务起点比较生产/消费序号；已开始的 M-block 必须继续处理其剩余 stage/行。
        // 序号相等时立即返回，让主循环执行通信，而不是在尚未提交的 FIX 事件上等待。
        if (epi.m_idx_in_block == 0 and asc_load_dev(num_issued_gemms_ptr) == epi.num_completed_gemms)
            return false;

        const bool is_shared = epi.phase == sched::BlockPhase::kSharedLinear1 or epi.phase == sched::BlockPhase::kSharedLinear2;
        const bool is_linear1 = epi.phase == sched::BlockPhase::kLinear1 or epi.phase == sched::BlockPhase::kSharedLinear1;
        const uint32_t num_n_blocks = is_linear1 ? (is_shared ? kNumSharedLinear1NBlocks : kNumLinear1NBlocks) : kNumLinear2NBlocks;
        const uint32_t pool_block_idx = epi.task_idx / num_n_blocks;

        if (not is_linear1) {
            // Wait for Linear2 stores and return FIX credit
            // GMM2 没有 SwiGLU：等 FIX 的 GM store 完成，归还 credit，再在下方为该 N-block 发布 Linear2Ready。
            wait_intra_block<PIPE_S, 1>(kEpilogueFull);
            notify_intra_block<PIPE_S, 1>(kEpilogueEmpty);
        } else {
            // SwiGLU 写的是 Linear2 输入 ring；必须等上一代同槽的全部 GMM2 N-block 消费完。
            // 与 dispatch 输入 ring、GMM2 输出 ring 的释放条件不同，不能共用一个 ready 标志。
            // Reuse Linear2 inputs after all output N-blocks finish
            // l2_acts ring 回收点：在新 GMM1 epilogue 的第一行检查 b-ring_blocks 的 H/256 个 GMM2 完成。
            // 槽仍被旧 GMM2 读取时返回 false，不占住 AIV1，主循环可推进旧块 combine/pull。
            if (not is_shared and epi.m_idx_in_block == 0 and pool_block_idx >= kNumRingBlocks and
                    not comm::is_block_ready<workspace>(buffer, layout::kLinear2Ready,
                                                        num_shared_m_blocks + pool_block_idx - kNumRingBlocks,
                                                        kNumLinear2NBlocks))
                return false;

            if (epi.m_idx_in_block == 0)
                epi.valid_m = is_shared ? min(num_tokens - pool_block_idx * BLOCK_M, BLOCK_M) :
                    epi.expert.advance_to(buffer, pool_block_idx);
            // 行号拆为 M128 stage 和 stage 内偏移；一个 stage 可分多次 32-row VF 处理，但只在 stage 起点等一次 FIX。
            const uint32_t stage_idx = epi.m_idx_in_block / STORE_BLOCK_M;
            const uint32_t m_idx_in_stage = epi.m_idx_in_block % STORE_BLOCK_M;
            const uint32_t stage_m_begin = stage_idx * STORE_BLOCK_M;
            const uint32_t stage_valid_m = min(STORE_BLOCK_M, epi.valid_m - stage_m_begin);
            // Yield after at most 32 rows
            // 一个 M128 UB stage 分最多4次32行处理；只有 stage 完整处理后才写 GM/归还 FIX credit。
            // VF 每次内部再按2行处理；小步是给通信状态机让出调度机会，不是改变 GEMM M256 切分。
            const uint32_t step_m = min(32u, epi.valid_m - epi.m_idx_in_block);

            // Load route weights and wait for FIX
            if (m_idx_in_stage == 0) {
                if (not is_shared) {
                    const auto route_weights = workspace.routed.topk_weights.get_ptr(
                        buffer, static_cast<uint64_t>(pool_block_idx) * BLOCK_M + stage_m_begin);
                    // route weight UB 只有一份，等上一 stage 的 Vector 读完才能加载新 M128 行权重。
                    wait<PIPE_V, PIPE_MTE2>(kTopkEmpty);
                    copy_gm_to_ub(ub_route_weights.ptr(), route_weights, 1, stage_valid_m,
                                  stage_valid_m * sizeof(float), asc_load_l2_cache_mode::NORMAL_FIRST_VICTIM);
                    notify<PIPE_MTE2, PIPE_V>(kTopkFull);
                }
                // 等待当前 stage 的 gate/up 均已由 FIX 填完，保证 vf_swiglu_quant 读取完整配对输出。
                wait_intra_block<PIPE_V, 1>(kEpilogueFull, stage_idx);
                // SwiGLU 需要 FIX gate/up 与 route weight 两条生产流都完成；单等一个 Full 不能证明另一份输入就绪。
                if (not is_shared)
                    wait<PIPE_MTE2, PIPE_V>(kTopkFull);
            }

            // 一个物理 N256 输出含 gate128 + up128，产生 N128 激活；M 仍是当前块的有效行数。
            // routed 分支在量化前乘 top-k weight，返回结果已加权，最终 reduce 不再乘一次。
            // Quantize one row group
            const auto input = ub_accum[stage_idx].ptr() + m_idx_in_stage * BLOCK_N;
            // FP8 量化输出压紧到同一 BF16 UB 的低地址，避免另开激活输出区；输入/输出 stride 分别为 N256*2B、N128*1B。
            const auto output = reinterpret_cast<__ubuf__ float8_e4m3_t*>(ub_accum[stage_idx].ptr()) +
                                m_idx_in_stage * (BLOCK_N / 2);
            const auto output_sf = ub_output_sf[stage_idx].ptr() + m_idx_in_stage;
            if (is_shared) {
                vf_swiglu_quant<false, STORE_BLOCK_M>(input, output, output_sf, nullptr, step_m, activation_clamp);
            } else {
                vf_swiglu_quant<true, STORE_BLOCK_M>(input, output, output_sf, ub_route_weights.ptr() + m_idx_in_stage, step_m, activation_clamp);
            }
            // 累积处理行前缀；stage 未结束时保留 UB data/scale，不提前释放给 FIX。
            epi.m_idx_in_block += step_m;

            // Finish GM stores before FIX reuses this UB stage
            if (m_idx_in_stage + step_m == stage_valid_m) {
                if (not is_shared)
                    notify<PIPE_V, PIPE_MTE2>(kTopkEmpty);
                notify<PIPE_V, PIPE_MTE3>(kQuantFull);
                wait<PIPE_V, PIPE_MTE3>(kQuantFull);

                const uint32_t n_block_idx = epi.task_idx % num_n_blocks;
                const uint64_t output_m_idx = static_cast<uint64_t>(
                    is_shared ? pool_block_idx : layout::get_ring_block_idx<workspace>(pool_block_idx)) * BLOCK_M;
                const auto dst_layout = is_shared ? workspace.shared_l2_acts : workspace.routed.l2_acts;
                // 每个任务产 N128，只写 [M][n_block*128:...+128]；GM data 行 stride 是整层 I（shared 为扩展 I）。
                const uint32_t num_cols = num_n_blocks * (BLOCK_N / 2);
                const uint32_t num_sf_pairs = num_cols / MX_SF_DIVISOR;
                const auto dst = dst_layout.data.get_ptr(buffer,
                    (output_m_idx + stage_m_begin) * num_cols + n_block_idx * (BLOCK_N / 2));
                // scale 的物理布局为 [M-block][K-pair][M256]；当前任务两对 scale 的列间 stride 为 M256。
                // stage_m_begin 只移动 M 维，不能按 token-major 的 data 布局计算 scale 地址。
                const auto dst_sf = dst_layout.sf.get_ptr(buffer,
                    output_m_idx * num_sf_pairs + n_block_idx * kNumOutputSFPairs * BLOCK_M + stage_m_begin);

                copy_ub_to_gm(dst, ub_ptr<float8_e4m3_t>(STORE_BLOCK_M, BLOCK_N / 2, ub_accum[stage_idx].addr),
                              stage_valid_m, BLOCK_N / 2, num_cols, kWriteThroughShare);
                // 每任务产生两对 scale，每对沿 stage_valid_m 行搬运；GM 目的对间距始终256行，尾块也不压缩 stride。
                copy_ub_to_gm(dst_sf, ub_output_sf[stage_idx],
                              kNumOutputSFPairs, stage_valid_m, BLOCK_M * sizeof(int16_t), kWriteThroughShare);
                // 必须让 data/scale 的 MTE3 读取 UB 完成后再释放 stage；FIX 随后可用同一 UB 写下一任务。
                notify_intra_block<PIPE_MTE3, 1>(kEpilogueEmpty, stage_idx);
            }
            if (epi.m_idx_in_block != epi.valid_m)
                return true;
            epi.m_idx_in_block = 0;
        }

        // 整个任务的有效 M 行都处理完才推进消费序号；GMM1 的 +1 对应一个完整 N128 激活任务。
        ++ epi.num_completed_gemms;

        // routed Linear1Ready/Linear2Ready 按 N-block 计数，分别等 I/128、H/256。
        // shared Linear2 特例：每核处理自己的最后一个 shared task 后只给 slot 0 加一次，早归并等待活跃核数。
        // Shared early reduce counts each active core once in slot 0
        if (is_linear1)
            publish_block_progress(layout::kLinear1Ready, (is_shared ? 0 : num_shared_m_blocks) + pool_block_idx, 1);
        // shared GMM2 的每核最后任务判定为 task_idx+32>=shared task 总数；
        // 只有这一次加 slot0，避免同核多任务重复贡献而过早放行 shared reduce。
        else if (not is_shared or (kOverlapReduce and epi.task_idx + kNumAICores >= num_shared_m_blocks * kNumLinear2NBlocks))
            publish_block_progress(layout::kLinear2Ready, is_shared ? 0 : num_shared_m_blocks + pool_block_idx, 1);
        epi.phase = sched::BlockPhase::kNone;
        return true;
    };

    const auto routed_output = workspace.routed.output.get_ptr(buffer);
    // push.pool_block_idx 是已排入发送的块前缀；num_completed_m_blocks 是已经等 stores 并发布 ready 的前缀。
    // 两者可以不同：允许下一块发送与此前块的远端写在途重叠。
    struct CombinePushState {
        sched::ExpertCursor<workspace> expert;
        uint32_t pool_block_idx = 0;
        uint32_t m_idx_in_block;
        uint32_t valid_m;
        uint32_t num_completed_m_blocks = 0;
    };
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
        if (push.pool_block_idx == num_total_m_blocks or (push.m_idx_in_block == core_idx and
                not comm::is_block_ready<workspace>(buffer, layout::kLinear2Ready,
                                                    num_shared_m_blocks + push.pool_block_idx,
                                                    kNumLinear2NBlocks)))
            return false;

        const uint32_t pool_block_idx = push.pool_block_idx;
        if (push.m_idx_in_block == core_idx)
            push.valid_m = push.expert.advance_to(buffer, pool_block_idx);
        const auto src_metadata_base = src_metadata + static_cast<uint64_t>(pool_block_idx) * BLOCK_M;
        const auto src_block = routed_output + layout::get_ring_m_idx<workspace>(pool_block_idx) * config.hidden;

        // Send at most two rows before yielding
        for (uint32_t step_idx = 0; step_idx < 2 and push.m_idx_in_block < push.valid_m; ++ step_idx) {
            const uint32_t m_idx_in_block = push.m_idx_in_block;
            // 本核负责的第n行使用n%4槽；每次调用最多发送2行，再返回调度器，避免 combine 独占 AIV1。
            const uint32_t stage_idx = (m_idx_in_block / kNumAICores) % kNumPushStages;
            const uint32_t combine_metadata = asc_load_dev(src_metadata_base + m_idx_in_block);
            // 从全量 src_metadata 取回目标槽，payload 则从 ring 取；两者不可都改用 ring 下标。
            const uint32_t combine_m_idx = combine_metadata & 0x00ffffffu;
            const auto src = src_block + m_idx_in_block * config.hidden;

            // Load a BF16 output row
            wait<PIPE_MTE3, PIPE_MTE2>(kPushEmpty, stage_idx);
            copy_gm_to_ub(ub_push_token[stage_idx].ptr(), src, 1, config.hidden,
                          config.hidden * sizeof(bfloat16_t), asc_load_l2_cache_mode::NOTALLOC_KEEP);
            notify<PIPE_MTE2, PIPE_MTE3>(kPushFull, stage_idx);

            const uint32_t dst_rank_idx = combine_metadata >> 24;
            // map 将同一 combine offset 平移到源 rank 的 peer VA；直接 MTE3 写它的独占 route 槽，无结果 AtomicAdd。
            const auto dst = sym_buffer.map(workspace.combine_input.get_ptr(buffer, static_cast<uint64_t>(combine_m_idx) * config.hidden), dst_rank_idx);

            // Send the BF16 row to its source token
            wait<PIPE_MTE2, PIPE_MTE3>(kPushFull, stage_idx);
            copy_ub_to_gm(dst, ub_push_token[stage_idx].ptr(), 1, config.hidden,
                          config.hidden * sizeof(bfloat16_t), asc_store_l2_cache_mode::NOTALLOC_CLEAN);
            // 旧 remote write 消费完 UB 后才能复用 push 槽；4 槽只增加在途行数，不代表 4 条独立网络链路。
            notify<PIPE_MTE3, PIPE_MTE2>(kPushEmpty, stage_idx);
            push.m_idx_in_block += kNumAICores;
        }

        if (push.m_idx_in_block >= push.valid_m) {
            push.m_idx_in_block = core_idx;
            ++ push.pool_block_idx;

            // Wait for batched stores; publish each last-expert block for early reduce
            // overlap 在最后专家的各块也主动 flush，以更频繁发布尾部前缀；普通路径只按4块/最终尾块 flush。
            if (push.pool_block_idx % kNumMBlocksPerWait == 0 or
                    (kOverlapReduce ? push.expert.pool_block_end == num_total_m_blocks : push.pool_block_idx == num_total_m_blocks)) {
                // num_completed_m_blocks 为本核已排空回传的右开前缀；发布 [旧前缀, 当前块末尾) 所有逻辑块。
                publish_block_progress(layout::kCombineSent,
                                       num_shared_m_blocks + push.num_completed_m_blocks, push.pool_block_idx - push.num_completed_m_blocks);
                push.num_completed_m_blocks = push.pool_block_idx;
            }
        }
        return true;
    };

    // 本 rank 的所有发送核都完成此前的块后，才向各源 rank 发布右开完成前缀。
    // 全 1 表示本 rank 全部回传完成；源端同时利用该标记保护下一次调用的 scratch 复用。
    // Publish the exact completed-block prefix shared by all source ranks
    struct CombinePrefixState {
        uint32_t num_published_m_blocks = 0;
        bool finished;
    };
    // publisher 按 source rank 分工；core_idx>=rank数时无需发布，rank>32时每核负责多个 source。
    CombinePrefixState ready_prefix{.finished = not kOverlapReduce or core_idx >= config.num_ranks};

    const auto publish_combine_prefix = [&]() __aicore__ {
        const uint32_t pool_block_end = push.num_completed_m_blocks;
        if (ready_prefix.finished or (num_total_m_blocks != 0 and pool_block_end <= ready_prefix.num_published_m_blocks))
            return;
        // 各核按逻辑块顺序发布；检查最新前缀的末块收到全部 32 核贡献，就代表更早块也已全部完成。
        if (pool_block_end != 0 and not comm::is_block_ready<workspace>(
                buffer, layout::kCombineSent, num_shared_m_blocks + pool_block_end - 1, kNumAICores))
            return;

        // 这是该发布核已确认“全32个发送核完成”的统一前缀，不能仅发送本核 push.pool_block_idx。
        ready_prefix.num_published_m_blocks = pool_block_end;
        ready_prefix.finished = pool_block_end == num_total_m_blocks;
        // 有限值是右开块边界；~0ull 兼作“全部结束”哨兵，即使本 rank 没有任何 routed 块也要发送。
        const uint64_t prefix = ready_prefix.finished ? ~uint64_t{0} : pool_block_end;
        for (uint32_t src_rank_idx = core_idx; src_rank_idx < config.num_ranks; src_rank_idx += kNumAICores) {
            asc_sync_data_barrier(DSB_DDR);
            // 写入对端源 rank 的 prefix[本 producer rank]，每槽仅由指定 core_idx 的 publisher 写，避免多写者。
            asc_store_dev(sym_buffer.map(workspace.metadata.combine_ready_prefix.get_ptr(buffer, rank_idx), src_rank_idx), prefix);
        }
    };

    bool prefer_push = false;
    comm::wait_until([&]() __aicore__ {
        // Service the reduce window; check token readiness inside
        // 只有接近回传尾段且本核所有已发 GEMM 都已消费时才插入 reduce；优先避免妨碍激活关键路径。
        const bool reduce_window_open = kOverlapReduce and metadata_ready and push.pool_block_idx + num_reduce_window_m_blocks >= num_total_m_blocks and
            asc_load_dev(num_issued_gemms_ptr) == epi.num_completed_gemms;

        if (reduce_window_open) {
            publish_combine_prefix();
            // 只有 routed 块数超出 ring 才在通信循环借剩余 UB 做早归并；小工作量留给最终 offset0 的大 UB reduce。
            if (num_total_m_blocks > kNumRingBlocks)
                advance_combine_reduce<workspace, kOverlapReduce>(buffer, output, num_tokens, reduce, ub_reduce_inputs, 2);
        }

        // Fetch the next epilogue task
        if (epi.phase == sched::BlockPhase::kNone)
            epi.task_idx = sched::take_linear_task<workspace>(
                num_tokens, num_total_m_blocks, epi.schedule, core_idx, epi.phase, metadata_ready);

        // Poll metadata arrival
        // AIV0 通过本 block 的 GM flag 通知 AIV1，读取 certificate 前做 DDR barrier；shared GMM1 可先行。
        if (not metadata_ready and epi.phase != sched::BlockPhase::kSharedLinear1 and
            asc_load_dev(workspace.progress.metadata_ready.get_ptr(buffer, core_idx)) != 0) {
            asc_sync_data_barrier(DSB_DDR);
            num_total_m_blocks = asc_load_dev(num_m_blocks_certificate) & layout::kCountValueMask;
            metadata_ready = true;
        }

        const bool pull_finished = metadata_ready and pull.pool_block_idx == num_total_m_blocks;
        // 退出需要 pull、epilogue、push 都结束且 prefix 发布完成；后面的 combine_reduce 负责排空/远端收尾。
        if (pull_finished and epi.phase == sched::BlockPhase::kNone and push.pool_block_idx == num_total_m_blocks and ready_prefix.finished)
            return true;

        // 优先处理可推进的 epilogue，避免通信占住 AIV1、阻塞 GMM2 所需激活。
        // push/pull 交替尝试；状态机在小步边界切换，已发出的 DMA 仍由各槽 credit 保护。
        // Advance epilogue before choosing push or pull
        const bool epi_progress = advance_epilogue();
        if (metadata_ready and not (pull_finished and epi_progress)) {
            // epilogue 已尝试后轮转通信：prefer_push 提供公平机会；push 未就绪则可退回 pull。
            // pull 的槽反压检查立即返回，但已发 DMA 仍可能通过 wait 阻塞；这是小步协作，不是任意指令抢占。
            if ((prefer_push or pull_finished) and advance_combine_push()) {
                prefer_push = false;
            } else if (not pull_finished) {
                // Fall back to pull; prefer push again on the next step
                advance_dispatch_pull();
                prefer_push = true;
            }
        }
        return false;
    }, [&]() __aicore__ { comm::print_local_timeout(rank_idx); });
    if constexpr (kHasSharedExperts)
        asc_sync_data_barrier(DSB_DDR);
}

template <uint32_t kNumRanks, uint32_t kNumExperts, uint32_t kNumSharedExperts,
          uint32_t kNumMaxTokensPerRank, uint32_t kNumTopk, uint32_t kHidden, uint32_t kIntermediateHidden>
__global__ __mix__(1, 2) void mega_moe_impl(__gm__ bfloat16_t* output, __gm__ int32_t* cumulative_local_expert_recv_stats,
                                            __gm__ const uint8_t* l1_weights, __gm__ const int16_t* l1_weights_sf,
                                            __gm__ const uint8_t* l2_weights, __gm__ const int16_t* l2_weights_sf,
                                            __gm__ const uint8_t* shared_l1_weights, __gm__ const int16_t* shared_l1_weights_sf,
                                            __gm__ const uint8_t* shared_l2_weights, __gm__ const int16_t* shared_l2_weights_sf,
                                            uint32_t num_tokens, float activation_clamp,
                                            layout::SymBuffer<> sym_buffer) {
    constexpr auto& workspace = layout::kWorkspace<kNumRanks, kNumExperts, kNumSharedExperts,
                                                   kNumMaxTokensPerRank, kNumTopk, kHidden, kIntermediateHidden>;
    constexpr auto& config = workspace.config;

    // 阈值按注册容量而非本次实际 token 数选择；这是实现中的经验策略，不是硬件通用拐点。
    // overlap 主要在尾段利用空闲 UB 推进归并，并非所有阶段都无条件并发执行 reduce。
    // Measured 32 MiB combine-work threshold for tail reduction
    constexpr bool kOverlapReduce = static_cast<uint64_t>(config.num_max_tokens_per_rank) *
        (config.num_topk + (config.num_shared_experts > 0)) * config.hidden * sizeof(bfloat16_t) > 32ull * 1024 * 1024;

    constexpr uint32_t kNumExpertsPerRank = config.num_experts / config.num_ranks;

    // Block readiness reads the low byte of each completion counter
    static_assert(config.hidden / BLOCK_N <= UINT8_MAX and
                  config.intermediate_hidden / (BLOCK_N / 2) <= UINT8_MAX and
                  config.shared_intermediate_hidden / (BLOCK_N / 2) <= UINT8_MAX);

    asc_init();
    const uint32_t core_idx = static_cast<uint32_t>(block_idx);

    const auto buffer = sym_buffer.get_base_ptr<__gm__ uint8_t*>();
    const uint32_t rank_idx = sym_buffer.rank_idx;
    const auto topk_idx = workspace.input.topk_idx.get_ptr(buffer);
    const auto num_m_blocks_certificate = workspace.metadata.num_m_blocks_certificate.get_ptr(buffer);

    const layout::Weights weights{{l1_weights, l1_weights_sf}, {l2_weights, l2_weights_sf},
                                  {shared_l1_weights, shared_l1_weights_sf}, {shared_l2_weights, shared_l2_weights_sf}};

    // AIC: GEMMs, then wait for both AIVs
    if ASCEND_IS_AIC {
        aic_gemm_pipeline<workspace>(buffer, num_tokens, rank_idx, weights, core_idx);
        // AIC GEMM 提交完后仍等待两份 AIV 的最终输出 MTE3 到达；所有角色结束才可把本次 kernel 视为完成。
        asc_sync_block_wait(PIPE_S, kCombineDoneID);
        return;
    }

    // Partition tokens across AIVs
    const uint32_t subblock_idx = static_cast<uint32_t>(asc_get_sub_block_id());
    // subblock 0 为权重/metadata 核，subblock 1 为 epilogue/通信核；归并 token 使用全局 64-AIV 索引分工。
    const uint32_t aiv_idx = 2 * core_idx + subblock_idx;

    // Overlap mode revisits round-robin tokens; the final-only path uses contiguous ranges
    const uint32_t first_token_idx = kOverlapReduce ? aiv_idx : min(aiv_idx * ceil_div(num_tokens, kNumAIVs), num_tokens);
    const uint32_t num_reduce_tokens = kOverlapReduce ?
        (num_tokens > aiv_idx ? ceil_div(num_tokens - aiv_idx, kNumAIVs) : 0) :
        min(ceil_div(num_tokens, kNumAIVs), num_tokens - first_token_idx);
    CombineReduceState<workspace> reduce{.token_idx = first_token_idx, .num_remaining_tokens = num_reduce_tokens};

    // Host 已在同一 stream 清零 progress 区；device 在这里清零本次接收计数。
    // 随后的 rank barrier 确保所有接收端清零完成，源端才可发布新一轮 metadata/count。
    // Clear counts before metadata publication
    for (uint32_t expert_idx = aiv_idx; expert_idx < kNumExpertsPerRank; expert_idx += kNumAIVs)
        asc_store_dev(workspace.metadata.expert_recv_count_sum.get_ptr(buffer, expert_idx), uint32_t{0});
    if (aiv_idx == 0)
        asc_store_dev(num_m_blocks_certificate, uint32_t{0});
    if (aiv_idx < kNumExpertsPerRank)
        asc_sync_data_barrier(DSB_DDR);
    comm::rank_barrier<workspace>(buffer, sym_buffer, aiv_idx);

    // AIV0: metadata and routed weights
    if (subblock_idx == 0) {
        // 每 worker 512 SIMT 线程；按 route 数启用最多 32 个 worker，T=0 也保留一个 worker 发布零 count。
        const uint32_t num_metadata_workers = min(max(ceil_div(num_tokens * config.num_topk, layout::kNumDispatchThreads), 1u), kNumAICores);
        if (core_idx < num_metadata_workers) {
            // SIMT reads UB offsets; metadata owns this scratch until the VF completes
            __ubuf__ int64_t offsets[kNumRanks];
            for (uint32_t i = 0; i < kNumRanks; ++ i)
                offsets[i] = sym_buffer.offsets[i];
            // Scalar 先填 UB peer offset 表，DSB_UB 后交给 SIMT VF；VF 执行期间该 scratch 不能改作权重槽。
            asc_sync_data_barrier(DSB_UB);
            asc_vf_call<comm::build_dispatch_metadata<workspace, kOverlapReduce>>(
                dim3(layout::kNumDispatchThreads, 1, 1),
                buffer, topk_idx, num_tokens, rank_idx, core_idx, num_metadata_workers, offsets);
            notify<PIPE_V, PIPE_S>(kMetadataDone);
            wait<PIPE_V, PIPE_S>(kMetadataDone);
        }

        const uint32_t num_total_m_blocks = comm::wait_dispatch_metadata<workspace>(buffer, rank_idx, core_idx, cumulative_local_expert_recv_stats);

        // Notify AIV1 via GM and AIC via event
        // 同一份 certificate 分别经 GM flag 通知 AIV1、经跨核事件通知 AIC，两种通知不可互相替代。
        asc_store_dev(workspace.progress.metadata_ready.get_ptr(buffer, core_idx), uint8_t{1});
        notify_intra_block<PIPE_S, 0>(kMetadataFull);

        aiv0_weight_pipeline<workspace, kOverlapReduce>(buffer, output, num_tokens, reduce, num_total_m_blocks, weights, core_idx);
    } else {
        // AIV1: epilogue and communication
        aiv1_epilogue_comm<workspace, kOverlapReduce>(buffer, output, num_tokens, rank_idx, reduce, activation_clamp, core_idx, sym_buffer);
    }

    combine_reduce<workspace, kOverlapReduce>(buffer, output, num_tokens, rank_idx, aiv_idx, reduce, sym_buffer);

    // Notify AIC after final output stores
    // 两份 AIV 都在最终 y store 之后到达；AIC 在等待两者之前不能返回，避免输出/通信仍在途。
    asc_sync_block_arrive(PIPE_MTE3, kCombineDoneID);
}

} // namespace deep_gemm::mega_moe
