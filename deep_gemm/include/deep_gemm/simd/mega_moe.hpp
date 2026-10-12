#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/mega_moe.hpp>

#include <c_api/asc_simd.h>

namespace deep_gemm::mega_moe {

// FP4 -> FP8 NZ, scaled by 2^-6; weight SFs compensate with +6
// Rows map to GEMM N; columns map to K
template <uint32_t kNumRows, uint32_t kNumCols>
__simd_vf__ inline void vf_fp4_to_fp8_nz(__ubuf__ uint8_t* src, __ubuf__ uint8_t* dst) {
    constexpr uint32_t kNumPackedCols = kNumCols / 2;
    // NZ scratch 在 GEMM N 行方向多留一行作 stride padding；真实复制到 L1 的仍是 kNumRows 行。
    constexpr uint32_t kNumPaddedRows = kNumRows + 1;
    constexpr uint32_t kNumPackedColsPerVector = sizeof(vector_int8_t) / 2;  // Two FP4 values per byte
    constexpr uint32_t kNumVectorsPerRow = kNumPackedCols / kNumPackedColsPerVector;
    static_assert(kNumPackedCols % kNumPackedColsPerVector == 0);

    const auto mask = asc_create_mask_b8(PAT_ALL);
    vector_int8_t shift_right, shift_left, fp8_mask;
    asc_duplicate_scalar(shift_right, static_cast<int8_t>(2));
    // 每个 FP4 byte 展开成两份后，通过交替移位抽取高/低 nibble；mask 保留 sign 和指数/尾数位。
    asc_duplicate_scalar(reinterpret_cast<vector_uint16_t&>(shift_left), uint16_t{4});  // Alternating byte shifts: 4, 0
    asc_duplicate_scalar(fp8_mask, static_cast<int8_t>(0x9c));  // Sign and shifted FP4 payload

    auto src_ptr = reinterpret_cast<__ubuf__ int8_t*>(src);
    constexpr uint32_t kNumRowsPerPass = 4;
    static_assert(kNumRows % kNumRowsPerPass == 0);
    constexpr uint32_t kNumVectors = kNumVectorsPerRow * kNumRowsPerPass;

    // Each vector writes one row fragment in NZ order; the extra row pads the stride
    __ubuf__ uint8_t* dst_ptrs[kNumVectors];
    #pragma unroll
    for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx)
        // 将线性 vector 编号拆成 K-fragment 和四行中的 row，直接写成 AIC 可消费的 NZ 排列。
        dst_ptrs[vector_idx] = dst + vector_idx % kNumVectorsPerRow * kNumPaddedRows * 2 * kNumPackedColsPerVector +
                               vector_idx / kNumVectorsPerRow * get_frac_k<uint8_t>();

    for (uint32_t row_idx = 0; row_idx < kNumRows; row_idx += kNumRowsPerPass) {
        // Expand packed nibbles into scaled FP8 values
        vector_int8_t values[kNumVectors];
        #pragma unroll
        for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx) {
            // 一字节两个 FP4，经 upsample/位运算展开为两字节 FP8；数值缩放 2^-6 由预处理后的权重 scale 补偿。
            asc_loadalign_upsample_postupdate(values[vector_idx], src_ptr, kNumPackedColsPerVector);
            asc_shiftleft(values[vector_idx], values[vector_idx], shift_left, mask);
            asc_shiftright(values[vector_idx], values[vector_idx], shift_right, mask);
            asc_and(values[vector_idx], values[vector_idx], fp8_mask, mask);
        }

        // Scatter into padded NZ and advance by four rows
        #pragma unroll
        for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx)
            asc_storealign_postupdate(dst_ptrs[vector_idx], reinterpret_cast<vector_uint8_t&>(values[vector_idx]),
                                      kNumPaddedRows, kNumRowsPerPass, mask);
    }
}

// 输入每行 [gate128 | up128]，输出 128 列 FP8，scale pair 覆盖两个 32-channel 组。
// routed 路径先做 SwiGLU、乘路由权重，再计算量化 scale；shared 路径不乘 route weight。
// 将权重移到最终 combine 再乘会改变中间量化误差，不能视为数值等价的重排。
// Linear1 epilogue: SwiGLU and FP8 quantization
template <bool kApplyRouteWeight, uint32_t STORE_BLOCK_M>
__simd_vf__ inline void vf_swiglu_quant(__ubuf__ bfloat16_t* input, __ubuf__ float8_e4m3_t* output,
                                        __ubuf__ int16_t* output_sf, __ubuf__ float* route_weights,
                                        uint32_t valid_m, float activation_clamp) {
    constexpr uint32_t kNumRowsPerIteration = 2;
    // 128 激活列 / 64 = 每行两个 scale pair；每次处理两行，共四个 64-channel vector 分组。
    constexpr uint32_t kNumPairsPerRow = (layout::BLOCK_N / 2) / MX_SF_DIVISOR;
    constexpr uint32_t kNumPairs = kNumRowsPerIteration * kNumPairsPerRow;
    static_assert(layout::BLOCK_N == 256);

    const vector_bool group_mask_b32 = asc_create_mask_b32(PAT_VL32);
    const vector_bool pair_mask_b32 = asc_create_mask_b32(PAT_VL64);
    vector_bfloat16_t zero_bf16;
    asc_duplicate_scalar(zero_bf16, 0);
    vector_int32_t arange_b32;
    vector_uint32_t high_group_idx;
    asc_arange(arange_b32, 0);
    asc_add_scalar(high_group_idx, reinterpret_cast<vector_uint32_t&>(arange_b32), MX_SF_DIVISOR / 2, group_mask_b32);
    vector_uint32_t exponent_mask, inv_scale_bias;
    asc_duplicate_scalar(exponent_mask, 0x7f800000u);
    asc_duplicate_scalar(inv_scale_bias, 0x7f000000u);  // (2 * 127) << 23

    auto gate_ptr = input;
    // gate 与 up 在同一物理输出行的前/后 128 列；每次加载后指针跨完整 N256 行。
    auto up_ptr = input + layout::BLOCK_N / 2;
    auto output_ptr = reinterpret_cast<__ubuf__ uint8_t*>(output);

    for (uint32_t m_idx = 0; m_idx < valid_m; m_idx += kNumRowsPerIteration) {
        // Load gate/up as FP32
        vector_float gate[kNumPairs], up[kNumPairs], value[kNumPairs];
        #pragma unroll
        for (uint32_t m_offset = 0; m_offset < kNumRowsPerIteration; ++ m_offset) {
            const uint32_t pair_idx = m_offset * kNumPairsPerRow;
            vector_bfloat16_t gate_bf16, up_bf16;
            asc_loadalign_postupdate(gate_bf16, gate_ptr, layout::BLOCK_N);
            asc_loadalign_postupdate(up_bf16, up_ptr, layout::BLOCK_N);
            // 将 BF16 位放到 FP32 高 16 位、低位补零，精确扩宽成两份 64-channel FP32 vector。
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(gate[pair_idx]),
                      reinterpret_cast<vector_bfloat16_t&>(gate[pair_idx + 1]), zero_bf16, gate_bf16);
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(up[pair_idx]),
                      reinterpret_cast<vector_bfloat16_t&>(up[pair_idx + 1]), zero_bf16, up_bf16);
        }

        // Load route weights; odd tails use the allocated padding row
        vector_float route_weight[kNumRowsPerIteration];
        if constexpr (kApplyRouteWeight) {
            asc_loadalign_brc_postupdate(route_weight[0], route_weights, 1);
            asc_loadalign_brc_postupdate(route_weight[1], route_weights, 1);
        }

        // Clamp gate above and up on both sides
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
            // gate 仅截上界，up 截正/负两侧；截断在 SiLU 和 route weight 之前执行。
            asc_min_scalar(gate[pair_idx], gate[pair_idx], activation_clamp, pair_mask_b32);
            asc_min_scalar(up[pair_idx], up[pair_idx], activation_clamp, pair_mask_b32);
            asc_max_scalar(up[pair_idx], up[pair_idx], -activation_clamp, pair_mask_b32);
            asc_exp_sub(value[pair_idx], reinterpret_cast<vector_float&>(zero_bf16), gate[pair_idx], pair_mask_b32);
        }

        // SwiGLU: gate * up / (1 + exp(-gate))
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
            asc_add_scalar(value[pair_idx], value[pair_idx], 1.0f, pair_mask_b32);
            asc_div(value[pair_idx], gate[pair_idx], value[pair_idx], pair_mask_b32);
            asc_mul(value[pair_idx], value[pair_idx], up[pair_idx], pair_mask_b32);
            // 每行 route weight 广播到该行所有列；scale 的 amax 基于乘权重后的激活计算。
            if constexpr (kApplyRouteWeight)
                asc_mul(value[pair_idx], value[pair_idx], route_weight[pair_idx / kNumPairsPerRow], pair_mask_b32);
        }

        // Reduce amax for both 32-channel groups in each SF pair
        #pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
            const uint32_t m_offset = pair_idx / kNumPairsPerRow;
            const uint32_t sf_pair_idx = pair_idx % kNumPairsPerRow;
            vector_uint32_t bits[2];
            asc_abs(reinterpret_cast<vector_float&>(bits[0]), value[pair_idx], pair_mask_b32);
            // 将后半 32 通道移到归约窗口，与前半 32 通道分别求 amax；一个 int16 存两组 scale。
            asc_gather(bits[1], bits[0], high_group_idx);
            #pragma unroll
            for (uint32_t group_idx = 0; group_idx < 2; ++ group_idx) {
                asc_reduce_max(bits[group_idx], bits[group_idx], group_mask_b32);
                asc_duplicate(bits[group_idx], bits[group_idx], pair_mask_b32);
            }

            // Round amax / 448 up to a power of two
            asc_select(bits[0], bits[0], bits[1], group_mask_b32);
            // 用最小 amax 避免零输入产生过小尺度；对浮点指数位操作，将 amax/448 向上取 2 的幂。
            // scale 是 2^s，inv_scale_bits 构造 2^-s，不通过逐元素除法计算逆尺度。
            asc_max_scalar(bits[0], bits[0], 0x38d1b717u, pair_mask_b32);  // FP32 bits of 1e-4
            asc_add_scalar(bits[0], bits[0], 0x001fffffu, pair_mask_b32);  // Carry when significand > 1.75
            asc_and(bits[0], bits[0], exponent_mask, pair_mask_b32);
            asc_add_scalar(reinterpret_cast<vector_int32_t&>(bits[0]),
                           reinterpret_cast<vector_int32_t&>(bits[0]), -0x04000000, pair_mask_b32);  // Subtract 8 from exponent
            asc_min_scalar(bits[0], bits[0], 0x7f000000u, pair_mask_b32);
            vector_uint32_t inv_scale_bits;
            asc_sub(inv_scale_bits, inv_scale_bias, bits[0], pair_mask_b32);
            asc_mul(value[pair_idx], value[pair_idx], reinterpret_cast<vector_float&>(inv_scale_bits), pair_mask_b32);

            // Store FP8 and UE8M0 scales
            vector_fp8_e4m3fn_t value_fp8;
            // 加权激活先乘逆 scale，再做 FP8 E4M3 round-to-nearest/saturate；不是直接把 BF16 截成 8 位。
            asc_float2e4m3_rn_sat(value_fp8, value[pair_idx], pair_mask_b32, ASC_DISPERSE_FIRST_QUARTER);
            asc_storealign_pack_quarter(reinterpret_cast<__ubuf__ uint32_t*>(output_ptr),
                                        reinterpret_cast<vector_uint32_t&>(value_fp8), pair_mask_b32);
            output_ptr += MX_SF_DIVISOR;

            asc_shiftright_scalar(bits[0], bits[0], 23, pair_mask_b32);
            asc_gather(bits[1], bits[0], high_group_idx);
            vector_uint8_t packed_sf_half, unused_sf_half;
            asc_intlv(packed_sf_half, unused_sf_half, reinterpret_cast<vector_uint8_t&>(bits[0]),
                      reinterpret_cast<vector_uint8_t&>(bits[1]));
            // scale 写成 [scale pair][STORE_BLOCK_M 行]，每对两个 E8M0 字节打包为 int16。
            // 调用方据此搬到 GM 的块内 K-major scale 区，与 GMM2 的读取布局配套。
            asc_storealign_1st(output_sf + sf_pair_idx * STORE_BLOCK_M,
                               reinterpret_cast<vector_int16_t&>(packed_sf_half), m_idx + m_offset);
        }
    }
}

// Combine: FP32 sum -> BF16; input stride is num_channels
__simd_vf__ inline void vf_combine_reduce(__ubuf__ bfloat16_t* inputs, uint32_t num_inputs, uint32_t num_channels) {
    constexpr uint32_t kNumChannelsPerVector = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);
    const auto mask_b16 = asc_create_mask_b16(PAT_ALL);
    const auto mask_b32 = asc_create_mask_b32(PAT_ALL);
    vector_bfloat16_t zero_bf16;
    asc_duplicate_scalar(zero_bf16, 0);
    auto output_ptr = inputs;

    for (uint32_t channel_idx = 0; channel_idx < num_channels; channel_idx += kNumChannelsPerVector) {
        vector_float sum0, sum1;
        asc_duplicate_scalar(sum0, 0.0f);
        asc_duplicate_scalar(sum1, 0.0f);
        auto input_ptr = inputs + channel_idx;
        // 同一 channel fragment 沿 route 输入维累加；输入已经在 GMM1 epilogue 加权，此处只有求和。
        for (uint32_t input_idx = 0; input_idx < num_inputs; ++ input_idx) {
            vector_bfloat16_t value;
            vector_float value0, value1;
            asc_loadalign_postupdate(value, input_ptr, num_channels);

            // Widen BF16 exactly with zero low bits
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(value0), reinterpret_cast<vector_bfloat16_t&>(value1), zero_bf16, value);
            asc_add(sum0, sum0, value0, mask_b32);
            asc_add(sum1, sum1, value1, mask_b32);
        }

        // Round to BF16 and pack both halves
        vector_bfloat16_t output0, output1, output, unused;
        // 全部 route（及 shared 额外槽）完成 FP32 累加后才一次舍入成 BF16；结果原地覆盖输入区前一行。
        asc_float2bfloat16_rn(output0, sum0, mask_b32, ASC_POSITION_EVEN);
        asc_float2bfloat16_rn(output1, sum1, mask_b32, ASC_POSITION_EVEN);
        asc_deintlv(output, unused, output0, output1);
        asc_storealign_postupdate(output_ptr, output, kNumChannelsPerVector, mask_b16);
    }
}

} // namespace deep_gemm::mega_moe
