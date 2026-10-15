#pragma once

#include <deep_gemm/ascend.hpp>
#include <deep_gemm/layout/mega_moe.hpp>

#include <c_api/asc_simd.h>

namespace deep_gemm::mega_moe {

// FP4 -> FP8 NZ, scaled by 2^-6; weight SFs compensate with +6
// Rows map to GEMM N; columns map to K
// 模板行对应GEMM输出N，模板列对应归约K；函数只负责FP4字节展开与NZ排布，不读取scale。
template <uint32_t kNumRows, uint32_t kNumCols>
__simd_vf__ inline void vf_fp4_to_fp8_nz(__ubuf__ uint8_t* src, __ubuf__ uint8_t* dst) {
    // FP4两个数打包在一字节，原始每行字节数是逻辑K的一半。
    constexpr uint32_t kNumPackedCols = kNumCols / 2;
    // NZ scratch 在 GEMM N 行方向多留一行作 stride padding；真实复制到 L1 的仍是 kNumRows 行。
    constexpr uint32_t kNumPaddedRows = kNumRows + 1;
    // 一个完整int8 vector展开后每原字节变两字节，因此每vector消费寄存器字节数/2的packed输入。
    constexpr uint32_t kNumPackedColsPerVector = sizeof(vector_int8_t) / 2;  // Two FP4 values per byte
    // 一行按完整向量处理所需的vector数；后面每次并行组织4行。
    constexpr uint32_t kNumVectorsPerRow = kNumPackedCols / kNumPackedColsPerVector;
    // 要求packed行宽为向量批次整数倍，当前kernel支持的K分块满足此条件。
    static_assert(kNumPackedCols % kNumPackedColsPerVector == 0);

    // 本转换每个vector都处理满宽，不依赖token有效M；被转换的是完整权重tile。
    const auto mask = asc_create_mask_b8(PAT_ALL);
    // 三个向量常量分别控制右移、交替左移和最终FP8位筛选。
    vector_int8_t shift_right, shift_left, fp8_mask;
    // 有符号右移2位，将FP4 sign/exponent/mantissa排列到目标FP8字节的位置。
    asc_duplicate_scalar(shift_right, static_cast<int8_t>(2));
    // 每个 FP4 byte 展开成两份后，通过交替移位抽取高/低 nibble；mask 保留 sign 和指数/尾数位。
    asc_duplicate_scalar(reinterpret_cast<vector_uint16_t&>(shift_left), uint16_t{4});  // Alternating byte shifts: 4, 0
    // 0x9c保留缩放后FP8编码的符号/有效位；这不是饱和值或量化scale。
    asc_duplicate_scalar(fp8_mask, static_cast<int8_t>(0x9c));  // Sign and shifted FP4 payload

    // 后继load使用int8解释位模式并自动推进输入指针，不改变原packed存储。
    auto src_ptr = reinterpret_cast<__ubuf__ int8_t*>(src);
    // 每次处理4个GEMM N行，匹配NZ交织写回模式。
    constexpr uint32_t kNumRowsPerPass = 4;
    // 行数必须能按4行完整展开，当前N128/N256均满足。
    static_assert(kNumRows % kNumRowsPerPass == 0);
    // 本次四行合计的vector数量，编译期数组尺寸随MAD形状特化。
    constexpr uint32_t kNumVectors = kNumVectorsPerRow * kNumRowsPerPass;

    // Each vector writes one row fragment in NZ order; the extra row pads the stride
    // 每个向量拥有自己的NZ输出片段指针，写回后各自自动前进至下一四行组。
    __ubuf__ uint8_t* dst_ptrs[kNumVectors];
    #pragma unroll
    // 按vector编号分别初始化K片段与四行内行位置；循环在编译期展开。
    for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx)
        // 将线性 vector 编号拆成 K-fragment 和四行中的 row，直接写成 AIC 可消费的 NZ 排列。
        dst_ptrs[vector_idx] = dst + vector_idx % kNumVectorsPerRow * kNumPaddedRows * 2 * kNumPackedColsPerVector +
                               // 行内位置乘NZ fractal K跨度，与前一行的K片段stride分开计算。
                               vector_idx / kNumVectorsPerRow * get_frac_k<uint8_t>();

    // 沿N轴每4行推进，K片段的写地址由各vector指针维护。
    for (uint32_t row_idx = 0; row_idx < kNumRows; row_idx += kNumRowsPerPass) {
        // Expand packed nibbles into scaled FP8 values
        // 寄存器数组暂存当前四行展开结果，避免用额外GM/UB中间布局。
        vector_int8_t values[kNumVectors];
        #pragma unroll
        // 对每个行/K片段独立load与位转换，展开循环减少动态控制开销。
        for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx) {
            // 一字节两个 FP4，经 upsample/位运算展开为两字节 FP8；数值缩放 2^-6 由预处理后的权重 scale 补偿。
            asc_loadalign_upsample_postupdate(values[vector_idx], src_ptr, kNumPackedColsPerVector);
            // 交替移位使同一原byte的高低nibble在展开副本中对齐。
            asc_shiftleft(values[vector_idx], values[vector_idx], shift_left, mask);
            // 再将对齐后的有效FP4位右移到FP8编码位置。
            asc_shiftright(values[vector_idx], values[vector_idx], shift_right, mask);
            // 过滤非目标位，结果为带2^-6缩放的FP8可计算权重字节。
            asc_and(values[vector_idx], values[vector_idx], fp8_mask, mask);
        }

        // Scatter into padded NZ and advance by four rows
        #pragma unroll
        // 把转换后的每vector直接散写到NZ，而不是先输出ND再二次转换。
        for (uint32_t vector_idx = 0; vector_idx < kNumVectors; ++ vector_idx)
            asc_storealign_postupdate(dst_ptrs[vector_idx], reinterpret_cast<vector_uint8_t&>(values[vector_idx]),
                                      // padded行数作为NZ分块stride，四行作为本轮指针推进量。
                                      kNumPaddedRows, kNumRowsPerPass, mask);
    }
}

// 输入每行 [gate128 | up128]，输出 128 列 FP8，scale pair 覆盖两个 32-channel 组。
// routed 路径先做 SwiGLU、乘路由权重，再计算量化 scale；shared 路径不乘 route weight。
// 将权重移到最终 combine 再乘会改变中间量化误差，不能视为数值等价的重排。
// Linear1 epilogue: SwiGLU and FP8 quantization
// kApplyRouteWeight为编译期开关：routed=true，共享路径=false；STORE_BLOCK_M固定本UB stage行stride。
template <bool kApplyRouteWeight, uint32_t STORE_BLOCK_M>
__simd_vf__ inline void vf_swiglu_quant(__ubuf__ bfloat16_t* input, __ubuf__ float8_e4m3_t* output,
                                        // output_sf为每对两个E8M0字节的int16存储；route_weights为当前行组的FP32权重。
                                        __ubuf__ int16_t* output_sf, __ubuf__ float* route_weights,
                                        // valid_m是本次最多32行的小步处理数，clamp先作用gate/up，再执行SwiGLU和量化。
                                        uint32_t valid_m, float activation_clamp) {
    // 每轮向量函数处理两行；尾奇数行使用已分配padding，写回由外层有效M截断。
    constexpr uint32_t kNumRowsPerIteration = 2;
    // 128 激活列 / 64 = 每行两个 scale pair；每次处理两行，共四个 64-channel vector 分组。
    constexpr uint32_t kNumPairsPerRow = (layout::BLOCK_N / 2) / MX_SF_DIVISOR;
    // 每两行×每行2对scale，共4个64-channel vector分组。
    constexpr uint32_t kNumPairs = kNumRowsPerIteration * kNumPairsPerRow;
    // gate/up各128列的布局依赖N256，若修改BLOCK_N必须同步重写此VF布局。
    static_assert(layout::BLOCK_N == 256);

    // group_mask 覆盖每个64通道向量的前32元素，pair_mask 覆盖全64；后半通道先 gather 到前32再独立归约。
    const vector_bool group_mask_b32 = asc_create_mask_b32(PAT_VL32);
    // mask64控制整个scale pair对应的64通道；mask32只在该pair内的单个量化组上归约。
    const vector_bool pair_mask_b32 = asc_create_mask_b32(PAT_VL64);
    // BF16零向量用于扩宽时补FP32低16位，同时提供exp(0-gate)中的零。
    vector_bfloat16_t zero_bf16;
    // 零位模式可复用为BF16/FP32零，避免每个pair重复构造。
    asc_duplicate_scalar(zero_bf16, 0);
    // 整型arange用于构建固定gather索引，不是token位置索引。
    vector_int32_t arange_b32;
    // 后半32通道的gather下标表。
    vector_uint32_t high_group_idx;
    // 构造0,1,2,...的lane序号。
    asc_arange(arange_b32, 0);
    // 前32个lane加32，令gather把后半组搬到前半归约窗口。
    asc_add_scalar(high_group_idx, reinterpret_cast<vector_uint32_t&>(arange_b32), MX_SF_DIVISOR / 2, group_mask_b32);
    // exponent_mask用于剥离amax尾数，inv_scale_bias用于整数位算术构造倒数幂。
    vector_uint32_t exponent_mask, inv_scale_bias;
    // 0x7f800000为FP32 exponent位掩码，清掉sign与mantissa。
    asc_duplicate_scalar(exponent_mask, 0x7f800000u);
    // 254<<23与纯指数scale位相减，得到对应的倒数2幂FP32位模式。
    asc_duplicate_scalar(inv_scale_bias, 0x7f000000u);  // (2 * 127) << 23

    // 第一行gate起点在本物理N256行的前128列。
    auto gate_ptr = input;
    // gate 与 up 在同一物理输出行的前/后 128 列；每次加载后指针跨完整 N256 行。
    auto up_ptr = input + layout::BLOCK_N / 2;
    // FP8输出按1B元素紧凑推进，可位于同一BF16 UB stage低地址。
    auto output_ptr = reinterpret_cast<__ubuf__ uint8_t*>(output);

    // 两行批处理，奇数尾行仍从已分配的 padding 行读；调用方 GM store 只提交 stage_valid_m，不写 padding。
    for (uint32_t m_idx = 0; m_idx < valid_m; m_idx += kNumRowsPerIteration) {
        // Load gate/up as FP32
        // gate/up/value四组vector分别容纳当前两行的两份64通道片段。
        vector_float gate[kNumPairs], up[kNumPairs], value[kNumPairs];
        #pragma unroll
        // 先加载本轮两行的完整gate/up，之后再做任何原地输出，避免压紧写覆盖未读取输入。
        for (uint32_t m_offset = 0; m_offset < kNumRowsPerIteration; ++ m_offset) {
            // 将本轮行偏移映射到寄存器数组的第0或第2个pair。
            const uint32_t pair_idx = m_offset * kNumPairsPerRow;
            // 暂存该行各128列gate/up的BF16加载结果。
            vector_bfloat16_t gate_bf16, up_bf16;
            // 加载gate128列后跨完整N256元素行stride，跳到下一行gate。
            asc_loadalign_postupdate(gate_bf16, gate_ptr, layout::BLOCK_N);
            // up指针独立跨同样N256 stride，保持与gate逐行配对。
            asc_loadalign_postupdate(up_bf16, up_ptr, layout::BLOCK_N);
            // 将 BF16 位放到 FP32 高 16 位、低位补零，精确扩宽成两份 64-channel FP32 vector。
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(gate[pair_idx]),
                      reinterpret_cast<vector_bfloat16_t&>(gate[pair_idx + 1]), zero_bf16, gate_bf16);
            // up采用与gate同样的位交织扩宽，每行得到两组64列FP32。
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(up[pair_idx]),
                      reinterpret_cast<vector_bfloat16_t&>(up[pair_idx + 1]), zero_bf16, up_bf16);
        }

        // Load route weights; odd tails use the allocated padding row
        // 每输入行一份广播路由权重，供该行所有pair共享。
        vector_float route_weight[kNumRowsPerIteration];
        if constexpr (kApplyRouteWeight) {
            // 广播第0行weight并将FP32源指针推进一元素。
            asc_loadalign_brc_postupdate(route_weight[0], route_weights, 1);
            // 广播第1行weight；奇数尾行读取的padding权重不写出为有效GM行。
            asc_loadalign_brc_postupdate(route_weight[1], route_weights, 1);
        }

        // Clamp gate above and up on both sides
        #pragma unroll
        // 四个pair分别执行同一截断/exp操作，输入已经是FP32。
        for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
            // gate 仅截上界，up 截正/负两侧；截断在 SiLU 和 route weight 之前执行。
            asc_min_scalar(gate[pair_idx], gate[pair_idx], activation_clamp, pair_mask_b32);
            // up正侧超过clamp的数截到clamp。
            asc_min_scalar(up[pair_idx], up[pair_idx], activation_clamp, pair_mask_b32);
            // up负侧小于-clamp的数截到-clamp，gate不执行此对称截下界。
            asc_max_scalar(up[pair_idx], up[pair_idx], -activation_clamp, pair_mask_b32);
            // 计算exp(-gate)，作为SiLU分母的指数项。
            asc_exp_sub(value[pair_idx], reinterpret_cast<vector_float&>(zero_bf16), gate[pair_idx], pair_mask_b32);
        }

        // SwiGLU: gate * up / (1 + exp(-gate))
        #pragma unroll
        // 逐pair形成SwiGLU并可选施加该行route weight；顺序影响后继量化误差。
        for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
            // exp(-gate)+1构成sigmoid分母。
            asc_add_scalar(value[pair_idx], value[pair_idx], 1.0f, pair_mask_b32);
            // gate除以分母得到SiLU(gate)。
            asc_div(value[pair_idx], gate[pair_idx], value[pair_idx], pair_mask_b32);
            // SiLU(gate)*up得到未量化的中间激活。
            asc_mul(value[pair_idx], value[pair_idx], up[pair_idx], pair_mask_b32);
            // 每行 route weight 广播到该行所有列；scale 的 amax 基于乘权重后的激活计算。
            if constexpr (kApplyRouteWeight)
                // 通过pair_idx/2找对应行weight，保证同一行两组64列使用相同路由权重。
                asc_mul(value[pair_idx], value[pair_idx], route_weight[pair_idx / kNumPairsPerRow], pair_mask_b32);
        }

        // Reduce amax for both 32-channel groups in each SF pair
        #pragma unroll
        // 对每个64通道pair各计算两份32通道amax，再量化并写两字节scale。
        for (uint32_t pair_idx = 0; pair_idx < kNumPairs; ++ pair_idx) {
            // 当前pair所属的两行之一，用于最后scale写入行号。
            const uint32_t m_offset = pair_idx / kNumPairsPerRow;
            // pair_idx / 2 选本次两行中的行，pair_idx % 2 选该行的前/后64激活列；
            // 因此 scale row/column 的写入位置与压紧 FP8 列顺序完全对应。
            const uint32_t sf_pair_idx = pair_idx % kNumPairsPerRow;
            // 保存当前pair的两个量化组amax/指数位中间结果。
            vector_uint32_t bits[2];
            // 先取加权激活绝对值；后续针对非负FP32位模式做比较/指数运算。
            asc_abs(reinterpret_cast<vector_float&>(bits[0]), value[pair_idx], pair_mask_b32);
            // 将后半 32 通道移到归约窗口，与前半 32 通道分别求 amax；一个 int16 存两组 scale。
            asc_gather(bits[1], bits[0], high_group_idx);
            #pragma unroll
            // 两个32通道组独立归约，不能把整64列共用单个amax。
            for (uint32_t group_idx = 0; group_idx < 2; ++ group_idx) {
                // 在mask32范围求组内最大绝对值。
                asc_reduce_max(bits[group_idx], bits[group_idx], group_mask_b32);
                // 将组amax广播以便后续按lane形成对应scale。
                asc_duplicate(bits[group_idx], bits[group_idx], pair_mask_b32);
            }

            // Round amax / 448 up to a power of two
            // 把两个32通道组的 amax 广播拼回同一64通道向量，各半使用自己的量化scale，而非整个128列共用scale。
            asc_select(bits[0], bits[0], bits[1], group_mask_b32);
            // 用最小 amax 避免零输入产生过小尺度；对浮点指数位操作，将 amax/448 向上取 2 的幂。
            // scale 是 2^s，inv_scale_bits 构造 2^-s，不通过逐元素除法计算逆尺度。
            asc_max_scalar(bits[0], bits[0], 0x38d1b717u, pair_mask_b32);  // FP32 bits of 1e-4
            // 加整数偏置使amax significand超过1.75时向下一指数进位，对应E4M3最大448的scale选择。
            asc_add_scalar(bits[0], bits[0], 0x001fffffu, pair_mask_b32);  // Carry when significand > 1.75
            // 只留下指数，丢掉浮点amax的mantissa，使scale成为2的整数幂。
            asc_and(bits[0], bits[0], exponent_mask, pair_mask_b32);
            asc_add_scalar(reinterpret_cast<vector_int32_t&>(bits[0]),
                           // 从amax指数减8，等效除256；配合前一步1.75进位近似amax/448的上取幂。
                           reinterpret_cast<vector_int32_t&>(bits[0]), -0x04000000, pair_mask_b32);  // Subtract 8 from exponent
            // 把scale的最高指数限制为254，避免生成FP32无穷/NaN exponent255。
            asc_min_scalar(bits[0], bits[0], 0x7f000000u, pair_mask_b32);
            // 暂存倒数scale的FP32原始位，不在此转成整数数值。
            vector_uint32_t inv_scale_bits;
            // scale 保留纯指数位，对 FP32 exponent 做 254-exponent 的位算术构造其倒数，再乘加权激活。
            // 后面右移23位取 scale exponent，存为 E8M0，不保存 FP32 尾数。
            asc_sub(inv_scale_bits, inv_scale_bias, bits[0], pair_mask_b32);
            // 加权激活乘2^-s归一化到FP8量化范围，后面再舍入/饱和转换。
            asc_mul(value[pair_idx], value[pair_idx], reinterpret_cast<vector_float&>(inv_scale_bits), pair_mask_b32);

            // Store FP8 and UE8M0 scales
            // E4M3向量寄存器承接量化结果，只有指定quarter布局存有效FP8字节。
            vector_fp8_e4m3fn_t value_fp8;
            // 加权激活先乘逆 scale，再做 FP8 E4M3 round-to-nearest/saturate；不是直接把 BF16 截成 8 位。
            asc_float2e4m3_rn_sat(value_fp8, value[pair_idx], pair_mask_b32, ASC_DISPERSE_FIRST_QUARTER);
            // 将有效quarter打包写成连续64字节，去除转换指令产生的稀疏寄存器布局。
            asc_storealign_pack_quarter(reinterpret_cast<__ubuf__ uint32_t*>(output_ptr),
                                        reinterpret_cast<vector_uint32_t&>(value_fp8), pair_mask_b32);
            // 当前64列输出占64字节；每次 pair 都推进，因此两个pair组成当前行128字节FP8，之后才进入下一行。
            output_ptr += MX_SF_DIVISOR;

            // 将纯FP32 exponent位右移23，得到UE8M0要保存的8位指数值。
            asc_shiftright_scalar(bits[0], bits[0], 23, pair_mask_b32);
            // 同时取得后半32通道组的指数，准备与前半组打包。
            asc_gather(bits[1], bits[0], high_group_idx);
            // 两份byte交织结果中只取首份有效scale pair，另一份是未使用布局副产物。
            vector_uint8_t packed_sf_half, unused_sf_half;
            // 两个E8M0 exponent字节交织为每行一个int16 scale pair。
            asc_intlv(packed_sf_half, unused_sf_half, reinterpret_cast<vector_uint8_t&>(bits[0]),
                      reinterpret_cast<vector_uint8_t&>(bits[1]));
            // scale 写成 [scale pair][STORE_BLOCK_M 行]，每对两个 E8M0 字节打包为 int16。
            // 调用方据此搬到 GM 的块内 K-major scale 区，与 GMM2 的读取布局配套。
            asc_storealign_1st(output_sf + sf_pair_idx * STORE_BLOCK_M,
                               // 写入当前pair列与当前有效行；scale pair之间跨度STORE_BLOCK_M，不是激活行宽128。
                               reinterpret_cast<vector_int16_t&>(packed_sf_half), m_idx + m_offset);
        }
    }
}

// Combine: FP32 sum -> BF16; input stride is num_channels
// 归并函数读取[num_inputs,num_channels]的BF16紧凑矩阵，输出覆盖第0行；hidden输入已经在前面加权。
__simd_vf__ inline void vf_combine_reduce(__ubuf__ bfloat16_t* inputs, uint32_t num_inputs, uint32_t num_channels) {
    // 每次处理一个完整BF16向量宽度，对应两份FP32向量。
    constexpr uint32_t kNumChannelsPerVector = sizeof(vector_bfloat16_t) / sizeof(bfloat16_t);
    // BF16整vector掩码用于最终连续写回。
    const auto mask_b16 = asc_create_mask_b16(PAT_ALL);
    // FP32整vector掩码用于扩宽后的两份累加。
    const auto mask_b32 = asc_create_mask_b32(PAT_ALL);
    // 零BF16位提供FP32扩宽所需低16位。
    vector_bfloat16_t zero_bf16;
    asc_duplicate_scalar(zero_bf16, 0);
    // 原地输出从inputs起点开始，后续不再需要已经求和的原输入值。
    auto output_ptr = inputs;

    // 沿hidden/channel轴逐向量处理，每次对所有route的同一片列求和。
    for (uint32_t channel_idx = 0; channel_idx < num_channels; channel_idx += kNumChannelsPerVector) {
        // 两个FP32累加器分别对应一个BF16向量的两半。
        vector_float sum0, sum1;
        // 第一半FP32累加器初始化为0，支持num_inputs=0输出零。
        asc_duplicate_scalar(sum0, 0.0f);
        // 第二半也清零，避免沿不同channel fragment复用旧寄存器值。
        asc_duplicate_scalar(sum1, 0.0f);
        // 当前channel起点；每次load后按num_channels跳到下一route同一片列。
        auto input_ptr = inputs + channel_idx;
        // 同一 channel fragment 沿 route 输入维累加；输入已经在 GMM1 epilogue 加权，此处只有求和。
        for (uint32_t input_idx = 0; input_idx < num_inputs; ++ input_idx) {
            // 单条route的原始BF16向量。
            vector_bfloat16_t value;
            // 扩宽后的两份FP32向量，保持BF16的精确值。
            vector_float value0, value1;
            // 读取当前route的channel fragment，并将指针推进到下一route。
            asc_loadalign_postupdate(value, input_ptr, num_channels);

            // Widen BF16 exactly with zero low bits
            asc_intlv(reinterpret_cast<vector_bfloat16_t&>(value0), reinterpret_cast<vector_bfloat16_t&>(value1), zero_bf16, value);
            // 将当前route前半向量累加到FP32 sum0。
            asc_add(sum0, sum0, value0, mask_b32);
            // 后半向量累加到FP32 sum1；无逐route BF16中间舍入。
            asc_add(sum1, sum1, value1, mask_b32);
        }

        // Round to BF16 and pack both halves
        // 准备两份舍入后的BF16临时结果和最终交织打包结果。
        vector_bfloat16_t output0, output1, output, unused;
        // 全部 route（及 shared 额外槽）完成 FP32 累加后才一次舍入成 BF16；结果原地覆盖输入区前一行。
        asc_float2bfloat16_rn(output0, sum0, mask_b32, ASC_POSITION_EVEN);
        // 将第二半FP32累加一次round-to-nearest为BF16。
        asc_float2bfloat16_rn(output1, sum1, mask_b32, ASC_POSITION_EVEN);
        // 剥离空白/无效lane并合并两半成一个连续BF16向量。
        asc_deintlv(output, unused, output0, output1);
        // 存当前channel向量并自动推进output_ptr；外层保证channel数量向量对齐。
        asc_storealign_postupdate(output_ptr, output, kNumChannelsPerVector, mask_b16);
    }
}

} // namespace deep_gemm::mega_moe
