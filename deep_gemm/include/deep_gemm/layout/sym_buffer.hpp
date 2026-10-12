#pragma once

#include <cstdint>

#include <deep_gemm/common.hpp>

namespace deep_gemm::layout {

// Communication kernels encode rank indices in 8 bits
constexpr uint32_t kNumMaxRanks = 1u << 8;

template <uint32_t kNumRanks = kNumMaxRanks>
struct SymBuffer {
    int64_t base;
    uint32_t rank_idx;
    // Keep rank and the active offsets near base in the launch data
    int64_t offsets[kNumMaxRanks];

    static_assert(kNumRanks <= kNumMaxRanks);
    SymBuffer() = default;

    template <typename Container>
    explicit SymBuffer(const Container& c, uint32_t rank_idx): base(c[rank_idx]), rank_idx(rank_idx) {
        for (uint32_t i = 0; i < kNumMaxRanks; ++ i)
            // peer_base - local_base 为 VA 差值；各 rank workspace 布局相同，所以任一区域同 offset 都可平移映射。
            offsets[i] = i < c.size() ? c[i] - base : 0;
    }

#if defined(__CCE__) || defined(__CLION_IDE__)
    template <typename ptr_t = __gm__ void*>
    constexpr __aicore__ __attribute__((always_inline)) ptr_t get_base_ptr() const {
        return __builtin_bit_cast(ptr_t, static_cast<uint64_t>(base));
    }

    // SIMT uses a UB copy of the rank offsets
    template <typename ptr_t, typename offset_ptr_t>
    static constexpr __aicore__ __attribute__((always_inline)) ptr_t map(ptr_t ptr, uint32_t dst_rank_idx, offset_ptr_t offsets) {
        if constexpr (kNumRanks == 1)
            return ptr;
        // 此处只做地址平移，不执行通信或同步；后续 Scalar/MTE 指令才访问远端已注册内存。
        const uint64_t address = __builtin_bit_cast(uint64_t, ptr) + offsets[dst_rank_idx];
        return __builtin_bit_cast(ptr_t, address);
    }

    template <typename ptr_t>
    constexpr __aicore__ __attribute__((always_inline)) ptr_t map(ptr_t ptr, uint32_t dst_rank_idx) const {
        return map(ptr, dst_rank_idx, offsets);
    }
#endif
};

} // namespace deep_gemm::layout
