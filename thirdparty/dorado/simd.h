#pragma once

// TSan's init breaks the call to __cpu_indicator_init (which determines which implementation to take)
#if defined(__GNUC__) && defined(__x86_64__) && !defined(__clang__) && !defined(__SANITIZE_THREAD__)
#define ENABLE_AVX2_IMPL 1
#include <immintrin.h>
#else
#define ENABLE_AVX2_IMPL 0
#endif

// GCC 5 can fail to emit function multiversion dispatchers for f16c targets.
// Keep AVX2 includes enabled, but disable target("avx2,f16c") overloads there.
#if ENABLE_AVX2_IMPL && defined(__GNUC__) && !defined(__clang__) && (__GNUC__ < 6)
#define ENABLE_AVX2_FMV 0
#else
#define ENABLE_AVX2_FMV ENABLE_AVX2_IMPL
#endif

#if defined(__arm64__) || defined(__aarch64__)
#define ENABLE_NEON_IMPL 1
#include <arm_neon.h>
#else
#define ENABLE_NEON_IMPL 0
#endif

#include <cstddef>
#include <cstring>
#include <torch/torch.h>

#if ENABLE_NEON_IMPL

// Neon registers have 4 floats.
static constexpr size_t kFloatsPerRegister = 4;

using FloatRegister = float32x4_t;
using HalfRegister = float16x4_t;
using Int16Register = int16x4_t;

#define simd_load_f32(ptr) vld1q_f32(ptr)
#define simd_load1_f32(ptr) vdupq_n_f32(*(ptr))
#define simd_convert_f32_f16(reg) vcvt_f16_f32(reg)
#define simd_store_f32(ptr, reg) vst1q_f32(ptr, reg)
#define simd_store1_f32(ptr, reg) *(ptr) = vgetq_lane_f32(reg, 0)

#define simd_load_f16(ptr) vld1_f16(reinterpret_cast<float16_t const *>(ptr))
#define simd_load1_f16(ptr) vdup_n_f16(*(ptr))
#define simd_convert_f16_f32(reg) vcvt_f32_f16(reg)
#define simd_add_f16(regA, regB) vadd_f16(regA, regB)
#define simd_store_f16(ptr, reg) vst1_f16(reinterpret_cast<float16_t *>(ptr), reg)
#define simd_store1_f16(ptr, reg) *(ptr) = vget_lane_f16(reg, 0)

#define simd_load_i16(ptr) vreinterpret_s16_f16(simd_load_f16(ptr))
#define simd_convert_i16_f32(reg) vcvtq_f32_s32(vmovl_s16(reg))

#elif ENABLE_AVX2_IMPL

// AVX registers have 8 floats.
static constexpr size_t kFloatsPerRegister = 8;

// Matches torch behaviour.
static constexpr int kRoundNearestEven = 0;

using FloatRegister = __m256;
using HalfRegister = __m128i;
using Int16Register = __m128i;

#define simd_load_f32(ptr) _mm256_loadu_ps(ptr)
#define simd_load1_f32(ptr) _mm256_broadcast_ss(ptr)
#define simd_convert_f32_f16(reg) _mm256_cvtps_ph(reg, kRoundNearestEven)

#define simd_load_f16(ptr) _mm_loadu_si128(reinterpret_cast<const __m128i *>(ptr))
#define simd_store_f16(ptr, reg) _mm_storeu_si128(reinterpret_cast<__m128i *>(ptr), reg)
#define simd_store1_f16(ptr, reg) \
    *(ptr) = c10::Half(_mm_extract_epi16(reg, 0), c10::Half::from_bits())

#define simd_load_i16(ptr) simd_load_f16(ptr)
#define simd_convert_i16_f32(reg) _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(reg))

#endif

#if !ENABLE_NEON_IMPL
#if ENABLE_AVX2_FMV
[[maybe_unused]] __attribute__((target("default")))
#endif
inline void shift_scale_tensor_i16_to_f16_inplace_impl(at::Tensor& tensor, float shift, float scale) {
    tensor = tensor.to(at::ScalarType::Float).sub_(shift).div_(scale).to(at::ScalarType::Half);
}

#if ENABLE_AVX2_FMV
[[maybe_unused]] __attribute__((target("default")))
#endif
inline void scale_shift_tensor_i16_to_f16_inplace_impl(at::Tensor& tensor, float shift, float scale) {
    tensor = tensor.to(at::ScalarType::Float).mul_(scale).add_(shift).to(at::ScalarType::Half);
}
#endif  // !ENABLE_NEON_IMPL

#if ENABLE_AVX2_FMV || ENABLE_NEON_IMPL
#if ENABLE_AVX2_FMV
[[maybe_unused]] __attribute__((target("avx2,f16c")))
#endif
inline void shift_scale_tensor_i16_to_f16_inplace_impl(at::Tensor& tensor, float shift, float scale) {
#if ENABLE_AVX2_FMV
    constexpr std::size_t kUnrollFactor = 4;
#else
    constexpr std::size_t kUnrollFactor = 1;
#endif

    constexpr std::size_t elem_size = 2;
    constexpr std::size_t elems_per_block = kFloatsPerRegister * kUnrollFactor;

    std::int16_t* data = tensor.data_ptr<std::int16_t>();
    const std::size_t size = tensor.numel();

    const FloatRegister shift_f32 = simd_load1_f32(&shift);
    const FloatRegister scale_f32 = simd_load1_f32(&scale);
    for (std::size_t block = 0; block < size / elems_per_block; block++) {
        for (std::size_t unroll_i = 0; unroll_i < kUnrollFactor; unroll_i++) {
            const Int16Register data_i16 = simd_load_i16(data);
            const FloatRegister data_f32 = simd_convert_i16_f32(data_i16);
            const FloatRegister scaled_f32 = (data_f32 - shift_f32) / scale_f32;
            const HalfRegister scaled_f16 = simd_convert_f32_f16(scaled_f32);
            simd_store_f16(data, scaled_f16);
            data += kFloatsPerRegister;
        }
    }

    std::int16_t block_in[elems_per_block];
    c10::Half block_out[elems_per_block];
    const std::size_t remaining = size % elems_per_block;
    std::memcpy(block_in, data, remaining * elem_size);
    for (std::size_t i = 0; i < remaining; i++) {
        const float val = static_cast<float>(block_in[i]);
        block_out[i] = (val - shift) / scale;
    }
    std::memcpy(data, block_out, remaining * elem_size);

    tensor = tensor.view(at::ScalarType::Half);
}

#if ENABLE_AVX2_FMV
[[maybe_unused]] __attribute__((target("avx2,f16c")))
#endif
inline void scale_shift_tensor_i16_to_f16_inplace_impl(at::Tensor& tensor, float shift, float scale) {
#if ENABLE_AVX2_FMV
    constexpr std::size_t kUnrollFactor = 4;
#else
    constexpr std::size_t kUnrollFactor = 1;
#endif

    constexpr std::size_t elem_size = 2;
    constexpr std::size_t elems_per_block = kFloatsPerRegister * kUnrollFactor;

    std::int16_t* data = tensor.data_ptr<std::int16_t>();
    const std::size_t size = tensor.numel();

    const FloatRegister shift_f32 = simd_load1_f32(&shift);
    const FloatRegister scale_f32 = simd_load1_f32(&scale);
    for (std::size_t block = 0; block < size / elems_per_block; block++) {
        for (std::size_t unroll_i = 0; unroll_i < kUnrollFactor; unroll_i++) {
            const Int16Register data_i16 = simd_load_i16(data);
            const FloatRegister data_f32 = simd_convert_i16_f32(data_i16);
            const FloatRegister scaled_f32 = (data_f32 * scale_f32) + shift_f32;
            const HalfRegister scaled_f16 = simd_convert_f32_f16(scaled_f32);
            simd_store_f16(data, scaled_f16);
            data += kFloatsPerRegister;
        }
    }

    std::int16_t block_in[elems_per_block];
    c10::Half block_out[elems_per_block];
    const std::size_t remaining = size % elems_per_block;
    std::memcpy(block_in, data, remaining * elem_size);
    for (std::size_t i = 0; i < remaining; i++) {
        const float val = static_cast<float>(block_in[i]);
        block_out[i] = (val * scale) + shift;
    }
    std::memcpy(data, block_out, remaining * elem_size);

    tensor = tensor.view(at::ScalarType::Half);
}
#endif  // ENABLE_AVX2_FMV || ENABLE_NEON_IMPL
