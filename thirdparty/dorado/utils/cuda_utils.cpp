#include "cuda_utils.h"

#include "../nn/CRFModel.h"
#include "torch/torch.h"

extern "C" {
#include "koi.h"
}

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <limits>
#include <regex>
#include <string>
#include <vector>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <numeric>
#include <type_traits>

void handle_cuda_result(int cuda_result) {
    if (cuda_result == cudaSuccess)
        return;

    if (cuda_result == cudaErrorNoKernelImageForDevice) {
        throw std::runtime_error(
                std::string("Dorado cannot support the CUDA device being used,"
                            " as the compute capability version is incompatible."));
    } else {
        throw std::runtime_error(std::string("Cuda error: ") +
                                 cudaGetErrorString(cudaError_t(cuda_result)));
    }
}

void matmul_f16_cublas(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor &C) {
    constexpr uint16_t HALF_ZERO = 0;      // 0.0 in __half format
    constexpr uint16_t HALF_ONE = 0x3C00;  // 1.0 in __half format
    assert(A.dtype() == torch::kF16 && B.dtype() == torch::kF16 && C.dtype() == torch::kF16);
    assert(A.stride(1) == 1 && B.stride(1) == 1 && C.stride(1) == 1);
    assert(A.size(0) == C.size(0));  // M
    assert(B.size(1) == C.size(1));  // N
    assert(A.size(1) == B.size(0));  // K
    auto res =
            cublasGemmEx(at::cuda::getCurrentCUDABlasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, B.size(1),
                         A.size(0), A.size(1), &HALF_ONE, B.data_ptr(), CUDA_R_16F, B.stride(0),
                         A.data_ptr(), CUDA_R_16F, A.stride(0), &HALF_ZERO, C.data_ptr(),
                         CUDA_R_16F, C.stride(0), CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (res != CUBLAS_STATUS_SUCCESS) {
        // spdlog::error("CuBLAS error {}", int(res));
        exit(EXIT_FAILURE);
    }
}

void matmul_f16_torch(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor &C) {
    C.copy_(torch::matmul(A, B));
}

void matmul_f16(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor &C) {
    // torch::matmul() is a bit slower than cublasGemmEx() on A100 and half the speed on V100,
    // but an order of magnitude faster on our Windows CI machines (1080 Ti), so dynamically
    // pick which one we should use on first invocation.
    // static auto const fastest_mat_mul = [] {
    //     CUDATimer cuda_timer;

    //     // Arbitrary tensor lengths to benchmark against.
    //     // Note: even with sizes this small it still takes ~2s to benchmark cuBLAS on a 1080 Ti.
    //     const int L = 2048;
    //     const int M = 192;
    //     const int N = 384;

    //     auto options = torch::TensorOptions().dtype(torch::kFloat16).device(c10::kCUDA);
    //     auto a = torch::empty({L, M}, options);
    //     auto b = torch::empty({M, N}, options);
    //     auto c = torch::empty({L, N}, options);

    //     auto run_N_times = [&](auto matmul_impl) {
    //         const size_t N = 1000;
    //         // Warmup then profile
    //         for (size_t i = 0; i < 10; i++) {
    //             matmul_impl(a, b, c);
    //         }
    //         cuda_timer.start();
    //         for (size_t i = 0; i < N; i++) {
    //             matmul_impl(a, b, c);
    //         }
    //         cuda_timer.stop();
    //         return cuda_timer.result_ms();
    //     };

    //     float const torch_time = run_N_times(matmul_f16_torch);
    //     float const cublas_time = run_N_times(matmul_f16_cublas);
    //     return cublas_time < torch_time ? matmul_f16_cublas : matmul_f16_torch;
    // }();
    matmul_f16_cublas(A, B, C);
}

template <typename T>
T div_round_closest(const T n, const T d) {
    return ((n < 0) ^ (d < 0)) ? ((n - d / 2) / d) : ((n + d / 2) / d);
}
template <typename T>
T pad_to(const T a, const T b) {
    return ((a + b - 1) / b) * b;
}

// Adapted from https://stackoverflow.com/questions/11964552/finding-quartiles
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value, T>::type>
inline std::vector<T> quantiles(const std::vector<T>& in_data, const std::vector<T>& quants) {
    if (in_data.empty()) {
        return {};
    }

    if (in_data.size() == 1) {
        return {in_data.front()};
    }

    auto data = in_data;
    std::sort(std::begin(data), std::end(data));
    std::vector<T> quantiles;
    quantiles.reserve(quants.size());

    auto linear_interp = [](T v0, T v1, T t) { return (1 - t) * v0 + t * v1; };

    for (size_t i = 0; i < quants.size(); ++i) {
        T pos = linear_interp(0, T(data.size() - 1), quants[i]);

        int64_t left = std::max(int64_t(std::floor(pos)), int64_t(0));
        int64_t right = std::min(int64_t(std::ceil(pos)), int64_t(data.size() - 1));
        T data_left = data.at(left);
        T data_right = data.at(right);

        T quantile = linear_interp(data_left, data_right, pos - left);
        quantiles.push_back(quantile);
    }

    return quantiles;
}

// Perform a least-squares linear regression of the form y = mx + b, solving for m and b.
// Returns a tuple {m, b, r} where r is the regression correlation coefficient
// Adapted from https://stackoverflow.com/questions/5083465/fast-efficient-least-squares-fit-algorithm-in-c
template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value, T>::type>
std::tuple<T, T, T> linear_regression(const std::vector<T>& x, const std::vector<T>& y) {
    assert(x.size() == y.size());
    auto sum_square = [](auto s2, auto q) { return s2 + q * q; };

    T sumx2 = std::accumulate(std::begin(x), std::end(x), T(0), sum_square);
    T sumy2 = std::accumulate(std::begin(y), std::end(y), T(0), sum_square);
    T sumx = std::accumulate(std::begin(x), std::end(x), T(0));
    T sumy = std::accumulate(std::begin(y), std::end(y), T(0));

    T sumxy = 0.0;
    size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        sumxy += x[i] * y[i];
    }

    T denom = (n * sumx2 - (sumx * sumx));
    if (denom == 0) {
        // singular matrix. can't solve the problem, return identity transform
        return {T(1), T(0), T(0)};
    }

    T m = (n * sumxy - sumx * sumy) / denom;
    T b = (sumy * sumx2 - sumx * sumxy) / denom;
    // compute correlation coeff
    T r = (sumxy - sumx * sumy / n) /
          std::sqrt((sumx2 - (sumx * sumx) / n) * (sumy2 - (sumy * sumy) / n));

    return {m, b, r};
}

template <typename T>
bool eq_with_tolerance(T a, T b, T tol) {
    return std::abs(a - b) <= tol;
}
