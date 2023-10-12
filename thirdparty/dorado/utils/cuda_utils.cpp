#include "cuda_utils.h"

#include "../../../src/globals.h"
#include "../../../src/misc.h"

#include "../nn/CRFModel.h"
#include "torch/torch.h"



extern "C" {
#include "koi.h"
//#include "winograd.cu"
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

// void winograd_mm(torch::Tensor const&,torch::Tensor const&,torch::Tensor&);
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

void matmul_f16_cublas(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor &C) {
    // std::cout << "\nmatMul\n" << std::endl; //Test
    matMul ++;
    // matMul -= realtime();
    constexpr uint16_t HALF_ZERO = 0;      // 0.0 in __half format
    constexpr uint16_t HALF_ONE = 0x3C00;  // 1.0 in __half format

    // std::cout << "\nA: " << A.sizes() << std::endl; // Test
    // std::cout << "\nB: " << B.sizes() << std::endl; // Test
    // std::cout << "\nC: " << C.sizes() << std::endl; // Test

        
    // assertT -= realtime();
    assert(A.dtype() == torch::kF16 && B.dtype() == torch::kF16 && C.dtype() == torch::kF16);
    assert(A.stride(1) == 1 && B.stride(1) == 1 && C.stride(1) == 1);
    // assert(A.size(0) == C.size(0));  // M
    // assert(B.size(1) == C.size(1));  // N
    // assert(A.size(1) == B.size(0));  // K
    // assertT += realtime();

    cublasGemmExT -= realtime();

    auto res =
            cublasGemmEx(at::cuda::getCurrentCUDABlasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, B.size(1),
                         A.size(0), A.size(1), &HALF_ONE, B.data_ptr(), CUDA_R_16F, B.stride(0),
                         A.data_ptr(), CUDA_R_16F, A.stride(0), &HALF_ZERO, C.data_ptr(),
                         CUDA_R_16F, C.stride(0), CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                         
    cublasGemmExT += realtime();
    if (res != CUBLAS_STATUS_SUCCESS) {
        // spdlog::error("CuBLAS error {}", int(res));
        exit(EXIT_FAILURE);
    }
    
    
    // matMul += realtime();

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
    // matmul_f16_cublas(A, B, C);
    // void winograd_mm(torch::Tensor const  &A,torch::Tensor const &B,torch::Tensor &C){
    // winograd_mm(A, B, C);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////











