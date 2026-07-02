#include "fluke.h"

// tensor_quant_t lives in the dorado tree; resolved via -I thirdparty.
#include "dorado/tensor_chunk_utils.h"

#include <cstdio>

enum fluke_format_t fluke_parse_format(const std::string &method) {
    if (method.rfind("int8", 0) == 0) return FLUKE_FORMAT_INT8;
    // Future real-kernel formats plug in here (fp8, mxfp4). Everything else — including "fp16",
    // "dummy", empty, and the fake-quant-only methods — stays on the fp16 path.
    return FLUKE_FORMAT_NONE;
}

#ifdef HAVE_CUDA
#include <cuda_runtime.h>  // defines CUDART_VERSION
#endif

// The precompiled kernels use CUDA 12's library-management API (cudaLibrary_t,
// cudaLibraryLoadData, cudaLaunchKernelEx). On older CUDA toolkits (or non-CUDA builds) the fused
// int8 path is unavailable and fluke compiles to no-op stubs — the model falls back to fp16.
#if defined(HAVE_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12000

#include "sm80/gemm_i8_dual_silu_N2048_K512.h"
#include "sm80/gemm_i8_rotary_N1536_K512_H8D64R64S2048.h"

// The backend is just config: dims + format. The kernel modules are process-global (below) and
// loaded once, so a single shared handle serves every layer and device.
struct fluke_backend {
    int d_model, dim_feedforward, nhead, head_dim, max_seq;
    enum fluke_format_t format;
};

// Process-global kernel modules, loaded once by fluke_select_backend.
static gemm_i8_rotary_N1536_K512_H8D64R64S2048_Kernel_Module_t g_rotary_module;
static gemm_i8_dual_silu_N2048_K512_Kernel_Module_t            g_mlp_module;
static int g_modules_loaded = 0;

// Fill a kernel tensor descriptor (dynamic_shapes[3] / dynamic_strides[2]) from an ATen tensor's
// real sizes/strides. The kernels expect 2D [M,K] memrefs with innermost stride 1.
static void fill_desc(int32_t shapes[3], int64_t strides[2], const at::Tensor &t) {
    const int nd = (int)t.dim();
    for (int i = 0; i < 3; ++i) shapes[i]  = i < nd ? (int32_t)t.size(i)   : 1;
    for (int i = 0; i < 2; ++i) strides[i] = i < nd ? (int64_t)t.stride(i) : 0;
}

fluke_backend_t *fluke_select_backend(int device_index, enum fluke_format_t desired, fluke_dims_t dims) {
    if (desired != FLUKE_FORMAT_INT8) return NULL; // only int8 kernels exist today

    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_index) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_index) != cudaSuccess) {
        return NULL;
    }
    const int cc = major * 10 + minor;
    // sm_80 cubins run on Ampere (80/86) and Ada (89). Widen after testing.
    if (cc != 80 && cc != 86 && cc != 89) {
        fprintf(stderr, "[fluke] no int8 backend for compute capability %d.%d — using fp16\n", major, minor);
        return NULL;
    }
    // The kernels are dimension-specialized (baked into the cubin symbols).
    if (dims.d_model != 512 || dims.dim_feedforward != 2048 || dims.nhead != 8 || dims.head_dim != 64) {
        fprintf(stderr, "[fluke] model dims do not match precompiled kernels "
                        "(need d_model=512, dim_feedforward=2048, nhead=8, head_dim=64) — using fp16\n");
        return NULL;
    }

    if (!g_modules_loaded) {
        gemm_i8_rotary_N1536_K512_H8D64R64S2048_Kernel_Module_Load(&g_rotary_module);
        gemm_i8_dual_silu_N2048_K512_Kernel_Module_Load(&g_mlp_module);
        g_modules_loaded = 1;
        fprintf(stderr, "[fluke] int8 kernel backend active on device %d (sm_%d)\n", device_index, cc);
    }

    static fluke_backend_t b; // process-lifetime; all layers share it
    b.d_model = dims.d_model;
    b.dim_feedforward = dims.dim_feedforward;
    b.nhead = dims.nhead;
    b.head_dim = dims.head_dim;
    b.max_seq = dims.max_seq;
    b.format = desired;
    return &b;
}

at::Tensor fluke_qkv_rotary_i8(const fluke_backend_t *b, const tensor_quant_t &x, const tensor_quant_t &wqkv,
                               const at::Tensor &sin, const at::Tensor &cos) {
    const int64_t N = x.tensor.size(0);
    const int64_t T = x.tensor.size(1);
    const int64_t M = N * T;
    auto a2d = x.tensor.reshape({M, b->d_model});
    auto out = torch::empty({M, 3 * b->d_model}, x.tensor.options().dtype(at::kHalf));

    gemm_i8_rotary_N1536_K512_H8D64R64S2048_Tensor_mA_t mA{};
    mA.data = a2d.data_ptr();
    fill_desc(mA.dynamic_shapes, mA.dynamic_strides, a2d);

    gemm_i8_rotary_N1536_K512_H8D64R64S2048_Tensor_mB_t mB{};
    mB.data = wqkv.tensor.data_ptr();
    fill_desc(mB.dynamic_shapes, mB.dynamic_strides, wqkv.tensor);

    gemm_i8_rotary_N1536_K512_H8D64R64S2048_Tensor_mC_t mC{};
    mC.data = out.data_ptr();
    fill_desc(mC.dynamic_shapes, mC.dynamic_strides, out);

    gemm_i8_rotary_N1536_K512_H8D64R64S2048_Tensor_mScaleA_t mScaleA{ x.scale.data_ptr() };
    gemm_i8_rotary_N1536_K512_H8D64R64S2048_Tensor_mScaleB_t mScaleB{ wqkv.scale.data_ptr() };
    gemm_i8_rotary_N1536_K512_H8D64R64S2048_Tensor_mSin_t    mSin{ (void *)sin.data_ptr() };
    gemm_i8_rotary_N1536_K512_H8D64R64S2048_Tensor_mCos_t    mCos{ (void *)cos.data_ptr() };

    // Runtime seqlen = T: the kernel indexes rotary as seq = row % seqlen, supporting any
    // T in [1, baked max_seq] (the sin/cos table extent) without a per-length re-export.
    cute_dsl_gemm_i8_rotary_N1536_K512_H8D64R64S2048_wrapper(
        &g_rotary_module, &mA, &mB, &mC, &mScaleA, &mScaleB, &mSin, &mCos, (int32_t)T);

    return out.view({N, T, 3, b->nhead, b->head_dim});
}

at::Tensor fluke_gated_mlp_i8(const fluke_backend_t *b, const tensor_quant_t &x, const tensor_quant_t &gate,
                              const tensor_quant_t &up) {
    const int64_t N = x.tensor.size(0);
    const int64_t T = x.tensor.size(1);
    const int64_t M = N * T;
    auto a2d = x.tensor.reshape({M, b->d_model});
    auto out = torch::empty({M, b->dim_feedforward}, x.tensor.options().dtype(at::kHalf));

    gemm_i8_dual_silu_N2048_K512_Tensor_mA_t mA{};
    mA.data = a2d.data_ptr();
    fill_desc(mA.dynamic_shapes, mA.dynamic_strides, a2d);

    gemm_i8_dual_silu_N2048_K512_Tensor_mB_gate_t mB_gate{};
    mB_gate.data = gate.tensor.data_ptr();
    fill_desc(mB_gate.dynamic_shapes, mB_gate.dynamic_strides, gate.tensor);

    gemm_i8_dual_silu_N2048_K512_Tensor_mB_up_t mB_up{};
    mB_up.data = up.tensor.data_ptr();
    fill_desc(mB_up.dynamic_shapes, mB_up.dynamic_strides, up.tensor);

    gemm_i8_dual_silu_N2048_K512_Tensor_mC_t mC{};
    mC.data = out.data_ptr();
    fill_desc(mC.dynamic_shapes, mC.dynamic_strides, out);

    gemm_i8_dual_silu_N2048_K512_Tensor_mScaleA_t      mScaleA{ x.scale.data_ptr() };
    gemm_i8_dual_silu_N2048_K512_Tensor_mScaleB_gate_t mScaleB_gate{ gate.scale.data_ptr() };
    gemm_i8_dual_silu_N2048_K512_Tensor_mScaleB_up_t   mScaleB_up{ up.scale.data_ptr() };

    cute_dsl_gemm_i8_dual_silu_N2048_K512_wrapper(
        &g_mlp_module, &mA, &mB_gate, &mB_up, &mC, &mScaleA, &mScaleB_gate, &mScaleB_up);

    return out.view({N, T, b->dim_feedforward});
}

#else // no CUDA-12 kernel support — no backend, ops never called.

fluke_backend_t *fluke_select_backend(int, enum fluke_format_t, fluke_dims_t) { return NULL; }

at::Tensor fluke_qkv_rotary_i8(const fluke_backend_t *, const tensor_quant_t &, const tensor_quant_t &,
                               const at::Tensor &, const at::Tensor &) { return at::Tensor(); }

at::Tensor fluke_gated_mlp_i8(const fluke_backend_t *, const tensor_quant_t &, const tensor_quant_t &,
                              const tensor_quant_t &) { return at::Tensor(); }

#endif // HAVE_CUDA
