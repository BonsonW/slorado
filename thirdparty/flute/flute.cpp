#include "flute.h"

// tensor_quant lives in the dorado tree; resolved via -I thirdparty.
#include "dorado/tensor_chunk_utils.h"

#include <cstdio>

namespace flute {

Format parse_format(const std::string &method) {
    if (method.rfind("int8", 0) == 0) return Format::Int8;
    // Future real-kernel formats plug in here (fp8, mxfp4). Everything else — including
    // "fp16", "dummy", empty, and the fake-quant-only methods — stays on the fp16 path.
    return Format::None;
}

} // namespace flute

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <stdexcept>

#include "sm80/gemm_i8_dual_silu_N2048_K512.h"
#include "sm80/gemm_i8_rotary_N1536_K512_H8D64R64S1024.h"

namespace flute {

namespace {

// Fill a kernel tensor descriptor (dynamic_shapes[3] / dynamic_strides[2]) from an ATen
// tensor's real sizes/strides. NOTE: the exact layout convention the precompiled kernels
// expect is the #1 correctness risk — validate against a reference before trusting output.
void fill_desc(int32_t shapes[3], int64_t strides[2], const at::Tensor &t) {
    const int nd = static_cast<int>(t.dim());
    for (int i = 0; i < 3; ++i) shapes[i]  = i < nd ? static_cast<int32_t>(t.size(i))   : 1;
    for (int i = 0; i < 2; ++i) strides[i] = i < nd ? static_cast<int64_t>(t.stride(i)) : 0;
}

} // namespace

// sm_80 (Ampere/Ada) int8 backend over the two precompiled fused kernels.
struct Sm80Int8Backend : Backend {
    int d_model_, dim_feedforward_, nhead_, head_dim_, max_seq_, device_index_;
    gemm_i8_rotary_N1536_K512_H8D64R64S1024_Kernel_Module_t rotary_module_{};
    gemm_i8_dual_silu_N2048_K512_Kernel_Module_t            mlp_module_{};

    Sm80Int8Backend(const ModelDims &d, int device_index)
        : d_model_(d.d_model), dim_feedforward_(d.dim_feedforward),
          nhead_(d.nhead), head_dim_(d.head_dim), max_seq_(d.max_seq), device_index_(device_index) {
        // The kernels are dimension-specialized (baked into the cubin symbols).
        if (d.d_model != 512 || d.dim_feedforward != 2048 || d.nhead != 8 || d.head_dim != 64) {
            throw std::runtime_error("flute sm80 int8: model dims do not match precompiled kernels "
                                     "(need d_model=512, dim_feedforward=2048, nhead=8, head_dim=64)");
        }
        gemm_i8_rotary_N1536_K512_H8D64R64S1024_Kernel_Module_Load(&rotary_module_);
        gemm_i8_dual_silu_N2048_K512_Kernel_Module_Load(&mlp_module_);
    }

    ~Sm80Int8Backend() override {
        gemm_i8_rotary_N1536_K512_H8D64R64S1024_Kernel_Module_Unload(&rotary_module_);
        gemm_i8_dual_silu_N2048_K512_Kernel_Module_Unload(&mlp_module_);
    }

    const char *name() const override { return "sm80-int8"; }
    Format format() const override { return Format::Int8; }

    at::Tensor qkv_rotary_i8(const tensor_quant &x, const tensor_quant &wqkv,
                             const at::Tensor &sin, const at::Tensor &cos) override {
        const int64_t N = x.tensor.size(0);
        const int64_t T = x.tensor.size(1);
        const int64_t M = N * T;
        // Kernels expect 2D [M,K] memrefs with innermost stride 1.
        auto a2d = x.tensor.reshape({M, d_model_});
        auto out = torch::empty({M, 3 * d_model_}, x.tensor.options().dtype(at::kHalf));

        gemm_i8_rotary_N1536_K512_H8D64R64S1024_Tensor_mA_t mA{};
        mA.data = a2d.data_ptr();
        fill_desc(mA.dynamic_shapes, mA.dynamic_strides, a2d);

        gemm_i8_rotary_N1536_K512_H8D64R64S1024_Tensor_mB_t mB{};
        mB.data = wqkv.tensor.data_ptr();
        fill_desc(mB.dynamic_shapes, mB.dynamic_strides, wqkv.tensor);

        gemm_i8_rotary_N1536_K512_H8D64R64S1024_Tensor_mC_t mC{};
        mC.data = out.data_ptr();
        fill_desc(mC.dynamic_shapes, mC.dynamic_strides, out);

        gemm_i8_rotary_N1536_K512_H8D64R64S1024_Tensor_mScaleA_t mScaleA{ x.scale.data_ptr() };
        gemm_i8_rotary_N1536_K512_H8D64R64S1024_Tensor_mScaleB_t mScaleB{ wqkv.scale.data_ptr() };
        gemm_i8_rotary_N1536_K512_H8D64R64S1024_Tensor_mSin_t    mSin{ (void *)sin.data_ptr() };
        gemm_i8_rotary_N1536_K512_H8D64R64S1024_Tensor_mCos_t    mCos{ (void *)cos.data_ptr() };

        cute_dsl_gemm_i8_rotary_N1536_K512_H8D64R64S1024_wrapper(
            &rotary_module_, &mA, &mB, &mC, &mScaleA, &mScaleB, &mSin, &mCos);

        return out.view({N, T, 3, nhead_, head_dim_});
    }

    at::Tensor gated_mlp_i8(const tensor_quant &x, const tensor_quant &gate,
                            const tensor_quant &up) override {
        const int64_t N = x.tensor.size(0);
        const int64_t T = x.tensor.size(1);
        const int64_t M = N * T;
        // Kernels expect 2D [M,K] memrefs with innermost stride 1.
        auto a2d = x.tensor.reshape({M, d_model_});
        auto out = torch::empty({M, dim_feedforward_}, x.tensor.options().dtype(at::kHalf));

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
            &mlp_module_, &mA, &mB_gate, &mB_up, &mC, &mScaleA, &mScaleB_gate, &mScaleB_up);

        return out.view({N, T, dim_feedforward_});
    }
};

std::shared_ptr<Backend> select_backend(int device_index, Format desired, const ModelDims &dims) {
    if (desired != Format::Int8) return nullptr; // only int8 kernels exist today

    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_index) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_index) != cudaSuccess) {
        return nullptr;
    }
    const int cc = major * 10 + minor;
    // sm_80 cubins run on Ampere (80/86) and Ada (89). Widen after testing.
    const bool sm80_family = (cc == 80 || cc == 86 || cc == 89);
    if (!sm80_family) {
        fprintf(stderr, "[flute] no int8 backend for compute capability %d.%d — using fp16 path\n",
                major, minor);
        return nullptr;
    }

    // Memoize per (device, format) so each cubin module loads once across all layers/runners.
    static std::mutex mtx;
    static std::map<std::pair<int, int>, std::weak_ptr<Backend>> cache;
    std::lock_guard<std::mutex> lk(mtx);
    auto key = std::make_pair(device_index, static_cast<int>(desired));
    if (auto it = cache.find(key); it != cache.end()) {
        if (auto sp = it->second.lock()) return sp;
    }
    std::shared_ptr<Backend> backend;
    try {
        backend = std::make_shared<Sm80Int8Backend>(dims, device_index);
    } catch (const std::exception &e) {
        fprintf(stderr, "[flute] %s — using fp16 path\n", e.what());
        return nullptr;
    }
    cache[key] = backend;
    fprintf(stderr, "[flute] int8 kernel backend '%s' active on device %d (sm_%d)\n",
            backend->name(), device_index, cc);
    return backend;
}

} // namespace flute

#else // !HAVE_CUDA

namespace flute {
std::shared_ptr<Backend> select_backend(int, Format, const ModelDims &) { return nullptr; }
} // namespace flute

#endif // HAVE_CUDA
