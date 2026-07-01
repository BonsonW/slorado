#pragma once

// flute — thin C++ facade over the precompiled cutedsl/CUTLASS fused int8 kernels.
//
// The verbose, arch-specific kernel headers (thirdparty/flute/<arch>/*.h) are included
// only by flute.cpp. Callers (e.g. TxModel.cpp) see just this facade, so future
// quantization formats (fp8, mxfp4) and architectures (rdna4) plug in behind the same
// Backend interface without touching the model code.

#include <ATen/core/Tensor.h>
#include <memory>
#include <string>

struct tensor_quant; // defined in thirdparty/dorado/tensor_chunk_utils.h

namespace flute {

// Quantization format a layer's method string maps to at the kernel level.
enum class Format { None, Int8, Fp8, Mxfp4 };

// Map a quant_config method ("int8_per_channel", "fp8_per_tensor", ...) to a Format.
// Anything that is not a recognised real-kernel format returns Format::None.
Format parse_format(const std::string &method);

// Model dimensions a backend must match. The kernels are dimension-specialized, so the
// backend verifies these against what it was compiled for and bows out on mismatch.
struct ModelDims {
    int d_model;
    int dim_feedforward;
    int nhead;
    int head_dim;
    int max_seq;
};

// A concrete (arch, format) fused-kernel backend. The two ops take an int8 activation
// tensor_quant + int8 weight tensor_quant(s) and return an fp16 tensor. Dims are verified
// once at selection time, so these do not fall back per-call.
struct Backend {
    virtual ~Backend() = default;
    virtual const char *name() const = 0;
    virtual Format format() const = 0;

    // Fused int8 wqkv GEMM + rotary. x: int8 [N,T,d_model] (+per-token scale). wqkv: int8
    // [3*d_model, d_model] (+per-out-channel scale). sin/cos: fp32 row-major [seq, head_dim/2]
    // (rotate-half). Returns fp16 qkv [N,T,3,nhead,head_dim] with rotary applied to Q and K.
    virtual at::Tensor qkv_rotary_i8(const tensor_quant &x, const tensor_quant &wqkv,
                                     const at::Tensor &sin, const at::Tensor &cos) = 0;

    // Fused gated-MLP: dual GEMM (gate, up) + SiLU. x: int8 [N,T,d_model] (+per-token
    // scale). gate/up: int8 [H, d_model] (+per-out-channel scale). Returns fp16
    // silu(gate) * up, [N,T,H].
    virtual at::Tensor gated_mlp_i8(const tensor_quant &x,
                                    const tensor_quant &gate,
                                    const tensor_quant &up) = 0;
};

// Detect the compute capability of `device_index` and return a backend for (arch, desired)
// when the precompiled kernels match both the arch and `dims`; otherwise nullptr (caller
// keeps the fp16 path). Backends are memoized per (device, format) so each cubin module is
// loaded only once across all layers / runners.
std::shared_ptr<Backend> select_backend(int device_index, Format desired, const ModelDims &dims);

} // namespace flute
