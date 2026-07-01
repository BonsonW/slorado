#pragma once

// fluke — thin facade over the precompiled cutedsl/CUTLASS fused int8 kernels.
//
// The verbose, arch-specific kernel headers (thirdparty/fluke/<arch>/*.h) are included only by
// fluke.cpp. Callers (e.g. TxModel.cpp) see just this facade. Plain-C style: an opaque backend
// handle + free functions, no classes/inheritance/smart pointers. The at:: tensor types are kept
// because they simplify the ops far more than hand-rolled shape/stride plumbing would.

#include <ATen/core/Tensor.h>
#include <string>

struct tensor_quant; // defined in thirdparty/dorado/tensor_chunk_utils.h

// Quantization format a layer's method string maps to at the kernel level.
enum fluke_format { FLUKE_FORMAT_NONE, FLUKE_FORMAT_INT8, FLUKE_FORMAT_FP8, FLUKE_FORMAT_MXFP4 };

// Model dimensions a backend must match. The kernels are dimension-specialized, so the backend
// verifies these against what it was compiled for and bows out on mismatch.
struct fluke_dims { int d_model, dim_feedforward, nhead, head_dim, max_seq; };

// Opaque, process-lifetime backend handle. Callers keep the pointer but do NOT own/free it.
typedef struct fluke_backend fluke_backend;

// Map a quant mode / method ("int8", "int8_per_channel", ...) to a format. Anything not backed by a
// real kernel returns FLUKE_FORMAT_NONE.
enum fluke_format fluke_parse_format(const std::string &method);

// Detect the compute capability of `device_index` and return a backend when the precompiled kernels
// match both the arch and `dims` for `desired`; otherwise NULL (caller keeps the fp16 path). The
// returned handle is shared across all callers and lives for the process; do not free it. The cubin
// modules are loaded only once.
fluke_backend *fluke_select_backend(int device_index, enum fluke_format desired, struct fluke_dims dims);

// Fused int8 wqkv GEMM + rotary. x: int8 [N,T,d_model] (+per-token scale). wqkv: int8
// [3*d_model, d_model] (+per-out-channel scale). sin/cos: fp32 row-major [seq, head_dim/2]
// (rotate-half). Returns fp16 qkv [N,T,3,nhead,head_dim] with rotary applied to Q and K.
at::Tensor fluke_qkv_rotary_i8(const fluke_backend *b, const tensor_quant &x, const tensor_quant &wqkv,
                               const at::Tensor &sin, const at::Tensor &cos);

// Fused int8 dual GEMM (gate, up) + SiLU. x: int8 [N,T,d_model] (+per-token scale). gate/up: int8
// [dim_feedforward, d_model] (+per-out-channel scale). Returns fp16 silu(gate)*up [N,T,dim_feedforward].
at::Tensor fluke_gated_mlp_i8(const fluke_backend *b, const tensor_quant &x, const tensor_quant &gate,
                              const tensor_quant &up);
