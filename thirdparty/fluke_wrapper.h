#pragma once

// fluke_wrapper — thin ATen facade over the fluke library's fused-int8 C ABI.
//
// The fluke submodule (<fluke/fluke.h> + libfluke.a) owns the kernels, the arch dispatch,
// the module loading, and the descriptor plumbing. This wrapper only bridges ATen: it turns
// tensor_quant_t / at::Tensor into device pointers + dims and calls fluke_qkv_rotary_i8_gpu /
// fluke_gated_mlp_i8_gpu. Plain-C style: an opaque backend handle + free functions.

#include <fluke/fluke.h>          // fluke_dims_t, fluke_int8_backend_t, the fused C ABI
#include <ATen/core/Tensor.h>
#include <string>

struct tensor_quant_t; // defined in thirdparty/dorado/tensor_chunk_utils.h

// Quantization format a layer's method string maps to at the kernel level.
enum fluke_format_t { FLUKE_FORMAT_NONE, FLUKE_FORMAT_INT8, FLUKE_FORMAT_FP8, FLUKE_FORMAT_MXFP4 };

// fluke_dims_t is provided by <fluke/fluke.h>.

// The backend handle is fluke's own opaque C type (fluke_int8_backend_t) — slorado stores and
// passes it directly; only the ops that take at::Tensors need a bridge here. Dims come from fluke
// (fluke_int8_dims) so nothing is cached slorado-side.

// Map a quant mode / method ("int8", "int8_per_channel", ...) to a format. Anything not backed by
// a real kernel returns FLUKE_FORMAT_NONE.
enum fluke_format_t fluke_parse_format(const std::string &method);

// Return a backend when `desired` is int8 and fluke has a precompiled kernel matching this
// device's arch and `dims`; otherwise NULL (caller keeps the fp16 path). The handle is shared
// for the process and must not be freed; the kernel modules load only once.
fluke_int8_backend_t *fluke_select_backend(int device_index, enum fluke_format_t desired, fluke_dims_t dims);

// Fused int8 wqkv GEMM + rotary. x: int8 [N,T,d_model] (+per-token scale). wqkv: int8
// [3*d_model, d_model] (+per-out-channel scale). sin/cos: fp32 row-major [seq, head_dim/2]
// (rotate-half). Returns fp16 qkv [N,T,3,nhead,head_dim] with rotary applied to Q and K.
at::Tensor fluke_qkv_rotary_i8(const fluke_int8_backend_t *b, const tensor_quant_t &x, const tensor_quant_t &wqkv,
                               const at::Tensor &sin, const at::Tensor &cos);

// Fused int8 dual GEMM (gate, up) + SiLU. x: int8 [N,T,d_model] (+per-token scale). gate/up: int8
// [dim_feedforward, d_model] (+per-out-channel scale). Returns fp16 silu(gate)*up [N,T,dim_feedforward].
at::Tensor fluke_gated_mlp_i8(const fluke_int8_backend_t *b, const tensor_quant_t &x, const tensor_quant_t &gate,
                              const tensor_quant_t &up);
