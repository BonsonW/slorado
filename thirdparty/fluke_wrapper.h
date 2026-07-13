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

// fluke.h only defines its opaque backend/recurrence types on CUDA/ROCm builds (the fused
// kernels are GPU-only). On other builds (CPU, Metal/MPS) forward-declare them so slorado's
// struct fields and the wrapper's stub declarations below still name a type; they are only ever
// held as null pointers and never dereferenced (all fluke call sites are HAVE_CUDA/HAVE_ROCM gated).
#if !defined(HAVE_CUDA) && !defined(HAVE_ROCM)
typedef struct fluke_int8_backend fluke_int8_backend_t;
typedef struct fluke_flstm_backend fluke_flstm_backend_t;
typedef struct fluke_flstm_rec fluke_flstm_rec_t;
#endif

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

// ── factored-LSTM (v6 hac) int8 path ────────────────────────────────────────────────────────────
// The backend (fluke_flstm_backend_t) and recurrence-state (fluke_flstm_rec_t) handles are fluke's
// own opaque C types — slorado stores and passes them directly (no extra wrapper handle). The rec
// lifecycle is fluke's C ABI too (fluke_flstm_rec_create/free); only the ops that take at::Tensors
// need a bridge here.

// Return an FLSTM backend when `desired` is int8 and fluke has a precompiled kernel matching this
// device's arch and shape (H hidden, K_hh recurrent rank, R input rank); otherwise NULL (caller
// keeps the fp16 path). The handle is shared for the process and must not be freed.
fluke_flstm_backend_t *fluke_select_flstm(int device_index, enum fluke_format_t desired, int H, int K_hh, int R);

// int8 down-projection into a caller-provided fp16 `out` [M, R] = (a_i8[M,H] * scale_a[M]) @
// (w.tensor[R,H] * w.scale[R])^T. Used for the ih precompute (M = T*N); the per-step hh projection
// is now internal to the recurrence. out and a_i8 give M; w gives R.
void fluke_flstm_down_proj_i8_into(const fluke_flstm_backend_t *b, at::Tensor &out, const at::Tensor &a_i8,
                                   const at::Tensor &scale_a, const tensor_quant_t &w);

// Fused dequantize + transpose: in int8 [T, N, C] (scale) -> out fp16 [N, T, C], out[n,t,c] =
// in[t,n,c] * scale. Used to convert the last FLSTM layer's int8 hidden ring to fp16 in one pass.
at::Tensor fluke_dequant_int8_transpose(const at::Tensor &in_tnc, float scale);

// ── Unified recurrence (fluke owns the loop + CUDA graph + fused/two-kernel choice) ──────────────
// Create/free the recurrence state via fluke's C ABI directly: fluke_flstm_rec_create(backend, N, T,
// num_layers) / fluke_flstm_rec_free(rec) (declared in <fluke/fluke.h>). Only the run below needs a
// bridge (it takes at::Tensors). Run one layer's full T-step recurrence (zeroes boundary hidden +
// cell, loops, captures/replays a CUDA graph, chooses fused vs two-kernel internally):
//   hh_all [T+1,N,C] int8 ring; cell [N,C] f32 (in place); x_down [T,N,K] f16 (precomputed);
//   w_dn [K_hh,H] int8 + comb_scale [K_hh] f32; gate_w[g] [H,Kc] f16, gate_b[g] [H] f32 (i,f,g,o);
//   reverse = scan direction (ring parity).
void fluke_flstm_run_recurrence(fluke_flstm_rec_t *rec, int layer_idx,
                                at::Tensor &hh_all, at::Tensor &cell, const at::Tensor &x_down,
                                const at::Tensor &w_dn, const at::Tensor &comb_scale,
                                const at::Tensor gate_w[4], const at::Tensor gate_b[4], bool reverse);
