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

// Opaque, process-lifetime backend handle. Callers keep the pointer but do NOT own/free it.
typedef struct fluke_backend fluke_backend_t;

// Map a quant mode / method ("int8", "int8_per_channel", ...) to a format. Anything not backed by
// a real kernel returns FLUKE_FORMAT_NONE.
enum fluke_format_t fluke_parse_format(const std::string &method);

// Return a backend when `desired` is int8 and fluke has a precompiled kernel matching this
// device's arch and `dims`; otherwise NULL (caller keeps the fp16 path). The handle is shared
// for the process and must not be freed; the kernel modules load only once.
fluke_backend_t *fluke_select_backend(int device_index, enum fluke_format_t desired, fluke_dims_t dims);

// Fused int8 wqkv GEMM + rotary. x: int8 [N,T,d_model] (+per-token scale). wqkv: int8
// [3*d_model, d_model] (+per-out-channel scale). sin/cos: fp32 row-major [seq, head_dim/2]
// (rotate-half). Returns fp16 qkv [N,T,3,nhead,head_dim] with rotary applied to Q and K.
at::Tensor fluke_qkv_rotary_i8(const fluke_backend_t *b, const tensor_quant_t &x, const tensor_quant_t &wqkv,
                               const at::Tensor &sin, const at::Tensor &cos);

// Fused int8 dual GEMM (gate, up) + SiLU. x: int8 [N,T,d_model] (+per-token scale). gate/up: int8
// [dim_feedforward, d_model] (+per-out-channel scale). Returns fp16 silu(gate)*up [N,T,dim_feedforward].
at::Tensor fluke_gated_mlp_i8(const fluke_backend_t *b, const tensor_quant_t &x, const tensor_quant_t &gate,
                              const tensor_quant_t &up);

// ── factored-LSTM (v6 hac) int8 path ────────────────────────────────────────────────────────────
// Opaque, process-lifetime FLSTM backend handle (wraps fluke_flstm_backend_t). Do NOT free.
typedef struct fluke_flstm_wrap fluke_flstm_wrap_t;

// Return an FLSTM backend when `desired` is int8 and fluke has a precompiled kernel matching this
// device's arch and shape (H hidden, K_hh recurrent rank, R input rank); otherwise NULL (caller
// keeps the fp16 path). The handle is shared for the process and must not be freed.
fluke_flstm_wrap_t *fluke_select_flstm(int device_index, enum fluke_format_t desired, int H, int K_hh, int R);

// int8 down-projection: returns fp16 [M, R] = (a_i8[M,H] * scale_a[M]) @ (w.tensor[R,H] * w.scale[R])^T.
// Used for both the ih precompute (M = T*N) and the per-step hh projection (M = N).
at::Tensor fluke_flstm_down_proj_i8(const fluke_flstm_wrap_t *b, const at::Tensor &a_i8,
                                    const at::Tensor &scale_a, const tensor_quant_t &w);

// Same as fluke_flstm_down_proj_i8 but writes into the caller-provided fp16 `out` [M, R] instead of
// allocating — lets the recurrence reuse persistent buffers (no per-step allocation). out and a_i8
// give M; w gives R.
void fluke_flstm_down_proj_i8_into(const fluke_flstm_wrap_t *b, at::Tensor &out, const at::Tensor &a_i8,
                                   const at::Tensor &scale_a, const tensor_quant_t &w);

// Fused dequantize + transpose: in int8 [T, N, C] (scale) -> out fp16 [N, T, C], out[n,t,c] =
// in[t,n,c] * scale. Used to convert the last FLSTM layer's int8 hidden ring to fp16 in one pass.
at::Tensor fluke_dequant_int8_transpose(const at::Tensor &in_tnc, float scale);

// Standalone per-token int8 quantize (GPU analogue of quantize_tensor(x, -1)): fp16 [M, C] ->
// {int8 [M, C], f32 scale [M]} with scale = amax/128 (dequant multiplier). C must be even and <= 2048.
tensor_quant_t fluke_quant_int8(const at::Tensor &x);

// Fused int8 FLSTM step. Writes h_i8 [B,H] int8 (fixed scale 1/127) and updates c_f32 [B,H] fp32
// in place. a_f16 [B, K_hh+R] fp16 = concat(hh_down | x_down_t). gate_w[g] [H, K_hh+R] fp16,
// gate_b[g] [H] fp32, gate order i,f,g,o.
void fluke_flstm_step_i8(const fluke_flstm_wrap_t *b, at::Tensor &h_i8, at::Tensor &c_f32,
                         const at::Tensor &a_f16, const at::Tensor gate_w[4], const at::Tensor gate_b[4]);

// Single-launch fused step: does the recurrent hh int8 down-projection AND the gate step in one
// kernel (no hh_down/a_scratch round-trip). Writes h_i8 [B,H] int8 (1/127) and updates c_f32 [B,H]
// in place. h_prev_i8 [B,H] int8 (previous hidden, 1/127); w_dn [K_hh,H] int8 recurrent down-weight;
// comb_scale [K_hh] f32 = w_dn per-channel scale * 1/127 (host-folded); x_f16 [B,R] this step's
// x_down; gate_w[g] [H,K_hh+R] fp16, gate_b[g] [H] fp32 (order i,f,g,o); hh_stage f16 [B,K_hh]
// scratch (producer-written); flags int32 [ceil(B/64)*4] zeroed at allocation (self-cleaning).
// Valid only for B <= 512 (grid co-residency); caller must fall back to the two-kernel path above.
void fluke_flstm_fused_step_i8(const fluke_flstm_wrap_t *b, at::Tensor &h_i8, at::Tensor &c_f32,
                               const at::Tensor &h_prev_i8, const at::Tensor &w_dn,
                               const at::Tensor &comb_scale, const at::Tensor &x_f16,
                               const at::Tensor gate_w[4], const at::Tensor gate_b[4],
                               at::Tensor &hh_stage, at::Tensor &flags);
