/* lstm_model_metal.h — Metal (Apple Silicon) acceleration for the standard-activation CRF LSTM stack.
 *
 * Drives dorado's tiled simdgroup LSTM + reorder Metal kernels (see lstm_model_metal.metal) to run
 * the fast/hac v5 bidirectional-alternating LSTM stack on the GPU in one dispatch-per-layer, instead
 * of torch's per-call _lstm_mps. Only the standard sigmoid/tanh LSTM is supported (v5); the factored
 * (v6) LSTM has a clamped activation dorado's kernel does not implement.
 *
 * Buffer contract (row-major): input/output are host fp32 [T, N, C] (dorado's reorder kernels take
 * fp32 activations in and out; the LSTM compute itself is fp16). Weights are host fp16 in dorado's
 * [3*C+1, C, 4] layout. The caller (lstm_model.cpp) transposes to/from its [N, T, C] tensors, casts
 * fp16<->fp32 at the boundary, and folds the two torch LSTM biases into one.
 */
#ifndef LSTM_MODEL_METAL_H
#define LSTM_MODEL_METAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct metal_lstm_ctx metal_lstm_ctx_t;

// Compile kernels for an lstm_size-wide, num_layers-deep stack. reverse_first=1 means layer 0 scans
// time in reverse (matches slorado's flip-before-every-layer order). Returns NULL on failure.
metal_lstm_ctx_t *metal_lstm_create(int lstm_size, int num_layers, int reverse_first);

// Whether the current device/config can run on Metal (lstm_size tiles correctly, kernels compiled).
int metal_lstm_ok(const metal_lstm_ctx_t *ctx);

// Per-layer reverse flag (reverse_first alternating), so the caller applies dorado's U/W weight swap
// consistently when preparing the reordered weight buffer for that layer.
int metal_lstm_layer_reverse(const metal_lstm_ctx_t *ctx, int layer);

// Hand over one layer's already-reordered weights: host fp16, dorado layout [3*C+1, C, 4] (U|W|W|bias
// combined, gates reordered IFGO->GIFO, bias = bias_ih+bias_hh folded into the last row). Copied into
// a private MTLBuffer; the caller's tensor can be freed after.
void metal_lstm_set_layer(metal_lstm_ctx_t *ctx, int layer, const void *reordered_w_f16, size_t bytes);

// Run the full stack, zero-copy: in_mtl / out_mtl are MTLBuffers (the conv-output and result MPS
// tensors' storage().data(), fp16 [T, N, C], contiguous, storage offset 0). N must be a multiple of
// 48 (SIMD_TILES_M*TILE_SIZE) — the caller pads. The caller must sync torch's MPS stream first (the
// LSTM runs on a separate command queue). Returns 0 on success.
int metal_lstm_run(metal_lstm_ctx_t *ctx, int N, int T, const void *in_mtl, const void *out_mtl);

void metal_lstm_free(metal_lstm_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif // LSTM_MODEL_METAL_H
