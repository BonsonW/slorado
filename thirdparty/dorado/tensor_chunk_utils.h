/* @file signal_prep.h
**
** methods for preparing a signal for the base calling step
** @@
******************************************************************************/

#ifndef SIGNAL_PREP_H
#define SIGNAL_PREP_H

#include "torchbox.h"

template <typename T>
T div_round_up(const T a, const T b) {
    return (a + b - 1) / b;
}
template <typename T>
T pad_to(const T a, const T b) {
    return div_round_up(a, b) * b;
}

// Symmetric int8 quantization container. `tensor` is the int8 data; `scale` is the
// per-slice dequant multiplier (reciprocal pre-applied) — i.e. fp ≈ tensor * scale.
struct tensor_quant {
    at::Tensor tensor; // int8 tensor
    at::Tensor scale;  // float scale per slice, reciprocal pre-applied
};

// Quantize `x` to symmetric int8 with one scale per slice along `dim`.
//   dim = -1 on activations [..., C]      -> one scale per token
//   dim =  1 on a weight    [out, in]     -> one scale per output channel
// The returned `scale` is the dequant multiplier (amax/128), matching the kernels'
// mScaleA/mScaleB and the fused RMSNorm's residual_scale conventions.
inline tensor_quant quantize_tensor(const at::Tensor &x, int dim) {
    auto fp_range = x.abs().amax(dim);
    constexpr int i_range = 256 / 2;
    auto quant_scale = (i_range / fp_range);
    auto quant_max = i_range - 1;
    auto x_quant = (x * quant_scale.unsqueeze(dim)).round().clip(-quant_max, quant_max);
    return tensor_quant {
        x_quant.to(torch::kInt8).contiguous(),
        quant_scale.to(torch::kFloat32).reciprocal_().contiguous()
    };
}

void scale_signal(core_t *core, torch::Tensor &signal, float scaling, float offset, SignalNormalisationParams &scaling_params);

// Scale a single record's signal and split it into overlapping basecall chunks (rec->len_raw_signal > 0).
void preprocess_signal(core_t *core, slow5_rec_t *rec, read_dat_t *read_dat, std::vector<basecall_chunk_t> &chunks);

// Given a read with unstitched chunks, stitch the chunks (accounting for overlap) and assign basecalled read and qstring to Read
void stitch_chunks(db_t *basecall_db, size_t i, std::string &sequence, std::string &qstring, std::vector<uint8_t> &moves, size_t len_raw_signal, int model_stride);

// Same as stitch_chunks but operating directly on a chunk vector (used by the streaming pipeline).
void stitch_chunks_vec(std::vector<basecall_chunk_t> &chunks, std::string &sequence, std::string &qstring, std::vector<uint8_t> &moves, size_t len_raw_signal, int model_stride);

// Load serialised tensor from disk.
std::vector<torch::Tensor> load_tensors(const std::string& dir, const std::vector<std::string>& tensors);

// Computes the q-th quantiles of each row of the input tensor `t`
// using a partial sort as opposed a full sort per torch::quantiles
// Only `interpolation='lower'` is currently implemented.
torch::Tensor quantile(const torch::Tensor t, const torch::Tensor q);

// Computes the q-th quantiles of each row of the input tensor `t`
// using a counting sort which is extremely fast for low range integers.
// Only `interpolation='lower'` is currently implemented.
torch::Tensor quantile_counting(const torch::Tensor t, const torch::Tensor q);

// temporary
inline void module_load_state_dict(torch::nn::Module& module, const std::vector<torch::Tensor>& weights) {
    assert(weights.size() == module.parameters().size());
    for (size_t idx = 0; idx < weights.size(); idx++) {
        module.parameters()[idx].data() = weights[idx].data();
    }
}

#endif