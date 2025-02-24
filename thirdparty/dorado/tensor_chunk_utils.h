/* @file signal_prep.h
**
** methods for preparing a signal for the base calling step
** @@
******************************************************************************/

#ifndef SIGNAL_PREP_H
#define SIGNAL_PREP_H

#include "slorado.h"
#include <torch/torch.h>

template <typename T>
T div_round_up(const T a, const T b) {
    return (a + b - 1) / b;
}
template <typename T>
T pad_to(const T a, const T b) {
    return div_round_up(a, b) * b;
}

void scale_signal(torch::Tensor &signal, float scaling, float offset, SignalNormalisationParams &scaling_params);

// Given a read with unstitched chunks, stitch the chunks (accounting for overlap) and assign basecalled read and qstring to Read
void stitch_chunks(std::vector<chunk_t> &chunks, std::string &sequence, std::string &qstring);

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
