/* @file signal_prep.h
**
** methods for preparing a signal for the base calling step
** @@
******************************************************************************/

#ifndef SIGNAL_PREP_H
#define SIGNAL_PREP_H

#include <slow5/slow5.h>

#include <torch/torch.h>

#include <string>
#include <vector>

#include "Chunk.h"

torch::Tensor tensor_from_record(slow5_rec_t *rec);
std::pair<float, float> normalisation(torch::Tensor& x);
int trim(
    torch::Tensor signal,
    int window_size = 40,
    float threshold = 2.4,
    int min_elements = 3,
    int max_samples = 8000,
    float max_trim = 0.3
);
void scale_signal(torch::Tensor &signal, float scaling, float offset);
std::vector<Chunk *> chunks_from_tensor(torch::Tensor &tensor, int chunk_size, int overlap);
std::vector<torch::Tensor> tensor_as_chunks(torch::Tensor &signal, std::vector<Chunk *> &chunks, size_t chunk_size);

// Given a read with unstitched chunks, stitch the chunks (accounting for overlap) and assign basecalled read and qstring to Read
void stitch_chunks(std::vector<Chunk *> &chunks, std::string &sequence, std::string &qstring);




// Serialise Torch tensor to disk.
void serialise_tensor(torch::Tensor t, const std::string& path);
// Load serialised tensor from disk.
std::vector<torch::Tensor> load_tensors(const std::string& dir,
                                        const std::vector<std::string>& tensors);

// Computes the q-th quantiles of each row of the input tensor `t`
// using a partial sort as opposed a full sort per torch::quantiles
// Only `interpolation='lower'` is currently implemented.
torch::Tensor quantile(const torch::Tensor t, const torch::Tensor q);

// Computes the q-th quantiles of each row of the input tensor `t`
// using a counting sort which is extremely fast for low range integers.
// Only `interpolation='lower'` is currently implemented.
torch::Tensor quantile_counting(const torch::Tensor t, const torch::Tensor q);

// temporary
inline void module_load_state_dict(torch::nn::Module& module,
                            const std::vector<torch::Tensor>& weights,
                            const std::vector<torch::Tensor>& buffers = {}) {
    assert(weights.size() == module.parameters().size());
    for (size_t idx = 0; idx < weights.size(); idx++) {
        module.parameters()[idx].data() = weights[idx].data();
    }

    assert(buffers.size() == module.buffers().size());
    for (size_t idx = 0; idx < buffers.size(); idx++) {
        module.buffers()[idx].data() = buffers[idx].data();
    }
}
#endif
