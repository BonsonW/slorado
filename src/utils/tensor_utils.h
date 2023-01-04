#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>

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
