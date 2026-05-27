#pragma once

#include <deque>
#include <string>
#include <unordered_map>

#include <torch/torch.h>

struct calib_layer_t {
    std::string name;

    // Weight stats — computed once from the weight matrix at registration.
    // weight shape: (out_features, in_features); operation: output = input @ weight.T
    float w_min = 0.f, w_max = 0.f, w_amax = 0.f;
    at::Tensor w_per_out_ch_amax;   // CPU float32, shape (out_features,)

    // Input activation stats — accumulated across forward calls.
    float x_min =  1e38f, x_max = -1e38f, x_amax = 0.f;
    at::Tensor x_per_in_ch_amax;    // CPU float32, shape (in_features,), lazy-initialized
    int64_t n_batches = 0;
};

struct calib_stats_t {
    // Storage with stable element addresses (unlike std::vector).
    std::deque<calib_layer_t> entries;
    std::unordered_map<std::string, calib_layer_t*> by_name;

    // Register a linear layer. Computes weight stats from the given weight tensor
    // (can be on any device/dtype). Returns a stable pointer valid for the lifetime
    // of this calib_stats_t. Call only during model construction, before inference.
    // Returns nullptr if a layer with this name was already registered (multi-runner dedup).
    calib_layer_t* register_layer(const std::string &name, const at::Tensor &weight);

    // Recompute weight stats for a layer after weights are loaded (e.g. FLSTM uses
    // torch::empty() during construction, so real values aren't available until
    // load_state_dict() runs). Safe to call on the first forward pass.
    void update_weight(calib_layer_t *layer, const at::Tensor &weight);

    // Accumulate activation stats from an input tensor (any device/dtype).
    // input can have any leading dimensions; last dim is treated as in_features.
    void accumulate(calib_layer_t *layer, const at::Tensor &input);

    // Write results as JSON to path.
    void save_json(const std::string &path) const;
};
