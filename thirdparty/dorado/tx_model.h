#pragma once

#include "lstm_model.h"
#include "calib.h"
#include "error.h"
#include "quant.h"
#include "misc.h"
#include "tensor_chunk_utils.h"
#include "fluke_wrapper.h"

#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <torch/nn.h>

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace torch::nn;

using MaskKey = std::pair<int64_t, torch::Device>;

// Hash function for std::pair<int64_t, torch::Device>
struct MaskKeyHash {
    std::size_t operator()(const MaskKey &key) const {
        auto hash1 = std::hash<int64_t>{}(key.first);
        auto hash2 = std::hash<torch::Device>{}(key.second);
        return hash1 ^ (hash2 << 1);
    }
};

// Procedural (C-style) transformer model — sup v5 / rna sup v6. Weight structs + free-function
// forward; the heavy math (conv, wqkv/out_proj/ff GEMMs, rotary, RMSNorm, SDPA/flash, fluke int8)
// stays on at:: ops and the openfish/fluke kernels.
typedef struct {
    // rotary embedding (precomputed sin/cos, fp32)
    at::Tensor rot_sin, rot_cos;
    // attention
    int d_model, nhead, head_dim, num_splits;
    std::pair<int, int> attn_window;
    at::Tensor wqkv_w;                 // [3C, C], no bias
    at::Tensor out_proj_w, out_proj_b; // out_proj (+ bias)
    std::string attn_prefix;
    calib_layer_t *cl_wqkv = nullptr, *cl_out_proj = nullptr;
    fluke_int8_backend_t *attn_backend = nullptr;
    tensor_quant_t qw_wqkv;
    // gated MLP
    int hidden_features;
    at::Tensor fc1_w, fc2_w;
    std::string ff_prefix;
    calib_layer_t *cl_fc1 = nullptr, *cl_fc2 = nullptr;
    fluke_int8_backend_t *ff_backend = nullptr;
    tensor_quant_t qw_gate, qw_up;
    // norms
    at::Tensor norm1_w, norm2_w;
    float deepnorm_alpha;
} tx_layer_t;

typedef struct {
    std::vector<conv_layer_t> convs;
    std::vector<tx_layer_t> layers;
    bool quant_stream = false;         // int8 fused residual-stream path available/enabled
    at::Tensor up_w, up_b;             // upsample linear (+bias)
    int scale_factor;
    at::Tensor crf_w;                  // CRF linear weight, pre-scaled at load
    tx_stats_t *stats;
    std::unordered_map<MaskKey, torch::Tensor, MaskKeyHash> mask_cache;  // attention-window masks
} tx_model_t;

tx_model_t *load_tx_model_proc(const model_config_t &config, const torch::TensorOptions &options, tx_stats_t *model_stats, bool use_flash, const std::string &quant_mode, int nthreads);
at::Tensor tx_model_forward(tx_model_t *m, at::Tensor x);
void free_tx_model(tx_model_t *m);
