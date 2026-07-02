#ifndef LSTM_MODEL_H
#define LSTM_MODEL_H

#include <torch/torch.h>

#include <vector>

#include "model_config.h"
#include "tensor_chunk_utils.h"
#include "calib.h"
#include "quant.h"

using namespace torch::nn;

// Procedural (C-style) plain-LSTM/CRF model — e.g. fast/hac v5. Weight structs + free-function
// forward; the heavy math stays on at:: ops (conv1d / cuDNN lstm / linear).
typedef struct {
    at::Tensor w;   // conv weight [out, in, winlen]
    at::Tensor b;   // conv bias [out]
    int stride;
    int padding;
    Activation activation;
} conv_layer_t;

typedef struct {
    at::Tensor w_ih, w_hh, b_ih, b_hh;   // single-layer LSTM params
    at::Tensor flat;                      // cuDNN-flattened weight buffer (keeps w_* views alive)
} lstm_layer_t;

// Pack a layer's {w_ih, w_hh, b_ih, b_hh} into one contiguous cuDNN/MIOpen weight buffer so
// torch::lstm doesn't recompact (and warn) on every call. No-op on CPU (device or CPU-only build).
// input_size == hidden for all our LSTMs.
void flatten_lstm_weights(lstm_layer_t &l, int input_size, int hidden, bool batch_first);

typedef struct {
    std::vector<conv_layer_t> convs;
    std::vector<lstm_layer_t> lstms;   // bidirectional-alternating (flip per layer)
    at::Tensor linear_w;               // CRF linear weight [outsize, lstm_size]
    at::Tensor linear_b;               // undefined => no bias
    int lstm_size;
    bool clamp;
    float clamp_min, clamp_max;
    lstm_stats_t *stats;
} lstm_model_t;

lstm_model_t *load_lstm_model_proc(const model_config_t &config, const torch::TensorOptions &options, lstm_stats_t *model_stats);
at::Tensor lstm_model_forward(const lstm_model_t *m, at::Tensor x);
void free_lstm_model(lstm_model_t *m);

// Conv stack forward: [N, C_in, T] -> [N, T, C_out]. Shared by the procedural LSTM/FLSTM/TX models.
at::Tensor conv_stack_forward(const std::vector<conv_layer_t> &convs, at::Tensor x);

// Procedural factored-LSTM (FLSTM) model — hac/fast v6. Down/up-projected LSTM with a per-timestep
// recurrence (fluke_flstm_step_gpu on GPU), decomposed linear1 + tanh-scaled linear2 CRF, no
// clamp. Weights are fake-quantised inline; calib hooks are registered with the real loaded weights.
typedef struct {
    at::Tensor dn_w_ih, dn_w_hh, up_w_ih, up_w_hh, up_b_ih, up_b_hh;
    std::string prefix;                        // quant_methods lookup key
    calib_layer_t *cl_dn_ih = nullptr, *cl_up_ih = nullptr;
    calib_layer_t *cl_dn_hh = nullptr, *cl_up_hh = nullptr;
} flstm_layer_t;

typedef struct {
    std::vector<conv_layer_t> convs;
    std::vector<flstm_layer_t> flstms;         // bidirectional-alternating
    int C, K;
    at::Tensor linear1_w, linear1_b;           // decomposed CRF linear (bias usually undefined)
    at::Tensor linear2_w;                       // tanh(scores) * scale
    lstm_stats_t *stats;
} flstm_model_t;

flstm_model_t *load_flstm_model_proc(const model_config_t &config, const torch::TensorOptions &options, lstm_stats_t *model_stats);
at::Tensor flstm_model_forward(const flstm_model_t *m, at::Tensor x);
void free_flstm_model(flstm_model_t *m);

#endif