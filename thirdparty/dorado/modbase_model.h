#pragma once

#include <torch/torch.h>

#include "model_config.h"
#include "lstm_model.h"   // conv_layer_t, lstm_layer_t

// Procedural (C-style) modbase model — the chunked ConvLSTM used by 5mCG_5hmCG@v3 (the only
// supported modbase model). sig/seq conv stacks (SWISH) -> merge conv -> 2 LSTMs (silu + time-flip
// between them) -> linear -> softmax. Weight structs + free-function forward on at:: ops.
typedef struct {
    conv_layer_t sig_conv[3];
    conv_layer_t seq_conv[2];
    conv_layer_t merge_conv;
    lstm_layer_t lstm1, lstm2;   // default (batch_first=false) LSTMs over [T, N, C]
    at::Tensor linear_w, linear_b;
    bool chunked;                // v2/v3 (per-timestep output) vs v1 (final timestep only)
    bool lstm_silu;              // v1/v2 apply SiLU after each LSTM; conv_lstm_v3 does NOT
} modbase_model_t;

modbase_model_t *load_modbase_model_proc(const modbase_model_config_t &config, const at::TensorOptions &options, int batchsize);
// sigs: [N, 1, T] (NCT); seqs: [N, T, kmer_len*4] one-hot int8 (NTC). Returns per-read mod scores.
at::Tensor modbase_model_forward(const modbase_model_t *m, at::Tensor sigs, at::Tensor seqs);
void free_modbase_model(modbase_model_t *m);
