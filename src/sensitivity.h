#pragma once

#include <string>

#include <torch/torch.h>

// Accumulates KL divergence between fp16 baseline and quantized forward passes.
// One instance lives in core_t when --sensitivity is active.
struct sensitivity_stats_t {
    int64_t n_batches = 0;
    double kl_sum = 0.0;
    float kl_max = 0.f;

    // Accumulate KL(fp16_logits || quant_logits) for one batch.
    // Inputs: [N, T, vocab] raw logits (any device/dtype — moved to CPU float32 internally).
    void accumulate(const at::Tensor &fp16_logits, const at::Tensor &quant_logits);

    void save_tsv(const std::string &path, const char *quant_config_path) const;
};
