#pragma once

#include <torch/torch.h>

#include <vector>
#include <string>

struct SignalNormalisationParams {
    float quantile_a = 0.2f;
    float quantile_b = 0.9f;
    float shift_multiplier = 0.51f;
    float scale_multiplier = 0.53f;
    bool quantile_scaling = true;
};

// Values extracted from config.toml used in construction of the model module.
struct CRFModelConfig {
    float qscale;
    float qbias;
    int conv;
    int insize;
    int stride;
    bool bias;
    bool clamp;
    // If there is a decomposition of the linear layer, this is the bottleneck feature size.
    bool decomposition;
    int out_features;
    int state_len;
    // Output feature size of the linear layer.  Dictated by state_len and whether
    // blank scores are explicitly stored in the linear layer output.
    int outsize;
    float blank_score;
    float scale = 1.0;
    int sample_rate = -1;
    int num_features;

    std::string model_path;

    SignalNormalisationParams signal_norm_params;
};

CRFModelConfig load_crf_model_config(const std::string& model_path);

std::vector<torch::Tensor> load_crf_model_weights(const std::string& dir,
                                                  bool decomposition,
                                                  bool bias);

torch::nn::ModuleHolder<torch::nn::AnyModule> load_crf_model(const CRFModelConfig& model_config,
                                                             const torch::TensorOptions& options);
