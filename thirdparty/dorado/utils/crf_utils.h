#pragma once

#include <torch/torch.h>
#include <vector>

enum class Activation { SWISH, SWISH_CLAMP, TANH };
std::string to_string(const Activation& activation);

enum class ScalingStrategy { MED_MAD, QUANTILE, PA };
std::string to_string(const ScalingStrategy& strategy);
ScalingStrategy scaling_strategy_from_string(const std::string& strategy);

struct StandardisationScalingParams {
    bool standardise = false;
    float mean = 0.0f;
    float stdev = 1.0f;
};

struct QuantileScalingParams {
    float quantile_a = 0.2f;
    float quantile_b = 0.9f;
    float shift_multiplier = 0.51f;
    float scale_multiplier = 0.53f;
};

struct SignalNormalisationParams {
    ScalingStrategy strategy = ScalingStrategy::QUANTILE;

    QuantileScalingParams quantile;
    StandardisationScalingParams standarisation;
};

struct ConvParams {
    int insize;
    int size;
    int winlen;
    int stride = 1;
    Activation activation;
};

enum SampleType {
    DNA,
    RNA002,
    RNA004,
};

// Values extracted from config.toml used in construction of the model module.
struct CRFModelConfig {
    float qscale;
    float qbias;
    int lstm_size;
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
    float scale;
    int num_features;

    // SignalNormalisationParams signal_norm_params;
    std::vector<ConvParams> convs;
};

inline void module_load_state_dict(torch::nn::Module& module,
                            const std::vector<at::Tensor>& weights,
                            const std::vector<at::Tensor>& buffers) {
    assert(weights.size() == module.parameters().size());
    for (size_t idx = 0; idx < weights.size(); idx++) {
        module.parameters()[idx].data() = weights[idx].data();
    }

    assert(buffers.size() == module.buffers().size());
    for (size_t idx = 0; idx < buffers.size(); idx++) {
        module.buffers()[idx].data() = buffers[idx].data();
    }
}

CRFModelConfig load_crf_model_config(const std::string& model_path);

std::vector<torch::Tensor> load_crf_model_weights(const std::string& dir,
                                                  bool decomposition,
                                                  bool bias);

torch::nn::ModuleHolder<torch::nn::AnyModule> load_crf_model(const std::string& model_path,
                                                             const CRFModelConfig& model_config,
                                                             int batch_size,
                                                             int chunk_size,
                                                             const torch::TensorOptions& options);
