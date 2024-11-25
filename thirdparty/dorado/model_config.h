#pragma once

#include "BasecallerParams.h"
#include "models/kits.h"

#include <cmath>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

enum class Activation { SWISH, SWISH_CLAMP, TANH };

enum class ScalingStrategy { MED_MAD, QUANTILE, PA };
ScalingStrategy scaling_strategy_from_string(const char *strategy);

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

struct TxEncoderParams {
    // The number of expected features in the encoder/decoder inputs
    int d_model{-1};
    // The number of heads in the multi-head attention (MHA) models
    int nhead{-1};
    // The number of transformer layers
    int depth{-1};
    // The dimension of the feedforward model
    int dim_feedforward{-1};
    // Pair of ints defining (possibly asymmetric) sliding attention window mask
    std::pair<int, int> attn_window{-1, -1};
    // The deepnorm normalisation alpha parameter
    float deepnorm_alpha{1.0};
};

struct EncoderUpsampleParams {
    // The number of expected features in the encoder/decoder inputs
    int d_model;
    // Linear upsample scale factor
    int scale_factor;
};

struct CRFEncoderParams {
    int insize;
    int n_base;
    int state_len;
    float scale;
    float blank_score;
    bool expand_blanks;
    std::vector<int> permute;

    // compute the outsize
    int outsize() const {
        if (expand_blanks) {
            return static_cast<int>(pow(n_base, state_len + 1));
        }
        return (n_base + 1) * static_cast<int>(pow(n_base, state_len));
    };

    // compute the out_features
    int out_features() const { return static_cast<int>(pow(n_base, state_len + 1)); };
};

struct TxParams {
    TxEncoderParams tx;
    EncoderUpsampleParams upsample;
    CRFEncoderParams crf;

    // Self consistency check
    void check() const;
};

// Values extracted from config.toml used in construction of the model module.
struct CRFModelConfig {
    float qscale = 1.0f;
    float qbias = 0.0f;
    int lstm_size = 0;
    int stride = 1;
    bool bias = true;
    bool clamp = false;
    // If there is a decomposition of the linear layer, this is the bottleneck feature size.
    std::optional<int> out_features;
    int state_len;
    // Output feature size of the linear layer.  Dictated by state_len and whether
    // blank scores are explicitly stored in the linear layer output.
    int outsize;
    float blank_score;
    // The encoder scale only appears in pre-v4 models.  In v4 models
    // the value of 1 is used.
    float scale = 1.0f;
    int num_features;
    int sample_rate = -1;
    SignalNormalisationParams signal_norm_params;

    // Start position for mean Q-score calculation for
    // short reads.
    int32_t mean_qscore_start_pos = -1;

    models::SampleType sample_type{models::SampleType::UNKNOWN};

    // convolution layer params
    std::vector<ConvParams> convs;

    // Tx Model Params
    std::optional<TxParams> tx = std::nullopt;

    BasecallerParams basecaller;

    // True if this model config describes a LSTM model
    bool is_lstm_model() const { return !is_tx_model(); }
    // True if this model config describes a transformer model
    bool is_tx_model() const { return tx.has_value(); };

    // The model upsampling scale factor
    int scale_factor() const { return is_tx_model() ? tx->upsample.scale_factor : 1; };
    // The model stride multiplied by the upsampling scale factor
    int stride_inner() const { return stride * scale_factor(); };

    // Normalise the basecaller parameters `chunk_size` and `overlap` to the `stride_inner`
    void normalise_basecaller_params() {
        basecaller.normalise(chunk_size_granularity(), stride_inner());
    }

    size_t chunk_size_granularity() const { return stride_inner() * (is_tx_model() ? 16 : 1); }

    // True if `chunk_size` and `overlap` is evenly divisible by the `strde_inner`
    bool has_normalised_basecaller_params() const;
};

// True if this config at path describes a transformer model
bool is_tx_model_config(const std::filesystem::path& path);

CRFModelConfig load_crf_model_config(const std::filesystem::path& path);

bool is_rna_model(const CRFModelConfig& model_config);
bool is_duplex_model(const CRFModelConfig& model_config);

models::Chemistry parse_model_chemistry(const std::filesystem::path& path);
