#pragma once

#include <cmath>
#include <string>
#include <vector>
#include <optional>

enum class Activation { SWISH, SWISH_CLAMP, TANH };

enum class ScalingStrategy { MED_MAD, QUANTILE, PA };
ScalingStrategy scaling_strategy_from_string(const char *strategy);

enum class SampleType {
    DNA,
    RNA002,
    RNA004,
    UNKNOWN,
};

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
    int d_model = -1;
    // The number of heads in the multi-head attention (MHA) models
    int nhead = -1;
    // The number of transformer layers
    int depth = -1;
    // The dimension of the feedforward model
    int dim_feedforward = -1;
    // Pair of ints defining (possibly asymmetric) sliding attention window mask
    std::pair<int, int> attn_window{-1, -1};
    // The deepnorm normalisation alpha parameter
    float deepnorm_alpha = 1.0;
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

    int outsize() const {
        if (expand_blanks) {
            return static_cast<int>(pow(n_base, state_len + 1));
        }
        return (n_base + 1) * static_cast<int>(pow(n_base, state_len));
    };

    int out_features() const { return static_cast<int>(pow(n_base, state_len + 1)); };
};

struct TxParams {
    TxEncoderParams tx;
    EncoderUpsampleParams upsample;
    CRFEncoderParams crf;
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
    bool has_out_features;
    int out_features;
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

    // convolution layer params
    std::vector<ConvParams> convs;

    // Tx Model Params
    TxParams *tx = NULL;

    std::string model_path;

    SampleType sample_type;
};

enum ModelType { CONV_LSTM_V1, CONV_LSTM_V2, CONV_LSTM_V3, CONV_V1, UNKNOWN };

struct LinearParams {
    int in_size;
    int out_size;
};

struct LSTMParams {
    int size;
    bool reverse;
};

struct ModulesParams {
    std::vector<ConvParams> sequence_convs;
    std::vector<ConvParams> signal_convs;
    ConvParams merge_conv;
    std::vector<LSTMParams> lstms;  //< LSTM sizes per layer
    LinearParams linear;
    std::optional<EncoderUpsampleParams> upsample;
};

struct ModelGeneralParams {
    const ModelType model_type;
    const int size;
    const int kmer_len;
    const int num_out;
    const int stride;
    const int sequence_stride;

    // For conv_lstm_v3 models only
    std::optional<ModulesParams> modules;

    ModelGeneralParams(ModelType model_type_,
                       int size_,
                       int kmer_len_,
                       int num_out_,
                       int stride_,
                       int sequence_stride_,
                       std::optional<ModulesParams> modules_);
};

struct RefinementParams {
    const bool do_rough_rescale;  ///< Whether to perform rough rescaling
    const size_t center_idx;      ///< The position in the kmer at which to check the levels

    RefinementParams() : do_rough_rescale(false), center_idx(0) {}
    RefinementParams(int center_idx_);
};

struct ModificationParams {
    const std::vector<std::string> codes;       ///< The modified bases codes (e.g 'h', 'm', CHEBI)
    const std::vector<std::string> long_names;  ///< The long names of the modified bases.
    const size_t count;                         ///< Number of mods

    const std::string motif;    ///< The motif to look for modified bases within.
    const size_t motif_offset;  ///< The position of the canonical base within the motif.

    const char base;    ///< The canonical base 'ACGT'
    const int base_id;  ///< The canonical base id 0-3

    ModificationParams(std::vector<std::string> codes_,
                       std::vector<std::string> long_names_,
                       std::string motif_,
                       const size_t motif_offset_);
};

struct ContextParams {
    const int64_t samples_before;  ///< Number of context signal samples before a context hit.
    const int64_t samples_after;   ///< Number of context signal samples after a context hit.
    const int64_t samples;         ///< The total context samples (before + after)
    const int64_t chunk_size;      ///< The total samples in a chunk

    const int bases_before;  ///< Number of bases before the primary base of a kmer.
    const int bases_after;   ///< Number of bases after the primary base of a kmer.
    const int kmer_len;      ///< The kmer length given by `bases_before + bases_after + 1`

    const bool reverse;             ///< Reverse model data before processing (rna model)
    const bool base_start_justify;  ///< Justify the kmer encoding to start the context hit

    ContextParams(int64_t samples_before_,
                  int64_t samples_after_,
                  int64_t chunk_size_,
                  int bases_before_,
                  int bases_after_,
                  bool reverse_,
                  bool base_start_justify_);

    // Normalise `v` by `stride` strictly increasing the if needed.
    static int64_t normalise(const int64_t v, const int64_t stride);
    // Return the context params but normalised by a stride
    ContextParams normalised(const int stride) const;
};

struct ModBaseModelConfig {
    std::string model_path;

    ModelGeneralParams general;        ///< General model params for legacy model architectures
    ModificationParams mods;           ///< Params for the modifications being detected
    ContextParams context;             ///< Params for the context over which mods are inferred
    RefinementParams refine;           ///< Params for kmer refinement

    bool is_chunked_input_model() const {
        return (general.model_type == ModelType::CONV_LSTM_V2) ||
               (general.model_type == ModelType::CONV_LSTM_V3);
    };

    ModBaseModelConfig(const char *model_path_,
                       ModelGeneralParams general_,
                       ModificationParams mods_,
                       ContextParams context_,
                       RefinementParams refine_);
};

struct ModBaseInfo {
    ModBaseInfo() = default;
    ModBaseInfo(std::vector<std::string> alphabet_, std::string long_names_, std::string context_)
            : alphabet(std::move(alphabet_)),
              long_names(std::move(long_names_)),
              context(std::move(context_)) {}
    std::vector<std::string> alphabet;
    std::string long_names;
    std::string context;
    std::array<size_t, 4> base_counts{};
    std::array<size_t, 4> base_probs_offsets{};
};

ModBaseInfo get_modbase_info(std::vector<ModBaseModelConfig>& base_mod_params);
ModBaseModelConfig load_modbase_model_config(const char *model_path);
CRFModelConfig load_lstm_model_config(const char *path);
CRFModelConfig load_tx_model_config(const char *path);
SampleType get_sample_type_from_model_name(const std::string& model_name);
bool is_rna(SampleType);

bool is_tx_model_config(const char *path);