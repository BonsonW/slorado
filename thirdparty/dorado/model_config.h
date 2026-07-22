#pragma once

#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <cassert>
#include <numeric>
#include <memory>
#include <sys/types.h>
#include <regex.h>
#include <sstream>

#include <unordered_map>
#include <cstdint>

#if __cplusplus >= 201703L
#include <optional>
template<typename T>
using std_optional = std::optional<T>;
#define STD_NULLOPT std::nullopt
#else
#include <experimental/optional>
template<typename T>
using std_optional = std::experimental::optional<T>;
#define STD_NULLOPT std::experimental::nullopt
#endif

enum class Activation { SWISH, SWISH_CLAMP, TANH };

enum class ScalingStrategy { MED_MAD, QUANTILE, PA };
ScalingStrategy scaling_strategy_from_string(const char *strategy);

enum class SampleType {
    DNA,
    RNA002,
    RNA004,
    UNKNOWN,
};

typedef struct {
    bool standardise = false;
    float mean = 0.0f;
    float stdev = 1.0f;
} standardisation_scaling_params_t;

typedef struct {
    float quantile_a = 0.2f;
    float quantile_b = 0.9f;
    float shift_multiplier = 0.51f;
    float scale_multiplier = 0.53f;
} quantile_scaling_params_t;

typedef struct {
    ScalingStrategy strategy = ScalingStrategy::QUANTILE;
    quantile_scaling_params_t quantile;
    standardisation_scaling_params_t standarisation;
} signal_norm_params_t;

typedef struct {
    int insize;
    int size;
    int winlen;
    int stride = 1;
    Activation activation;
} conv_params_t;

typedef struct {
    int d_model = -1;         // expected features in the encoder/decoder inputs
    int nhead = -1;           // heads in the multi-head attention
    int depth = -1;           // transformer layers
    int dim_feedforward = -1; // feedforward dimension
    std::pair<int, int> attn_window{-1, -1};  // (possibly asymmetric) sliding attention window
    float deepnorm_alpha = 1.0;
} tx_encoder_params_t;

typedef struct {
    int d_model;      // expected features in the encoder/decoder inputs
    int scale_factor; // linear upsample scale factor
} encoder_upsample_params_t;

typedef struct {
    int insize;
    int n_base;
    int state_len;
    float scale;
    float blank_score;
    bool expand_blanks;
    std::vector<int> permute;
} crf_encoder_params_t;

int crf_outsize(const crf_encoder_params_t &p);
int crf_out_features(const crf_encoder_params_t &p);

typedef struct {
    tx_encoder_params_t tx;
    encoder_upsample_params_t upsample;
    crf_encoder_params_t crf;
} tx_params_t;

// Simplified basecall model config (models >= v5.0.0 only). Fixed architecture per family; carries
// only the fields used downstream.
typedef enum { MODEL_FAMILY_LSTM, MODEL_FAMILY_FLSTM, MODEL_FAMILY_TX } model_family_t;

typedef struct {
    model_family_t family;
    SampleType sample_type;
    std::string model_path;

    int chunk_size = -1;   // from [basecaller]; -1 if absent
    int overlap = -1;
    int stride = 1;        // product of conv strides (÷ upsample for TX)

    float qscale = 1.0f;   // [qscore]
    float qbias = 0.0f;

    int state_len = 0;
    int outsize = 0;       // 4^state_len * 4
    int num_features = 1;

    signal_norm_params_t signal_norm_params;  // used by scale_signal() in preprocessing

    std::vector<conv_params_t> convs;

    // LSTM / FLSTM families
    int lstm_size = 0;
    int lstm_layers = 0;
    int lstm_inner_dim = -1;   // >= 0 => FLSTM
    bool has_out_features = false;
    int out_features = 0;
    bool bias = true;
    bool clamp = false;
    bool crf_encoder_has_tanh = false;

    // TX family
    tx_params_t tx;
} model_config_t;

// v5.0.0+ loader: reads the model dir and returns the simplified config.
model_config_t load_model_config(const char *path);

enum ModelType { CONV_LSTM_V1, CONV_LSTM_V2, CONV_LSTM_V3, CONV_V1, UNKNOWN };

typedef struct {
    int in_size;
    int out_size;
} linear_params_t;

typedef struct {
    int size;
    bool reverse;
} lstm_config_params_t;

// Inert modbase parameter blocks (populated by the parse_* builders in model_config.cpp; behaviour
// lives in the free functions below).
typedef struct {
    std::vector<conv_params_t> sequence_convs;
    std::vector<conv_params_t> signal_convs;
    conv_params_t merge_conv;
    std::vector<lstm_config_params_t> lstms;  //< LSTM sizes per layer
    linear_params_t linear;
    std_optional<encoder_upsample_params_t> upsample;
} modules_params_t;

typedef struct {
    ModelType model_type;
    int size;
    int kmer_len;
    int num_out;
    int stride;
    int sequence_stride;
    std_optional<modules_params_t> modules;  // conv_lstm_v3 models only
} model_general_params_t;

typedef struct {
    bool do_rough_rescale = false;  ///< Whether to perform rough rescaling
    size_t center_idx = 0;          ///< The position in the kmer at which to check the levels
} refinement_params_t;

typedef struct {
    std::vector<std::string> codes;       ///< The modified bases codes (e.g 'h', 'm', CHEBI)
    std::vector<std::string> long_names;  ///< The long names of the modified bases.
    size_t count;                         ///< Number of mods

    std::string motif;    ///< The motif to look for modified bases within.
    size_t motif_offset;  ///< The position of the canonical base within the motif.

    char base;    ///< The canonical base 'ACGT'
    int base_id;  ///< The canonical base id 0-3

    std::vector<float> kmer_levels;
} modification_params_t;

typedef struct {
    int64_t samples_before;  ///< Number of context signal samples before a context hit.
    int64_t samples_after;   ///< Number of context signal samples after a context hit.
    int64_t samples;         ///< The total context samples (before + after)
    int64_t chunk_size;      ///< The total samples in a chunk

    int bases_before;  ///< Number of bases before the primary base of a kmer.
    int bases_after;   ///< Number of bases after the primary base of a kmer.
    int kmer_len;      ///< The kmer length given by `bases_before + bases_after + 1`

    bool reverse;             ///< Reverse model data before processing (rna model)
    bool base_start_justify;  ///< Justify the kmer encoding to start the context hit
} context_params_t;

// Product of conv strides.
int stride_product(const std::vector<conv_params_t> &cs);
// Sequence/signal stride ratio for a v3 modules block.
int modules_stride_ratio(const modules_params_t &m);
// Overall stride ratio (1 when there are no v3 modules).
int general_stride_ratio(const model_general_params_t &g);
// Normalise `v` up to the next multiple of `stride`; return the context params normalised likewise.
int64_t context_normalise(int64_t v, int64_t stride);
context_params_t context_normalised(const context_params_t &c, int stride);

typedef struct {
    std::string model_path;

    model_general_params_t general;  ///< General model params for legacy model architectures
    modification_params_t mods;      ///< Params for the modifications being detected
    context_params_t context;        ///< Params for the context over which mods are inferred
    refinement_params_t refine;      ///< Params for kmer refinement
} modbase_model_config_t;

bool is_chunked_input_model(const modbase_model_config_t &config);
// Read just the model type from a modbase model dir's config.toml (UNKNOWN if unrecognized).
ModelType get_modbase_model_type(const char *path);

typedef struct {
    std::vector<std::string> alphabet;
    std::string long_names;
    std::string context;
    std::array<size_t, 4> base_counts{};
    std::array<size_t, 4> base_probs_offsets{};
} modbase_info_t;

// Inert base-modification context: which motif (per canonical base) marks a modifiable site.
// Behaviour is in the mb_* free functions below; create with `modbase_context_t ctx{};`.
typedef struct {
    std::array<std::string, 4> motifs;   // indexed by mb_ubase_to_int(base); empty = none
    std::array<size_t, 4> offsets;
} modbase_context_t;

// Signal-space motif hit positions of `motif` (used as an REG_EXTENDED regex) within seq[0..seqlen),
// shifted by `offset`. Shared by populate_hits_seq and mb_get_sequence_mask.
std::vector<size_t> modbase_motif_hits(const std::string &motif, size_t offset, const char *seq, size_t seqlen);

int mb_ubase_to_int(char c);
void mb_set_context(modbase_context_t &ctx, std::string motif, size_t offset);
const std::string &mb_motif(const modbase_context_t &ctx, char base);
std::string mb_encode(const modbase_context_t &ctx);
bool mb_decode(modbase_context_t &ctx, const std::string &context_string);
std::vector<bool> mb_get_sequence_mask(const modbase_context_t &ctx, const char *seq, size_t seqlen);
void mb_update_mask(const modbase_context_t &ctx,
                    std::vector<bool> &mask,
                    const std::string &sequence,
                    const std::vector<std::string> &modbase_alphabet,
                    const std::vector<uint8_t> &modbase_probs,
                    uint8_t threshold);

modbase_info_t get_modbase_info(std::vector<modbase_model_config_t>& base_mod_params);
modbase_model_config_t load_modbase_model_config(const char *model_path);
SampleType get_sample_type_from_model_name(const std::string& model_name);
bool is_rna(SampleType);

bool is_tx_model_config(const char *path);