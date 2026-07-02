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

    SignalNormalisationParams signal_norm_params;  // used by scale_signal() in preprocessing

    std::vector<ConvParams> convs;

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
    TxParams tx;
} model_config_t;

// v5.0.0+ loader: reads the model dir and returns the simplified config.
model_config_t load_model_config(const char *path);

enum ModelType { CONV_LSTM_V1, CONV_LSTM_V2, CONV_LSTM_V3, CONV_V1, UNKNOWN };

struct LinearParams {
    int in_size;
    int out_size;
};

struct LSTMConfigParams {
    int size;
    bool reverse;
};

struct ModulesParams {
    std::vector<ConvParams> sequence_convs;
    std::vector<ConvParams> signal_convs;
    ConvParams merge_conv;
    std::vector<LSTMConfigParams> lstms;  //< LSTM sizes per layer
    LinearParams linear;
    std_optional<EncoderUpsampleParams> upsample;

    int stride_product(const std::vector<ConvParams>& cs) {
        return std::accumulate(cs.cbegin(), cs.cend(), 1, [](const int s, const auto& c) { return s * c.stride; });
    }

    int sequence_stride() { return stride_product(sequence_convs); };
    int signal_stride() { return stride_product(signal_convs); };
    int stride_ratio() {
        const auto seq = sequence_stride();
        const auto sig = signal_stride();
        assert(sig < seq);
        assert(sig % seq != 0);
        return sig / seq;
    };
};

struct ModelGeneralParams {
    const ModelType model_type;
    const int size;
    const int kmer_len;
    const int num_out;
    const int stride;
    const int sequence_stride;

    // For conv_lstm_v3 models only
    std_optional<ModulesParams> modules;

    int stride_ratio() {
        if (modules) {
            return modules->stride_ratio();
        } else {
            return 1;
        }
    }

    ModelGeneralParams(ModelType model_type_,
                       int size_,
                       int kmer_len_,
                       int num_out_,
                       int stride_,
                       int sequence_stride_,
                       std_optional<ModulesParams> modules_);
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

    std::vector<float> kmer_levels;

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

static const std::unordered_map<char, std::string> IUPAC_CODES =
        {
                // clang-format off
        {'A', "A"},
        {'C', "C"},
        {'G', "G"},
        {'T', "T"},
        {'U', "T"},  // basecalls will have "T"s instead of "U"s
        {'R', "[AG]"},
        {'Y', "[CT]"}, 
        {'S', "[GC]"}, 
        {'W', "[AT]"},
        {'K', "[GT]"}, 
        {'M', "[AC]"}, 
        {'B', "[CGT]"},
        {'D', "[AGT]"},
        {'H', "[ACT]"},
        {'V', "[ACG]"},
        {'N', "[ACGT]"},
                // clang-format on
};

struct MotifMatcher {
    MotifMatcher(const std::string& _motif, size_t _offset) : motif{_motif}, motif_offset{_offset} {};

    std::string expand_motif_regex(const std::string& motif) {
        std::string motif_regex = "(";
        for (auto base : motif) {
            motif_regex += IUPAC_CODES.at(base);
        }
        motif_regex += ")";
        return motif_regex;
    }

    std::vector<size_t> get_motif_hits(const char *seq, size_t seqlen) {
        std::vector<size_t> context_hits;
        regex_t compiled;
        if (regcomp(&compiled, motif.c_str(), REG_EXTENDED) != 0) {
            return context_hits;
        }

        size_t pos = 0;
        while (pos < seqlen) {
            regmatch_t match;
            if (regexec(&compiled, seq + pos, 1, &match, 0) != 0) {
                break;
            }
            auto hit = pos + match.rm_so + motif_offset;
            context_hits.push_back(hit);
            pos += match.rm_so + 1;
        }

        regfree(&compiled);
        return context_hits;
    }

    const std::string motif;
    const size_t motif_offset;
};

class ModBaseContext {
public:
    ModBaseContext() {};
    ~ModBaseContext() {};

    int ubase_to_int(char c) { return 0b11 & ((c >> 2) ^ (c >> 1)); }

    void set_context(std::string motif, size_t offset) {
        if (motif.size() < 2) {
            // empty motif, or just the canonical base
            return;
        }
        char base = motif.at(offset);
        auto index = ubase_to_int(base);
        motif_matchers[index] = std::make_unique<MotifMatcher>(motif, offset);
        motifs[index] = std::move(motif);
        offsets[index] = offset;
    }

    const std::string& motif(char base) {
        return motifs[ubase_to_int(base)];
    }

    size_t motif_offset(char base) { return offsets[ubase_to_int(base)]; }

    std::vector<bool> get_sequence_mask(char *sequence, size_t seqlen) {
        std::vector<bool> mask(seqlen, false);
        for (auto& matcher : motif_matchers) {
            if (matcher) {
                auto hits = matcher->get_motif_hits(sequence, seqlen);
                for (auto hit : hits) {
                    mask[hit] = true;
                }
            }
        }
        return mask;
    }

    std::string encode() {
        std::ostringstream s;
        for (size_t i = 0; i < 4; ++i) {
            if (motifs[i].empty()) {
                s << '_';
            } else {
                auto m = motifs[i];
                m[offsets[i]] = 'X';
                s << m;
            }
            if (i < 3) {
                s << ':';
            }
        }
        return s.str();
    }

    void update_mask(
        std::vector<bool>& mask,
        const std::string& sequence,
        const std::vector<std::string>& modbase_alphabet,
        const std::vector<uint8_t>& modbase_probs,
        uint8_t threshold
    ) {
        // First decide which elements of modbase_alphabet are modifications.
        struct ModifiedBase {
            char cardinal_base{0};
            std::vector<size_t> modified_channels;
        };
        const size_t num_channels = modbase_alphabet.size();
        const std::string CARDINAL_BASES{"ACGT"};
        std::vector<ModifiedBase> adjustments;
        ModifiedBase current_adjustment;
        for (size_t channel_idx = 0; channel_idx < num_channels; channel_idx++) {
            if (CARDINAL_BASES.find(modbase_alphabet[channel_idx]) != std::string::npos) {
                if (!current_adjustment.modified_channels.empty()) {
                    adjustments.emplace_back(std::move(current_adjustment));
                }
                current_adjustment = {modbase_alphabet[channel_idx][0], {}};
            } else {
                if (!motifs[ubase_to_int(current_adjustment.cardinal_base)].empty()) {
                    // This cardinal base has a context associated with modifications, so the mask should
                    // not be updated, regardless of the threshold.
                    continue;
                }
                current_adjustment.modified_channels.push_back(channel_idx);
            }
        }
        if (!current_adjustment.modified_channels.empty()) {
            adjustments.emplace_back(std::move(current_adjustment));
        }

        if (adjustments.empty()) {
            // No bases to adjust, so nothing to do.
            return;
        }

        // Update the mask only for canonical bases we have determined require an update.
        for (size_t base_idx = 0; base_idx < sequence.size(); ++base_idx) {
            bool requires_update = false;
            bool flag = false;
            for (const auto& adjustment : adjustments) {
                if (adjustment.cardinal_base == sequence[base_idx]) {
                    requires_update = true;
                    for (const auto channel_idx : adjustment.modified_channels) {
                        // We use |= here so that if there are multiple modifications possible for
                        // a canonical base, and any of them exceed the threshold, then we will have
                        // set the flag to true.
                        flag |= (modbase_probs[base_idx * num_channels + channel_idx] >= threshold);
                    }
                }
            }
            if (requires_update) {
                // Replace the flag if we need to, otherwise leave it unchanged.
                mask[base_idx] = flag;
            }
        }
    }


    bool decode(const std::string& context_string, bool create_matchers) {
        std::vector<std::string> tokens;
        std::istringstream context_stream(context_string);
        std::string token;
        while (std::getline(context_stream, token, ':')) {
            tokens.push_back(token);
        }
        if (tokens.size() != 4) {
            return false;
        }
        auto canonical = "ACGT";
        for (size_t i = 0; i < 4; ++i) {
            if (tokens[i] == "_") {
                motif_matchers[i].reset();
                motifs[i].clear();
                offsets[i] = 0;
            } else {
                auto x = tokens[i].find('X');
                if (x == std::string::npos) {
                    return false;
                }
                motifs[i] = tokens[i];
                motifs[i][x] = canonical[i];
                offsets[i] = x;
                if (create_matchers) {
                    motif_matchers[i] = std::make_unique<MotifMatcher>(motifs[i], offsets[i]);
                } else {
                    motif_matchers[i].reset();
                }
            }
        }
        return true;
    }
private:
    std::array<std::string, 4> motifs;
    std::array<size_t, 4> offsets = {{0, 0, 0, 0}};
    std::array<std::unique_ptr<MotifMatcher>, 4> motif_matchers;
};

ModBaseInfo get_modbase_info(std::vector<ModBaseModelConfig>& base_mod_params);
ModBaseModelConfig load_modbase_model_config(const char *model_path);
SampleType get_sample_type_from_model_name(const std::string& model_name);
bool is_rna(SampleType);

bool is_tx_model_config(const char *path);