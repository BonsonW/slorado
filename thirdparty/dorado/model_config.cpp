#include "toml.h"
#include "src/error.h"
#include "model_config.h"

#include <unordered_map>
#include <numeric>
#include <algorithm>

const std::vector<int> BASE_IDS = []() {
    std::vector<int> base_ids(256, -1);
    base_ids['A'] = 0;
    base_ids['C'] = 1;
    base_ids['G'] = 2;
    base_ids['T'] = 3;
    return base_ids;
}();

const std::string ERR_STR = "Invalid modbase model parameter in ";

// Indicates that a value has no default and is therefore required
constexpr std::optional<int> REQUIRED = std::nullopt;

enum SublayerType { CLAMP, CONVOLUTION, LINEAR, LINEAR_CRF_ENCODER, LSTM, PERMUTE, UPSAMPLE, UNRECOGNISED };
static const std::unordered_map<std::string, SublayerType> sublayer_map = {
    {"clamp", SublayerType::CLAMP},
    {"convolution", SublayerType::CONVOLUTION},
    {"linear", SublayerType::LINEAR},
    {"linearcrfencoder", SublayerType::LINEAR_CRF_ENCODER},
    {"lstm", SublayerType::LSTM},
    {"permute", SublayerType::PERMUTE},
    {"upsample", SublayerType::UPSAMPLE},
};

void check_toml_table(const toml_table_t *table) {
    if (!table) {
        ERROR("%s", "missing table in toml, please make sure your model version is supported");
        exit(EXIT_FAILURE);
    }
}

void check_toml_array(const toml_array_t *arr) {
    if (!arr) {
        ERROR("%s", "missing array in toml, please make sure your model version is supported");
        exit(EXIT_FAILURE);
    }
}

void check_toml_datum(const toml_datum_t datum) {
    if (!datum.ok) {
        ERROR("%s", "error reading field in toml, please make sure your model version is supported");
        exit(EXIT_FAILURE);
    }
}

const toml_table_t *toml_table_fallback_prereq(const toml_table_t *config_toml, std::vector<std::string> fallbacks) {
    const toml_table_t *ret = config_toml;
    for (size_t i = 0; i < fallbacks.size()-1; ++i) {
        const char *fallback = fallbacks[i].c_str();
        ret = toml_table_in(ret, fallback);
        check_toml_table(ret);
    }
    return ret;
}

toml_table_t *toml_table_fallback(toml_table_t *config_toml, std::vector<std::string> fallbacks) {
    const toml_table_t *prereq = toml_table_fallback_prereq(config_toml, fallbacks);
    const char *fallback = fallbacks.back().c_str();
    toml_table_t *ret = toml_table_in(prereq, fallback);
    check_toml_table(ret);
    return ret;
}

toml_array_t *toml_array_fallback(toml_table_t *config_toml, std::vector<std::string> fallbacks) {
    const toml_table_t *prereq = toml_table_fallback_prereq(config_toml, fallbacks);
    const char *fallback = fallbacks.back().c_str();
    toml_array_t *ret = toml_array_in(prereq, fallback);
    check_toml_array(ret);
    return ret;
}

toml_datum_t toml_int_fallback(toml_table_t *config_toml, std::vector<std::string> fallbacks) {
    const toml_table_t *prereq = toml_table_fallback_prereq(config_toml, fallbacks);
    const char *fallback = fallbacks.back().c_str();
    toml_datum_t ret = toml_int_in(prereq, fallback);
    check_toml_datum(ret);
    return ret;
}

toml_datum_t toml_double_fallback(toml_table_t *config_toml, std::vector<std::string> fallbacks) {
    const toml_table_t *prereq = toml_table_fallback_prereq(config_toml, fallbacks);
    const char *fallback = fallbacks.back().c_str();
    toml_datum_t ret = toml_double_in(prereq, fallback);
    check_toml_datum(ret);
    return ret;
}

toml_datum_t toml_string_fallback(const toml_table_t *config_toml, std::vector<std::string> fallbacks) {
    const toml_table_t *prereq = toml_table_fallback_prereq(config_toml, fallbacks);
    const char *fallback = fallbacks.back().c_str();
    toml_datum_t ret = toml_string_in(prereq, fallback);
    check_toml_datum(ret);
    return ret;
}

bool toml_key_fallback(toml_table_t *config_toml, std::vector<std::string> fallbacks) {
    toml_table_t *ret = config_toml;
    for (size_t i = 0; i < fallbacks.size(); ++i) {
        const char *fallback = fallbacks[i].c_str();
        if (toml_key_exists(ret, fallback)) {
            ret = toml_table_in(ret, fallback);
            check_toml_table(ret);
        } else {
            return false;
        }
    }
    return true;
}

// Parse sublayer extracting convolution parameters. This is for use on v4+ models only
ConvParams parse_conv_params(const toml_table_t *segment, bool clamp) {
    ConvParams params;
    toml_datum_t insize = toml_int_in(segment, "insize");
    check_toml_datum(insize);
    toml_datum_t size   = toml_int_in(segment, "size");
    check_toml_datum(size);
    toml_datum_t winlen = toml_int_in(segment, "winlen");
    check_toml_datum(winlen);
    toml_datum_t stride = toml_int_in(segment, "stride");
    check_toml_datum(stride);

    params.insize = insize.u.i;
    params.size   = size.u.i;
    params.winlen = winlen.u.i;
    params.stride = stride.u.i;

    toml_datum_t activation = toml_string_in(segment, "activation");
    check_toml_datum(activation);
    if (strcmp(activation.u.s, "swish") == 0) {
        params.activation = clamp ? Activation::SWISH_CLAMP : Activation::SWISH;
    } else if (strcmp(activation.u.s, "tanh") == 0) {
        params.activation = Activation::TANH;
    } else {
        ERROR("Unknown activation: `%s` in model config, expected `swish` or `tanh`", activation.u.s);
        exit(EXIT_FAILURE);
    }

    free(activation.u.s);

    return params;
}

SublayerType sublayer_type(const toml_table_t *segment) {
    toml_datum_t type = toml_string_in(segment, "type");
    check_toml_datum(type);
    auto mapping_iter = sublayer_map.find(type.u.s);
    if (mapping_iter == sublayer_map.end()) {
        return SublayerType::UNRECOGNISED;
    }

    free(type.u.s);

    return mapping_iter->second;
}

bool has_clamp(const std::vector<toml_table_t *> &sublayers) {
    for (const auto &segment : sublayers) {
        if (sublayer_type(segment) == SublayerType::CLAMP) {
            return true;
        }
    }
    return false;
}

// Parse sublayers extracting convolution parameters. This is for use on v4+ models only
std::vector<ConvParams> parse_convs(const std::vector<toml_table_t *> &sublayers) {
    std::vector<ConvParams> convs;
    for (size_t i = 0; i < sublayers.size(); ++i) {
        // If the sublayer after a convolution is a clamp, the activation function may have
        // a fused implementation
        if (sublayer_type(sublayers.at(i)) == SublayerType::CONVOLUTION) {
            const bool has_clamp_next = ((i + 1) < sublayers.size()) &&sublayer_type(sublayers.at(i + 1)) == SublayerType::CLAMP;
            ConvParams conv = parse_conv_params(sublayers.at(i), has_clamp_next);
            convs.push_back(conv);
        }
    }
    return convs;
}

void parse_qscore_params(CRFModelConfig &config, const toml_table_t *config_toml) {
    if (toml_key_exists(config_toml, "qscore")) {
        toml_table_t *qscore = toml_table_in(config_toml, "qscore");
        check_toml_table(qscore);
        toml_datum_t qbias = toml_double_in(qscore, "bias");
        check_toml_datum(qbias);
        toml_datum_t qscale = toml_double_in(qscore, "scale");
        check_toml_datum(qscale);

        config.qbias = qbias.u.d;
        config.qscale = qscale.u.d;

        if (toml_key_exists(qscore, "mean_qscore_start_pos")) {
            toml_datum_t mean_qscore_start_pos = toml_int_in(qscore, "mean_qscore_start_pos");
            check_toml_datum(mean_qscore_start_pos);
            config.mean_qscore_start_pos = mean_qscore_start_pos.u.i;
        } else {
            int fallback = 60;
            INFO("mean_qscore_start_pos not found in config toml, setting default to %d", fallback);
            config.mean_qscore_start_pos = 60;
        }
    } else {
        // no qscore calibration found
    }
}

ScalingStrategy scaling_strategy_from_string(const char *strategy) {
    if (strcmp(strategy, "med_mad") == 0) {
        return ScalingStrategy::MED_MAD;
    }
    if (strcmp(strategy, "quantile") == 0) {
        return ScalingStrategy::QUANTILE;
    }
    if (strcmp(strategy, "pa") == 0) {
        return ScalingStrategy::PA;
    }
    ERROR("Unknown scaling strategy: `%s`", strategy);
    exit(EXIT_FAILURE);
}

// Parse a the config.toml to resolve the scaling parameters.
SignalNormalisationParams parse_signal_normalisation_params(const toml_table_t *config_toml) {
    SignalNormalisationParams params;

    // scaling.strategy introduced with v4.3 models
    if (toml_key_exists(config_toml, "scaling")) {
        const toml_table_t *scaling = toml_table_in(config_toml, "scaling");
        check_toml_table(scaling);

        toml_datum_t strategy = toml_string_in(scaling, "strategy");
        check_toml_datum(strategy);

        params.strategy = scaling_strategy_from_string(strategy.u.s);
        free(strategy.u.s);
    }

    if (toml_key_exists(config_toml, "normalisation")) {
        const toml_table_t *norm = toml_table_in(config_toml, "normalisation");
        check_toml_table(norm);

        toml_datum_t quantile_a = toml_double_in(norm, "quantile_a");
        check_toml_datum(quantile_a);
        toml_datum_t quantile_b = toml_double_in(norm, "quantile_b");
        check_toml_datum(quantile_b);
        toml_datum_t shift_multiplier = toml_double_in(norm, "shift_multiplier");
        check_toml_datum(shift_multiplier);
        toml_datum_t scale_multiplier = toml_double_in(norm, "scale_multiplier");
        check_toml_datum(scale_multiplier);
        
        params.quantile.quantile_a       = quantile_a.u.d;
        params.quantile.quantile_b       = quantile_b.u.d;
        params.quantile.shift_multiplier = shift_multiplier.u.d;
        params.quantile.scale_multiplier = scale_multiplier.u.d;

        if (params.strategy != ScalingStrategy::QUANTILE) {
            WARNING("%s", "Normalisation parameters are only used when `scaling.strategy = quantile`");
        }
    }

    if (toml_key_exists(config_toml, "standardisation")) {
        const toml_table_t *norm = toml_table_in(config_toml, "standardisation");
        check_toml_table(norm);

        toml_datum_t standardise = toml_int_in(norm, "standardise");
        check_toml_datum(standardise);
        params.standarisation.standardise = standardise.u.i > 0;
        if (params.standarisation.standardise) {
            toml_datum_t mean = toml_double_in(norm, "mean");
            check_toml_datum(mean);
            toml_datum_t stdev = toml_double_in(norm, "stdev");
            check_toml_datum(stdev);

            params.standarisation.mean = mean.u.d;
            params.standarisation.stdev = stdev.u.d;
        }

        if (params.standarisation.standardise && params.strategy != ScalingStrategy::PA) {
            ERROR("%s", "Signal standardisation is implemented only for `scaling.strategy = pa`");
            exit(EXIT_FAILURE);
        }

        if (params.standarisation.stdev <= 0.0f) {
            ERROR("Config error: `standardisation.stdev` must be greater than 0, got: %f", params.standarisation.stdev);
            exit(EXIT_FAILURE);
        }
    }

    return params;
}

TxEncoderParams parse_tx_encoder_params(toml_table_t *cfg) {
    toml_table_t *enc = toml_table_fallback(cfg, {"model", "encoder", "transformer_encoder"});
    TxEncoderParams params;

    toml_datum_t depth = toml_int_in(enc, "depth");
    check_toml_datum(depth);
    toml_datum_t d_model = toml_int_fallback(enc, {"layer", "d_model"});
    check_toml_datum(d_model);
    toml_datum_t nhead = toml_int_fallback(enc, {"layer", "nhead"});
    check_toml_datum(nhead);
    toml_datum_t dim_feedforward = toml_int_fallback(enc, {"layer", "dim_feedforward"});
    check_toml_datum(dim_feedforward);
    toml_datum_t deepnorm_alpha = toml_double_fallback(enc, {"layer", "deepnorm_alpha"});
    check_toml_datum(deepnorm_alpha);

    params.depth = depth.u.i;
    params.d_model = d_model.u.i;
    params.nhead = nhead.u.i;
    params.dim_feedforward = dim_feedforward.u.i;
    params.deepnorm_alpha = deepnorm_alpha.u.d;

    const toml_array_t *attn_window_ = toml_array_fallback(enc, {"layer", "attn_window"});
    check_toml_array(attn_window_);
    
    params.attn_window = {};
    {
        toml_datum_t e = toml_int_at(attn_window_, 0);
        if (!e.ok)  {
            ERROR("%s", "error loading window");
            exit(EXIT_FAILURE);
        }
        params.attn_window.first = e.u.i;
    }
    {
        toml_datum_t e = toml_int_at(attn_window_, 1);
        if (!e.ok)  {
            ERROR("%s", "error loading window");
            exit(EXIT_FAILURE);
        };
        params.attn_window.second = e.u.i;
    }

    return params;
}

EncoderUpsampleParams parse_encoder_upsample_params(toml_table_t *cfg) {
    toml_table_t *ups = toml_table_fallback(cfg, {"model", "encoder", "upsample"});
    EncoderUpsampleParams params;

    toml_datum_t d_model = toml_int_in(ups, "d_model");
    check_toml_datum(d_model);
    toml_datum_t scale_factor = toml_int_in(ups, "scale_factor");
    check_toml_datum(scale_factor);

    params.d_model = d_model.u.i;
    params.scale_factor = scale_factor.u.i;

    return params;
}

CRFEncoderParams parse_crf_encoder_params(toml_table_t *cfg) {
    toml_table_t *crf = toml_table_fallback(cfg, {"model", "encoder", "crf"});
    CRFEncoderParams params;

    toml_datum_t insize = toml_int_in(crf, "insize");
    check_toml_datum(insize);
    toml_datum_t n_base = toml_int_in(crf, "n_base");
    check_toml_datum(n_base);
    toml_datum_t state_len = toml_int_in(crf, "state_len");
    check_toml_datum(state_len);
    toml_datum_t scale = toml_double_in(crf, "scale");
    check_toml_datum(scale);
    toml_datum_t blank_score = toml_double_in(crf, "blank_score");
    check_toml_datum(blank_score);
    toml_datum_t expand_blanks = toml_bool_in(crf, "expand_blanks");
    check_toml_datum(expand_blanks);
    toml_array_t *permute = toml_array_in(crf, "permute");
    check_toml_array(permute);

    params.insize = insize.u.i;
    params.n_base = n_base.u.i;
    params.state_len = state_len.u.i;
    params.scale = scale.u.d;
    params.blank_score = blank_score.u.d;
    params.expand_blanks = expand_blanks.u.b;

    params.permute = {};
    for (int i = 0; ; i++) {
        toml_datum_t e = toml_int_at(permute, i);
        if (!e.ok) break;
        params.permute.push_back(e.u.i);
    }

    return params;
}

CRFModelConfig load_lstm_model_config(const char *path) {
    FILE* fp;
    char errbuf[200];

    char *cpath = (char *)malloc(strlen(path) + 100);
    MALLOC_CHK(cpath);
    sprintf(cpath, "%s/config.toml", path);

    fp = fopen(cpath, "r");
    if (!fp) {
        ERROR("cannot open toml - %s: %s", cpath, strerror(errno));
        exit(EXIT_FAILURE);
    }

    toml_table_t *config_toml = toml_parse_file(fp, errbuf, sizeof(errbuf));
    fclose(fp);
    check_toml_table(config_toml);

    CRFModelConfig config;
    config.has_out_features = false;

    parse_qscore_params(config, config_toml);

    toml_table_t *input = toml_table_in(config_toml, "input");
    check_toml_table(input);
    toml_datum_t num_features = toml_int_in(input, "features");
    check_toml_datum(num_features);
    config.num_features = num_features.u.i;

    toml_table_t *encoder = toml_table_in(config_toml, "encoder");
    check_toml_table(encoder);

    if (toml_key_exists(encoder, "type")) {
        // v4-type model
        toml_array_t *_sublayers = toml_array_in(encoder, "sublayers");
        check_toml_array(_sublayers);
        config.bias = false;

        std::vector<toml_table_t *> sublayers = {};
        for (int i = 0; ; i++) {
            toml_table_t *segment = toml_table_at(_sublayers, i);
            if (!segment) break;
            sublayers.push_back(segment);
        }
        
        // todo:
        // warn_unrecognised_sublayers(sublayers);
        
        // v4-type model
        config.clamp = has_clamp(sublayers);
        config.convs = parse_convs(sublayers);
        // Overall stride is the product of all conv layers' strides.
        for (const auto &cv : config.convs) {
            config.stride *= cv.stride;
        }
        config.lstm_size = config.convs.back().size;

        for (const auto &segment : sublayers) {
            const auto type = sublayer_type(segment);
            if (type == SublayerType::LINEAR) {
                // Specifying out_features implies a decomposition of the linear layer matrix
                // multiply with a bottleneck before the final feature size.
                toml_datum_t out_features = toml_int_in(segment, "out_features");
                check_toml_datum(out_features);
                config.out_features = out_features.u.i;
                config.has_out_features = true;
                config.bias = config.lstm_size > 128;
            } else if (type == SublayerType::LINEAR_CRF_ENCODER) {
                toml_datum_t blank_score = toml_double_in(segment, "blank_score");
                check_toml_datum(blank_score);
                config.blank_score = blank_score.u.d;
            }
        }
        
    } else {
        // pre-v4 model
        toml_datum_t stride = toml_int_in(encoder, "stride");
        check_toml_datum(stride);
        toml_datum_t features = toml_int_in(encoder, "features");
        check_toml_datum(features);
        toml_datum_t blank_score = toml_double_in(encoder, "blank_score");
        check_toml_datum(blank_score);
        toml_datum_t scale = toml_double_in(encoder, "scale");
        check_toml_datum(scale);
        
        config.stride = stride.u.i;
        config.lstm_size = features.u.i;
        config.blank_score = blank_score.u.d;
        config.scale = scale.u.d;

        int first_conv = 4;
        if (toml_key_exists(encoder, "first_conv_size")) {
            toml_datum_t conv = toml_int_in(encoder, "first_conv_size");
            check_toml_datum(conv);
            first_conv = conv.u.i;
        }

        config.convs.push_back(ConvParams{config.num_features, first_conv, 5, 1, Activation::SWISH});
        config.convs.push_back(ConvParams{first_conv, 16, 5, 1, Activation::SWISH});
        config.convs.push_back(ConvParams{16, config.lstm_size, 19, config.stride, Activation::SWISH});
    }

    toml_table_t *global_norm = toml_table_in(config_toml, "global_norm");
    check_toml_table(global_norm);

    // Note that in v4 files state_len appears twice: under global_norm and under
    // linearcrfencoder.  We are ignoring the latter.
    toml_datum_t state_len = toml_int_in(global_norm, "state_len");
    check_toml_datum(state_len);
    config.state_len = state_len.u.i;

    // CUDA and CPU paths do not output explicit stay scores from the NN.
    config.outsize = pow(4, config.state_len) * 4;

    config.signal_norm_params = parse_signal_normalisation_params(config_toml);

    if (config.convs.size() != 3) {
        ERROR("Expected 3 convolution layers but found: %lu", config.convs.size());
        exit(EXIT_FAILURE);
    }
    if (config.convs[0].size != 4 && config.convs[0].size != 16) {
        ERROR("Invalid CRF model configuration - first convolution layer must be size 4 or 16. Got: %u", config.convs[0].size);
        exit(EXIT_FAILURE);
    }

    toml_free(config_toml);

    free(cpath);

    return config;
}

CRFModelConfig load_tx_model_config(const char *path) {
    FILE* fp;
    char errbuf[200];

    char *cpath = (char *)malloc(strlen(path) + 100);
    MALLOC_CHK(cpath);
    sprintf(cpath, "%s/config.toml", path);

    fp = fopen(cpath, "r");
    if (!fp) {
        ERROR("cannot open toml - %s: %s", cpath, strerror(errno));
        exit(EXIT_FAILURE);
    }

    toml_table_t *config_toml = toml_parse_file(fp, errbuf, sizeof(errbuf));
    fclose(fp);
    check_toml_table(config_toml);
    toml_table_t *model_toml = toml_table_in(config_toml, "model");
    check_toml_table(model_toml);

    CRFModelConfig config;

    parse_qscore_params(config, config_toml);

    const TxEncoderParams tx_encoder = parse_tx_encoder_params(config_toml);
    const EncoderUpsampleParams upsample = parse_encoder_upsample_params(config_toml);
    const CRFEncoderParams crf_encoder = parse_crf_encoder_params(config_toml);

    config.tx = new TxParams{tx_encoder, upsample, crf_encoder}; // todo: should delete this at some point
    if (crf_encoder.insize != tx_encoder.d_model) {
        WARNING("crf_encoder.insize: %d !=tx_encoder.d_model: %d", crf_encoder.insize, tx_encoder.d_model);
    }
    if (upsample.d_model != tx_encoder.d_model) {
        WARNING("upsample.d_model: %d !=tx_encoder.d_model: %d", upsample.d_model, tx_encoder.d_model);
    }

    toml_table_t *convs = toml_table_fallback(model_toml, {"encoder", "conv"});

    toml_array_t *sublayers = toml_array_in(convs, "sublayers");
    check_toml_array(sublayers);

    for (int i = 0; ; i++) {
        toml_table_t *segment = toml_table_at(sublayers, i);
        if (!segment) break;

        toml_datum_t type_dt = toml_string_in(segment, "type");
        check_toml_datum(type_dt);

        if (strcmp(type_dt.u.s, "convolution") != 0) {
            continue;
        }

        free(type_dt.u.s);

        const ConvParams conv = parse_conv_params(segment, false /* Tx models do not have swish clamp */);
        config.convs.push_back(conv);
        config.stride *= conv.stride;
    }

    // Recalculate the stride by accounting for upsampling / downsampling
    config.stride /= upsample.scale_factor;
    config.out_features = pow(crf_encoder.n_base, crf_encoder.state_len + 1);
    config.outsize = crf_encoder.outsize();

    config.state_len = config.tx->crf.state_len;
    config.num_features = config.convs.front().insize;

    config.signal_norm_params = parse_signal_normalisation_params(config_toml);

    // Force downstream issue (negative lstm size) if a tx model config is incorrectly
    // used to define an LSTM model. Incorrect use should be guarded against by using is_tx_model()
    config.lstm_size = -1;

    toml_free(config_toml);

    free(cpath);

    return config;
}

bool is_tx_model_config(const char *path) {
    FILE* fp;
    char errbuf[200];

    char *cpath = (char *)malloc(strlen(path) + 100);
    MALLOC_CHK(cpath);
    sprintf(cpath, "%s/config.toml", path);

    fp = fopen(cpath, "r");
    if (!fp) {
        ERROR("cannot open toml - %s: %s", cpath, strerror(errno));
        exit(EXIT_FAILURE);
    }
    toml_table_t *config_toml = toml_parse_file(fp, errbuf, sizeof(errbuf));
    fclose(fp);
    check_toml_table(config_toml);

    bool is_tx_model = toml_key_fallback(config_toml, {"model", "encoder", "transformer_encoder"});
    if (is_tx_model) {
        INFO("transformer model detected for config at: %s", cpath);
    }

    toml_free(config_toml);
    free(cpath);

    return is_tx_model;
}

/////////////////////////////////////////////////////////////////////////////////////// modbase

// Get an integer value from a toml::value asserting that it is within a closed interval.
// If no default is given then the key must exist in the toml::value.
int get_int_in_range(
    const toml_table_t *p,
    const char *key,
    int min_val,
    int max_val,
    std::optional<int> default_val
) {
    int val = 0;

    toml_datum_t datum = toml_int_in(p, key);
    if (datum.ok) {
        val = datum.u.i;
    } else if (default_val.has_value()) {
        val = default_val.value();
    } else {
        ERROR("%s", "could not find int");
    }
    
    if (val < min_val || val > max_val) {
        auto v = std::to_string(val);
        auto r = std::to_string(min_val) + " <= x <= " + std::to_string(max_val);
        ERROR("%s", "get_int_in_range fail");
    }
    return val;
}

ModelType model_type_from_string(char *model_type) {
    if (model_type == "conv_lstm") {
        return ModelType::CONV_LSTM_V1;
    }
    if (model_type == "conv_lstm_v2") {
        return ModelType::CONV_LSTM_V2;
    }
    if (model_type == "conv_lstm_v3") {
        return ModelType::CONV_LSTM_V3;
    }
    if (model_type == "conv_only" || model_type == "conv_v1") {
        return ModelType::CONV_V1;
    }
    return ModelType::UNKNOWN;
}

ModelType get_modbase_model_type(const const char *path) {
    FILE* fp;
    char errbuf[200];

    char *cpath = (char *)malloc(strlen(path) + 100);
    MALLOC_CHK(cpath);
    sprintf(cpath, "%s/config.toml", path);

    fp = fopen(cpath, "r");
    if (!fp) {
        ERROR("cannot open toml - %s: %s", cpath, strerror(errno));
        exit(EXIT_FAILURE);
    }

    toml_table_t *config_toml = toml_parse_file(fp, errbuf, sizeof(errbuf));
    fclose(fp);
    check_toml_table(config_toml);

    if (!toml_key_exists(config_toml, "general")) {
        return ModelType::UNKNOWN;
    }

    toml_datum_t type = toml_string_fallback(config_toml, {"general", "model"});
    check_toml_datum(type);

    auto ret = model_type_from_string(type.u.s);
    free(type.u.s);

    return ret;
}

bool is_modbase_model(const const char *path) {
    return get_modbase_model_type(path) != ModelType::UNKNOWN;
}

LinearParams parse_linear(toml_table_t *segment) {
    LinearParams p;

    toml_datum_t in_size = toml_int_in(segment, "in_features");
    check_toml_datum(in_size);

    toml_datum_t out_size = toml_int_in(segment, "out_features");
    check_toml_datum(out_size);

    p.in_size = in_size.u.i;
    p.out_size = out_size.u.i;

    return p;
}

LSTMParams parse_lstm(toml_table_t *segment) {
    LSTMParams p;

    toml_datum_t lstm_size = toml_int_in(segment, "size");
    check_toml_datum(lstm_size);

    toml_datum_t reverse = toml_bool_in(segment, "reverse");
    check_toml_datum(reverse);

    p.size = lstm_size.u.i;
    p.reverse = reverse.u.b;

    return p;
}

std::vector<LSTMParams> parse_lstms(const std::vector<toml_table_t *>& sublayers) {
    std::vector<LSTMParams> lstms;
    for (const auto& sublayer : sublayers) {
        if (sublayer_type(sublayer) == SublayerType::LSTM) {
            lstms.push_back(parse_lstm(sublayer));
        }
    }

    if (lstms.empty()) {
        ERROR("%s", "Modbase model config has no lstm layers");
    }
    if (lstms.front().reverse) {
        ERROR("%s", "Modbase model config first lstm layer must be forward");
    }
    for (size_t i = 0; i < (lstms.size() - 1); ++i) {
        if (lstms[i].size != lstms[i + 1].size) {
            ERROR("%s", "Modbase model config lstm layers unequal sizes");
        }
        if (lstms[i].reverse == lstms[i + 1].reverse) {
            ERROR("%s", "Modbase model config lstm layers must alternate direction");
        }
    }
    return lstms;
}

ConvParams parse_merge_conv(const std::vector<toml_table_t *>& sublayers) {
    if (sublayers.empty()) {
        ERROR("%s", "Modbase model config missing enoder sublayers");
    }
    const auto& front = sublayers.front();
    if (sublayer_type(front) != SublayerType::CONVOLUTION) {
        ERROR("%s", "Modbase model config missing enconder merge convolution");
    }
    return parse_conv_params(front, false);
}

std::vector<toml_table_t *> get_layers(const toml_table_t *config_toml, const char *key) {
    toml_table_t *encoder = toml_table_in(config_toml, key);
    check_toml_table(encoder);

    toml_array_t *layers = toml_array_in(encoder, "sublayers");
    check_toml_array(layers);

    std::vector<toml_table_t *> ret = {};
    for (int i = 0; ; i++) {
        toml_table_t *segment = toml_table_at(layers, i);
        if (!segment) break;
        ret.push_back(segment);
    }
    
    return ret;
}

EncoderUpsampleParams parse_linear_upsample(const toml_table_t *segment) {
    EncoderUpsampleParams params;

    toml_datum_t d_model = toml_int_in(segment, "size");
    check_toml_datum(d_model);
    toml_datum_t scale_factor = toml_int_in(segment, "scale_factor");
    check_toml_datum(scale_factor);

    params.d_model = d_model.u.i;
    params.scale_factor = scale_factor.u.i;

    return params;
}

ModulesParams parse_modules_params(const toml_table_t *config_toml) {
    ModulesParams m;
    m.sequence_convs = parse_convs(get_layers(config_toml, "sequence_encoder"));
    m.sequence_convs = parse_convs(get_layers(config_toml, "signal_encoder"));

    auto layers = get_layers(config_toml, "encoder");

    m.merge_conv = parse_merge_conv(layers);
    m.lstms = parse_lstms(layers);

    for (const auto& layer : layers) {
        if (sublayer_type(layer) == SublayerType::LINEAR) {
            m.linear = parse_linear(layer);
        }
        if (sublayer_type(layer) == SublayerType::UPSAMPLE) {
            m.upsample = parse_linear_upsample(layer);
        }
    }

    if (m.lstms.back().size != m.linear.in_size) {
        ERROR("%s", "Modbase model config lstm and linear size mismatch");
    }

    return m;
}

int stride_product(const std::vector<ConvParams>& cs) {
    return std::accumulate(cs.cbegin(), cs.cend(), 1,
                           [](const int s, const auto& c) { return s * c.stride; });
}

ModelGeneralParams::ModelGeneralParams(
    ModelType model_type_,
    int size_,
    int kmer_len_,
    int num_out_,
    int stride_,
    int sequence_stride_,
    std::optional<ModulesParams> modules_
) : model_type(model_type_),
    size(size_),
    kmer_len(kmer_len_),
    num_out(num_out_),
    stride(stride_),
    sequence_stride(sequence_stride_),
    modules(std::move(modules_)) 
{
    if (model_type == ModelType::UNKNOWN) {
        ERROR("%s", "general params: 'model type is unknown'");
    }
    if (size < 1 || kmer_len < 1 || num_out < 1 || stride < 1) {
        ERROR("%s", "general params: 'negative or zero value'.");
    }
    if (kmer_len % 2 != 1) {
        ERROR("%s", "general params: 'kmer_length is not odd'");
    }

    if (modules.has_value()) {
        if ((size != modules->lstms.front().size) ||
            (modules->lstms.front().size != modules->lstms.back().size)) {
            ERROR("%s", "Modbase model config lstm size mismatch");
        }
        if (stride != stride_product(modules->signal_convs)) {
            ERROR("%s", "Modbase model config signal convolution stride mismatch");
        }
        if (sequence_stride != stride_product(modules->sequence_convs)) {
            ERROR("%s", "Modbase model config sequence convolution stride mismatch");
        }
        if (num_out != modules->linear.out_size) {
            ERROR("%s", "Modbase model config linear and num_out mismatch");
        }
    }
}

ModelGeneralParams parse_general_params(const toml_table_t *config_toml) {
    const auto type_datum = toml_string_fallback(config_toml, {"general", "model"});
    check_toml_datum(type_datum);
    ModelType model_type = model_type_from_string(type_datum.u.s);
    free(type_datum.u.s);

    std::optional<ModulesParams> modules = model_type == ModelType::CONV_LSTM_V3 ? std::optional(parse_modules_params(config_toml)) : std::nullopt;
    const auto segment = toml_table_in(config_toml, "model_params");
    check_toml_table(segment);

    constexpr int MAX_SIZE = 4096;
    constexpr int MAX_KMER = 19;
    constexpr int MAX_FEATURES = 10;
    constexpr int MAX_STRIDE = 6;

    const auto size = get_int_in_range(segment, "size", 1, MAX_SIZE, REQUIRED);
    const auto kmer_len = get_int_in_range(segment, "kmer_len", 1, MAX_KMER, REQUIRED);
    const auto num_out = get_int_in_range(segment, "num_out", 1, MAX_FEATURES, REQUIRED);
    const auto stride = get_int_in_range(segment, "stride", 1, MAX_STRIDE, 3);
    const auto sequence_stride = get_int_in_range(segment, "sequence_stride", 1, MAX_STRIDE, stride);

    ModelGeneralParams params{model_type, size, kmer_len, num_out, stride, sequence_stride, modules};
    return params;
}

char get_canonical_base_name(const std::string& motif, size_t motif_offset) {
    if (motif.size() < motif_offset) {
        ERROR("%s", "mods params: 'invalid motif offset'.");
    }

    // Assert a canonical base is at motif[motif_offset]
    constexpr std::string_view canonical_bases = "ACGT";
    std::string motif_base = motif.substr(motif_offset, 1);
    if (canonical_bases.find(motif_base) == std::string::npos) {
        ERROR("%s", "mods params: 'invalid motif base'");
    }

    return motif_base[0];
}

bool validate_bam_tag_code(const std::string& bam_name) {
    // Check the supplied bam_name is a single character
    if (bam_name.size() == 1 && std::isalpha(static_cast<unsigned char>(bam_name[0]))) {
        return true;
    }

    // Check the supplied bam_name is a simple integer and if so, assume it's a CHEBI code.
    if (std::all_of(bam_name.begin(), bam_name.end(), [](const char& c) { return std::isdigit(static_cast<unsigned char>(c)); })) {
        return true;
    }
    return false;
}

ModificationParams::ModificationParams(
    std::vector<std::string> codes_,
    std::vector<std::string> long_names_,
    std::string motif_,
    const size_t motif_offset_
) : codes(std::move(codes_)),
    long_names(std::move(long_names_)),
    count(codes.size()),
    motif(std::move(motif_)),
    motif_offset(motif_offset_),
    base(get_canonical_base_name(motif, motif_offset)),
    base_id(BASE_IDS[base])
{
    if (codes.empty()) {
        ERROR("%s", "mods params: 'empty modifications.");
    }
    if (long_names.empty()) {
        ERROR("%s", "mods params: 'empty long names.");
    }
    if (codes.size() != long_names.size()) {
        ERROR("%s", "mods params: 'mods and names size mismatch.");
    }

    for (const auto& code : codes) {
        if (!validate_bam_tag_code(code)) {
            ERROR("%s", "mods params: 'invalid mod code ");
        }
    }
}

ModificationParams parse_modification_params(const toml_table_t *config_toml) {
    const auto& params = toml_table_in(config_toml, "modbases");
    check_toml_table(params);

    std::vector<std::string> codes;
    toml_array_t *mod_bases_arr = toml_array_in(params, "mod_bases");
    if (!mod_bases_arr) {
        toml_datum_t mod_bases_string = toml_string_in(params, "mod_bases");
        // style: mod_bases = "hm" - does not accept chebi codes
        for (const auto& mod_base : std::string(mod_bases_string.u.s)) {
            codes.push_back(std::string(1, mod_base));
        }
        free(mod_bases_string.u.s);
    } else {
        // style: mod_bases = [ "h", "m",]
        for (int i = 0; ; i++) {
            toml_datum_t mod_base_string = toml_string_at(mod_bases_arr, i);
            if (!mod_base_string.ok) break;
            codes.push_back(std::string(mod_base_string.u.s));
            free(mod_base_string.u.s);
        }
    }

    std::vector<std::string> long_names;
    long_names.reserve(codes.size());
    for (size_t i = 0; i < codes.size(); ++i) {
        auto key = "mod_long_names_" + std::to_string(i);
        toml_datum_t mod_long_names = toml_string_in(params, key.c_str());
        check_toml_datum(mod_long_names);
        long_names.push_back(std::string(mod_long_names.u.s));
        free(mod_long_names.u.s);
    }

    toml_datum_t motif = toml_string_in(params, "motif");
    check_toml_datum(motif);
    auto motif_string = std::string(motif.u.s);
    const auto motif_offset = static_cast<size_t>(get_int_in_range(params, "motif_offset", 0, int(motif_string.size()), REQUIRED));
    free(motif.u.s);

    return ModificationParams{std::move(codes), std::move(long_names), motif_string, motif_offset};
}

ContextParams::ContextParams(
    int64_t samples_before_,
    int64_t samples_after_,
    int64_t chunk_size_,
    int bases_before_,
    int bases_after_,
    bool reverse_,
    bool base_start_justify_
) : samples_before(samples_before_),
    samples_after(samples_after_),
    samples(samples_before + samples_after),
    chunk_size(chunk_size_),
    bases_before(bases_before_),
    bases_after(bases_after_),
    kmer_len(bases_before_ + bases_after_ + 1),
    reverse(reverse_),
    base_start_justify(base_start_justify_)
{
    if (samples_before < 0 || samples_after < 0) {
        ERROR("%s", "context params: 'negative context samples'.");
    }
    if (chunk_size < samples) {
        ERROR("%s", "mods params: 'context params: 'chunk size < context size'.");
    }
    if (bases_before < 1 || bases_after < 1) {
        ERROR("%s", "mods params: 'context params: 'negative or zero context bases'.");
    }
}

ContextParams parse_context_params(const toml_table_t *config_toml) {
    const auto& params = toml_table_in(config_toml, "modbases");
    check_toml_table(params);

    const int context_before = get_int_in_range(params, "chunk_context_0", 0, 4096, REQUIRED);
    const int context_after = get_int_in_range(params, "chunk_context_1", 1, 4096, REQUIRED);

    constexpr int MAX_CHUNK_SIZE = 102400;
    const int min_chunk_size = context_before + context_after;
    const int chunk_size = get_int_in_range(params, "chunk_size", min_chunk_size, MAX_CHUNK_SIZE, min_chunk_size);

    const auto bases_before = get_int_in_range(params, "kmer_context_bases_0", 0, 9, REQUIRED);
    const auto bases_after = get_int_in_range(params, "kmer_context_bases_1", 0, 9, REQUIRED);

    toml_datum_t reverse_datum = toml_bool_in(params, "reverse_signal");
    const auto reverse = reverse_datum.ok ? reverse_datum.u.b : false;

    toml_datum_t justify_datum = toml_bool_in(params, "base_start_justify");
    const auto base_start_justify = justify_datum.ok ? justify_datum.u.b : false;

    return ContextParams(context_before, context_after, chunk_size, bases_before, bases_after,
                         reverse, base_start_justify);
}
