#include "toml.h"
#include "error.h"
#include "model_config.h"

#include <unordered_map>

enum SublayerType { CLAMP, CONVOLUTION, LINEAR, LINEAR_CRF_ENCODER, LSTM, PERMUTE, UNRECOGNISED };
static const std::unordered_map<std::string, SublayerType> sublayer_map = {
    {"clamp", SublayerType::CLAMP},   {"convolution", SublayerType::CONVOLUTION},
    {"linear", SublayerType::LINEAR}, {"linearcrfencoder", SublayerType::LINEAR_CRF_ENCODER},
    {"lstm", SublayerType::LSTM},     {"permute", SublayerType::PERMUTE},
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

toml_table_t *toml_table_fallback_prereq(toml_table_t *config_toml, std::vector<std::string> fallbacks) {
    toml_table_t *ret = config_toml;
    for (size_t i = 0; i < fallbacks.size()-1; ++i) {
        const char *fallback = fallbacks[i].c_str();
        ret = toml_table_in(ret, fallback);
        check_toml_table(ret);
    }
    return ret;
}

toml_table_t *toml_table_fallback(toml_table_t *config_toml, std::vector<std::string> fallbacks) {
    toml_table_t *prereq = toml_table_fallback_prereq(config_toml, fallbacks);
    const char *fallback = fallbacks.back().c_str();
    toml_table_t *ret = toml_table_in(prereq, fallback);
    check_toml_table(ret);
    return ret;
}

toml_array_t *toml_array_fallback(toml_table_t *config_toml, std::vector<std::string> fallbacks) {
    toml_table_t *prereq = toml_table_fallback_prereq(config_toml, fallbacks);
    const char *fallback = fallbacks.back().c_str();
    toml_array_t *ret = toml_array_in(prereq, fallback);
    check_toml_array(ret);
    return ret;
}

toml_datum_t toml_int_fallback(toml_table_t *config_toml, std::vector<std::string> fallbacks) {
    toml_table_t *prereq = toml_table_fallback_prereq(config_toml, fallbacks);
    const char *fallback = fallbacks.back().c_str();
    toml_datum_t ret = toml_int_in(prereq, fallback);
    check_toml_datum(ret);
    return ret;
}

toml_datum_t toml_double_fallback(toml_table_t *config_toml, std::vector<std::string> fallbacks) {
    toml_table_t *prereq = toml_table_fallback_prereq(config_toml, fallbacks);
    const char *fallback = fallbacks.back().c_str();
    toml_datum_t ret = toml_double_in(prereq, fallback);
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