#include "toml.h"
#include "error.h"
#include "model_config.h"

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

toml_table_t *toml_table_fallback(const toml_table_t *config_toml, std::vector<std::string> fallbacks) {
    for (size_t i = 0; i < fallbacks.size(); ++i) {
        char *fallback = fallbacks[i].c_str();
        if (toml_key_exists(config_toml, fallback)) {
            toml_table_t *ret = toml_table_in(config_toml, fallback);
            check_toml_table(ret);
            return ret;
        }
    }
    ERROR("%s", "could not find table from list of fallbacks in config toml");
    exit(EXIT_FAILURE);
}

toml_array_t *toml_array_fallback(const toml_table_t *config_toml, std::vector<std::string> fallbacks) {
    for (size_t i = 0; i < fallbacks.size(); ++i) {
        char *fallback = fallbacks[i].c_str();
        if (toml_key_exists(config_toml, fallback)) {
            toml_array_t *ret = toml_array_in(config_toml, fallback);
            check_toml_array(ret);
            return ret;
        }
    }
    ERROR("%s", "could not find array from list of fallbacks in config toml");
    exit(EXIT_FAILURE);
}

toml_datum_t *toml_int_fallback(const toml_table_t *config_toml, std::vector<std::string> fallbacks) {
    for (size_t i = 0; i < fallbacks.size(); ++i) {
        char *fallback = fallbacks[i].c_str();
        if (toml_key_exists(config_toml, fallback)) {
            toml_datum_t *ret = toml_int_in(config_toml, fallback);
            check_toml_datum(ret);
            return ret;
        }
    }
    ERROR("%s", "could not find int from list of fallbacks in config toml");
    exit(EXIT_FAILURE);
}

toml_datum_t *toml_double_fallback(const toml_table_t *config_toml, std::vector<std::string> fallbacks) {
    for (size_t i = 0; i < fallbacks.size(); ++i) {
        char *fallback = fallbacks[i].c_str();
        if (toml_key_exists(config_toml, fallback)) {
            toml_datum_t *ret = toml_double_in(config_toml, fallback);
            check_toml_datum(ret);
            return ret;
        }
    }
    ERROR("%s", "could not find double from list of fallbacks in config toml");
    exit(EXIT_FAILURE);
}

void parse_qscore_params(CRFModelConfig &config, const toml_table_t *config_toml) {
    if (toml_key_exists(config_toml, "qscore")) {
        toml_table_t *qscore = toml_table_in(config_toml, "qscore");
        check_toml_table(qscore);
        toml_datum_t qbias = toml_double_in(qscore, "bias");
        check_toml_datum(qbias);
        toml_datum_t qscale = toml_double_in(qscore, "scale");
        check_toml_datum(qscale);

        config.qbias = (float)qbias.u.d;
        config.qscale = (float)qscale.u.d;

        if (toml_key_exists(qscore, "mean_qscore_start_pos")) {
            toml_datum_t mean_qscore_start_pos = toml_int_in(qscore, "mean_qscore_start_pos");
            check_toml_datum(mean_qscore_start_pos);
            config.mean_qscore_start_pos =  mean_qscore_start_pos;
        } else {
            ERROR("%s", "mean_qscore_start_pos not found in config toml");
        }
    } else {
        // no qscore calibration found
    }
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

    return params;
}

// Parse a the config.toml to resolve the scaling parameters.
SignalNormalisationParams parse_signal_normalisation_params(const toml::value &config_toml) {
    SignalNormalisationParams params;

    // scaling.strategy introduced with v4.3 models
    if (config_toml.contains("scaling")) {
        const auto &scaling = toml::find(config_toml, "scaling");
        params.strategy =
                scaling_strategy_from_string(toml::find<std::string>(scaling, "strategy"));
    }

    if (config_toml.contains("normalisation")) {
        const auto &norm = toml::find(config_toml, "normalisation");
        params.quantile.quantile_a = toml::find<float>(norm, "quantile_a");
        params.quantile.quantile_b = toml::find<float>(norm, "quantile_b");
        params.quantile.shift_multiplier = toml::find<float>(norm, "shift_multiplier");
        params.quantile.scale_multiplier = toml::find<float>(norm, "scale_multiplier");

        if (params.strategy != ScalingStrategy::QUANTILE) {
            spdlog::warn(
                    "Normalisation parameters are only used when `scaling.strategy = quantile`");
        }
    }

    if (config_toml.contains("standardisation")) {
        const auto &norm = toml::find(config_toml, "standardisation");
        params.standarisation.standardise = toml::find<int>(norm, "standardise") > 0;
        if (params.standarisation.standardise) {
            params.standarisation.mean = toml::find<float>(norm, "mean");
            params.standarisation.stdev = toml::find<float>(norm, "stdev");
        }

        if (params.standarisation.standardise && params.strategy != ScalingStrategy::PA) {
            throw std::runtime_error(
                    "Signal standardisation is implemented only for `scaling.strategy = pa`");
        }

        if (params.standarisation.stdev <= 0.0f) {
            throw std::runtime_error(
                    "Config error: `standardisation.stdev` must be greater than 0, got: " +
                    std::to_string(params.standarisation.stdev));
        }
    }

    return params;
}

CRFModelConfig load_crf_model_config(char *path) {
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
    config.qscale = 1.0f;
    config.qbias = 0.0f;

    parse_qscore_params(&config, config_toml);

    config.conv = 4;
    config.insize = 0;
    config.stride = 1;
    config.bias = true;
    config.clamp = false;
    config.decomposition = false;

    // The encoder scale only appears in pre-v4 models.  In v4 models
    // the value of 1 is used.
    config.scale = 1.0f;

    toml_table_t *input = toml_table_in(config_toml, "input");
    check_toml_table(input);
    toml_datum_t num_features = toml_int_in(input, "features");
    check_toml_datum(num_features);
    config.num_features = num_features.u.i;

    toml_table_t *encoder = toml_table_in(config_toml, "encoder");
    check_toml_table(encoder);
    if (toml_key_exists(encoder, "type")) {
        // v4-type model
        toml_array_t *sublayers = toml_array_in(encoder, "sublayers");
        check_toml_array(sublayers);

        for (int i = 0; ; i++) {
            toml_table_t *segment = toml_table_at(sublayers, i);
            if (!segment) break;

            toml_datum_t type_dt = toml_string_in(segment, "type");
            check_toml_datum(type_dt);
            char *type = type_dt.u.s;

            if (strcmp(type, "convolution") == 0) {
                // Overall stride is the product of all conv layers' strides.
                toml_datum_t stride = toml_int_in(segment, "stride");
                check_toml_datum(stride);
                config.stride *= stride.u.i;
            } else if (strcmp(type, "lstm") == 0) {
                toml_datum_t insize = toml_int_in(segment, "insize");
                check_toml_datum(insize);
                config.insize = insize.u.i;
            } else if (strcmp(type, "linear") == 0) {
                // Specifying out_features implies a decomposition of the linear layer matrix
                // multiply with a bottleneck before the final feature size.
                if (toml_key_exists(segment, "out_features")) {
                    toml_datum_t out_features = toml_int_in(segment, "out_features");
                    check_toml_datum(out_features);
                    config.out_features = out_features.u.i;
                    config.decomposition = true;
                } else {
                    config.decomposition = false;
                }
            } else if (strcmp(type, "clamp") == 0) {
                config.clamp = true;
            } else if (strcmp(type, "linearcrfencoder") == 0) {
                toml_datum_t blank_score = toml_double_in(segment, "blank_score");
                check_toml_datum(blank_score);
                config.blank_score = (float)blank_score.u.d;
            }

            free(type);
        }

        config.conv = 16;
        config.bias = config.insize > 128;
    } else {
        // pre-v4 model
        toml_datum_t stride = toml_int_in(encoder, "stride");
        check_toml_datum(stride);
        config.stride = stride.u.i;

        toml_datum_t features = toml_int_in(encoder, "features");
        check_toml_datum(features);
        config.insize = features.u.i;

        toml_datum_t blank_score = toml_double_in(encoder, "blank_score");
        check_toml_datum(blank_score);
        config.blank_score = (float)blank_score.u.d;

        toml_datum_t scale = toml_double_in(encoder, "scale");
        check_toml_datum(scale);
        config.scale = (float)scale.u.d;

        if (toml_key_exists(encoder, "first_conv_size")) {
            toml_datum_t conv = toml_int_in(encoder, "first_conv_size");
            check_toml_datum(conv);
            config.conv = conv.u.i;
        }
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

    toml_free(config_toml);

    free(cpath);

    return config;
}

TxEncoderParams parse_tx_encoder_params(const toml_table_t *cfg) {
    const toml_table_t *enc = toml_table_fallback(cfg, {"model", "encoder", "transformer_encoder"});
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
    for (int i = 0; ; i++; i < 2) {
        toml_datum_t *e = toml_int_at(attn_window, i);
        if (!e.ok) break;
        params.attn_window.push_back(e);
    }

    return params;
}

EncoderUpsampleParams parse_encoder_upsample_params(const toml_table_t *cfg) {
    const toml_table_t *ups = toml_table_fallback(cfg, {"model", "encoder", "upsample"});
    EncoderUpsampleParams params;

    toml_datum_t d_model = toml_int_in(ups, "d_model");
    check_toml_datum(d_model);
    toml_datum_t scale_factor = toml_int_in(ups, "scale_factor");
    check_toml_datum(scale_factor);

    params.d_model = d_model.u.i;
    params.scale_factor = scale_factor.u.i;

    return params;
}

CRFEncoderParams parse_crf_encoder_params(const toml_table_t *cfg) {
    const toml_table_t *crf = toml_table_fallback(cfg, {"model", "encoder", "crf"});
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
        toml_datum_t *e = toml_int_at(permute, i);
        if (!e.ok) break;
        params.permute.push_back(e);
    }

    return params;
}

CRFModelConfig load_tx_model_config(char *path) {
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

    parse_qscore_params(&config, config_toml);

    const TxEncoderParams tx_encoder = parse_tx_encoder_params(config_toml);
    const EncoderUpsampleParams upsample = parse_encoder_upsample_params(config_toml);
    const CRFEncoderParams crf_encoder = parse_crf_encoder_params(config_toml);

    config.tx = TxParams{tx_encoder, upsample, crf_encoder};
    config.tx->check();

    toml_table_t *convs = toml_table_fallback(model_toml, {"encoder", "conv"});

    toml_array_t *sublayers = toml_array_in(encoder, "sublayers");
    check_toml_array(sublayers);

    for (int i = 0; ; i++) {
        toml_table_t *segment = toml_table_at(sublayers, i);
        if (!segment) break;

        toml_datum_t type_dt = toml_string_in(segment, "type");
        check_toml_datum(type_dt);
        char *type = type_dt.u.s;

        if (strcmp(type, "convolution") != 0) {
            continue;
        }

        const ConvParams conv = parse_conv_params(segment, false /* Tx models do not have swish clamp */);
        config.convs.push_back(conv);
        config.stride *= conv.stride;
    }
    // Recalculate the stride by accounting for upsampling / downsampling
    config.stride /= upsample.scale_factor;
    config.out_features = crf_encoder.out_features();
    config.outsize = crf_encoder.outsize();

    config.state_len = config.tx->crf.state_len;
    config.num_features = config.convs.front().insize;

    config.signal_norm_params = parse_signal_normalisation_params(config_toml, model_name);

    // Force downstream issue (negative lstm size) if a tx model config is incorrectly
    // used to define an LSTM model. Incorrect use should be guarded against by using is_tx_model()
    config.lstm_size = -1;

    return config;
}