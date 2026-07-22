#include "toml.h"
#include "error.h"
#include "model_config.h"
#include "tensor_chunk_utils.h"

#include <numeric>
#include <algorithm>
#include <torch/torch.h>

static const std::vector<int> BASE_IDS = []() {
    std::vector<int> base_ids(256, -1);
    base_ids['A'] = 0;
    base_ids['C'] = 1;
    base_ids['G'] = 2;
    base_ids['T'] = 3;
    return base_ids;
}();

static const std::string ERR_STR = "Invalid modbase model parameter in ";

// Indicates that a value has no default and is therefore required
static constexpr std_optional<int> REQUIRED = STD_NULLOPT;

enum SublayerType { CLAMP, CONVOLUTION, FLSTM_SOFTOUT, LINEAR, LINEAR_CRF_ENCODER, LSTM, PERMUTE, UPSAMPLE, UNRECOGNISED };
static const std::unordered_map<std::string, SublayerType> sublayer_map = {
    {"clamp", SublayerType::CLAMP},
    {"convolution", SublayerType::CONVOLUTION},
    {"flstm_softout", SublayerType::FLSTM_SOFTOUT},
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
conv_params_t parse_conv_params(const toml_table_t *segment, bool clamp) {
    conv_params_t params;
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
std::vector<conv_params_t> parse_convs(const std::vector<toml_table_t *> &sublayers) {
    std::vector<conv_params_t> convs;
    for (size_t i = 0; i < sublayers.size(); ++i) {
        // If the sublayer after a convolution is a clamp, the activation function may have
        // a fused implementation
        if (sublayer_type(sublayers.at(i)) == SublayerType::CONVOLUTION) {
            const bool has_clamp_next = ((i + 1) < sublayers.size()) &&sublayer_type(sublayers.at(i + 1)) == SublayerType::CLAMP;
            conv_params_t conv = parse_conv_params(sublayers.at(i), has_clamp_next);
            convs.push_back(conv);
        }
    }
    return convs;
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
signal_norm_params_t parse_signal_normalisation_params(const toml_table_t *config_toml) {
    signal_norm_params_t params;

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

tx_encoder_params_t parse_tx_encoder_params(toml_table_t *cfg) {
    toml_table_t *enc = toml_table_fallback(cfg, {"model", "encoder", "transformer_encoder"});
    tx_encoder_params_t params;

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

encoder_upsample_params_t parse_encoder_upsample_params(toml_table_t *cfg) {
    toml_table_t *ups = toml_table_fallback(cfg, {"model", "encoder", "upsample"});
    encoder_upsample_params_t params;

    toml_datum_t d_model = toml_int_in(ups, "d_model");
    check_toml_datum(d_model);
    toml_datum_t scale_factor = toml_int_in(ups, "scale_factor");
    check_toml_datum(scale_factor);

    params.d_model = d_model.u.i;
    params.scale_factor = scale_factor.u.i;

    return params;
}

crf_encoder_params_t parse_crf_encoder_params(toml_table_t *cfg) {
    toml_table_t *crf = toml_table_fallback(cfg, {"model", "encoder", "crf"});
    crf_encoder_params_t params;

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

// --- simplified v5.0.0+ loader ----------------------------------------------------------------

static toml_table_t *open_config_toml(const char *path) {
    char errbuf[200];
    char *cpath = (char *)malloc(strlen(path) + 100);
    MALLOC_CHK(cpath);
    sprintf(cpath, "%s/config.toml", path);
    FILE *fp = fopen(cpath, "r");
    if (!fp) {
        ERROR("cannot open toml - %s: %s", cpath, strerror(errno));
        exit(EXIT_FAILURE);
    }
    toml_table_t *config_toml = toml_parse_file(fp, errbuf, sizeof(errbuf));
    fclose(fp);
    free(cpath);
    check_toml_table(config_toml);
    return config_toml;
}

// Read [qscore] scale/bias into the simplified config (no defaulting noise).
static void load_qscore(model_config_t &cfg, toml_table_t *config_toml) {
    if (!toml_key_exists(config_toml, "qscore")) return;
    toml_table_t *qscore = toml_table_in(config_toml, "qscore");
    check_toml_table(qscore);
    toml_datum_t qbias = toml_double_in(qscore, "bias");
    check_toml_datum(qbias);
    toml_datum_t qscale = toml_double_in(qscore, "scale");
    check_toml_datum(qscale);
    cfg.qbias = qbias.u.d;
    cfg.qscale = qscale.u.d;
}

static void load_basecaller(model_config_t &cfg, toml_table_t *config_toml) {
    toml_table_t *basecaller = toml_table_in(config_toml, "basecaller");
    if (!basecaller) return;
    toml_datum_t chunksize = toml_int_in(basecaller, "chunksize");
    if (chunksize.ok) cfg.chunk_size = (int)chunksize.u.i;
    toml_datum_t overlap = toml_int_in(basecaller, "overlap");
    if (overlap.ok) cfg.overlap = (int)overlap.u.i;
}

static void load_lstm_family(model_config_t &cfg, toml_table_t *config_toml) {
    toml_table_t *input = toml_table_in(config_toml, "input");
    check_toml_table(input);
    toml_datum_t num_features = toml_int_in(input, "features");
    check_toml_datum(num_features);
    cfg.num_features = num_features.u.i;

    toml_table_t *encoder = toml_table_in(config_toml, "encoder");
    check_toml_table(encoder);
    if (!toml_key_exists(encoder, "type")) {
        ERROR("%s", "pre-v4 model configs are not supported (require models >= v5.0.0)");
        exit(EXIT_FAILURE);
    }

    toml_array_t *_sublayers = toml_array_in(encoder, "sublayers");
    check_toml_array(_sublayers);
    std::vector<toml_table_t *> sublayers;
    for (int i = 0;; i++) {
        toml_table_t *segment = toml_table_at(_sublayers, i);
        if (!segment) break;
        sublayers.push_back(segment);
    }

    cfg.bias = false;
    cfg.clamp = has_clamp(sublayers);
    cfg.convs = parse_convs(sublayers);
    for (const auto &cv : cfg.convs) cfg.stride *= cv.stride;
    cfg.lstm_size = cfg.convs.back().size;

    cfg.lstm_layers = 0;
    for (const auto &segment : sublayers) {
        const auto type = sublayer_type(segment);
        if (type == SublayerType::LSTM) {
            cfg.lstm_layers++;
        } else if (type == SublayerType::FLSTM_SOFTOUT) {
            cfg.lstm_layers++;
            toml_datum_t inner_dim = toml_int_in(segment, "inner_dim");
            check_toml_datum(inner_dim);
            cfg.lstm_inner_dim = inner_dim.u.i;
        } else if (type == SublayerType::LINEAR) {
            toml_datum_t out_features = toml_int_in(segment, "out_features");
            check_toml_datum(out_features);
            cfg.out_features = out_features.u.i;
            cfg.has_out_features = true;
            toml_datum_t bias_d = toml_bool_in(segment, "bias");
            cfg.bias = bias_d.ok ? (bool)bias_d.u.b : (cfg.lstm_size > 128);
        } else if (type == SublayerType::LINEAR_CRF_ENCODER) {
            toml_datum_t activation = toml_string_in(segment, "activation");
            if (activation.ok) {
                cfg.crf_encoder_has_tanh = (strcmp(activation.u.s, "tanh") == 0);
                free(activation.u.s);
            }
        }
    }

    toml_table_t *global_norm = toml_table_in(config_toml, "global_norm");
    check_toml_table(global_norm);
    toml_datum_t state_len = toml_int_in(global_norm, "state_len");
    check_toml_datum(state_len);
    cfg.state_len = state_len.u.i;
    cfg.outsize = pow(4, cfg.state_len) * 4;

    if (cfg.convs.size() != 3) {
        ERROR("Expected 3 convolution layers but found: %lu", cfg.convs.size());
        exit(EXIT_FAILURE);
    }
    if (cfg.convs[0].size != 4 && cfg.convs[0].size != 16) {
        ERROR("Invalid CRF model configuration - first convolution layer must be size 4 or 16. Got: %u", cfg.convs[0].size);
        exit(EXIT_FAILURE);
    }

    cfg.family = (cfg.lstm_inner_dim >= 0) ? MODEL_FAMILY_FLSTM : MODEL_FAMILY_LSTM;
}

static void load_tx_family(model_config_t &cfg, toml_table_t *config_toml) {
    toml_table_t *model_toml = toml_table_in(config_toml, "model");
    check_toml_table(model_toml);

    cfg.tx.tx = parse_tx_encoder_params(config_toml);
    cfg.tx.upsample = parse_encoder_upsample_params(config_toml);
    cfg.tx.crf = parse_crf_encoder_params(config_toml);

    toml_table_t *convs = toml_table_fallback(model_toml, {"encoder", "conv"});
    toml_array_t *sublayers = toml_array_in(convs, "sublayers");
    check_toml_array(sublayers);
    for (int i = 0;; i++) {
        toml_table_t *segment = toml_table_at(sublayers, i);
        if (!segment) break;
        toml_datum_t type_dt = toml_string_in(segment, "type");
        check_toml_datum(type_dt);
        bool is_conv = (strcmp(type_dt.u.s, "convolution") == 0);
        free(type_dt.u.s);
        if (!is_conv) continue;
        const conv_params_t conv = parse_conv_params(segment, false); // TX has no swish clamp
        cfg.convs.push_back(conv);
        cfg.stride *= conv.stride;
    }

    cfg.stride /= cfg.tx.upsample.scale_factor;
    cfg.out_features = pow(cfg.tx.crf.n_base, cfg.tx.crf.state_len + 1);
    cfg.outsize = crf_outsize(cfg.tx.crf);
    cfg.state_len = cfg.tx.crf.state_len;
    cfg.num_features = cfg.convs.front().insize;
    cfg.lstm_size = -1; // force a downstream error if misused as an LSTM model
    cfg.family = MODEL_FAMILY_TX;
}

model_config_t load_model_config(const char *path) {
    model_config_t cfg;
    cfg.model_path = std::string(path);
    cfg.sample_type = get_sample_type_from_model_name(cfg.model_path);

    toml_table_t *config_toml = open_config_toml(path);
    load_qscore(cfg, config_toml);
    cfg.signal_norm_params = parse_signal_normalisation_params(config_toml);
    if (is_tx_model_config(path)) {
        load_tx_family(cfg, config_toml);
    } else {
        load_lstm_family(cfg, config_toml);
    }
    load_basecaller(cfg, config_toml);
    toml_free(config_toml);
    return cfg;
}


/////////////////////////////////////////////////////////////////////////////////////// modbase

// Get an integer value from a toml::value asserting that it is within a closed interval.
// If no default is given then the key must exist in the toml::value.
int get_int_in_range(
    const toml_table_t *p,
    const char *key,
    int min_val,
    int max_val,
    std_optional<int> default_val
) {
    int val = 0;

    toml_datum_t datum = toml_int_in(p, key);
    if (datum.ok) {
        val = datum.u.i;
    } else if (default_val) {
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

ModelType model_type_from_string(char *_model_type) {
    auto model_type = std::string(_model_type
    );
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

ModelType get_modbase_model_type(const char *path) {
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

    toml_free(config_toml);
    free(cpath);

    return ret;
}

bool is_modbase_model(const char *path) {
    return get_modbase_model_type(path) != ModelType::UNKNOWN;
}

linear_params_t parse_linear(toml_table_t *segment) {
    linear_params_t p;

    toml_datum_t in_size = toml_int_in(segment, "in_features");
    check_toml_datum(in_size);

    toml_datum_t out_size = toml_int_in(segment, "out_features");
    check_toml_datum(out_size);

    p.in_size = in_size.u.i;
    p.out_size = out_size.u.i;

    return p;
}

lstm_config_params_t parse_lstm(toml_table_t *segment) {
    lstm_config_params_t p;

    toml_datum_t lstm_size = toml_int_in(segment, "size");
    check_toml_datum(lstm_size);

    // v3 modbase configs write `reverse = 0/1` (int); accept bool too for robustness.
    toml_datum_t reverse_i = toml_int_in(segment, "reverse");
    toml_datum_t reverse_b = toml_bool_in(segment, "reverse");
    if (reverse_i.ok) {
        p.reverse = (reverse_i.u.i != 0);
    } else if (reverse_b.ok) {
        p.reverse = reverse_b.u.b;
    } else {
        check_toml_datum(reverse_i);  // report the missing/invalid field
    }

    p.size = lstm_size.u.i;

    return p;
}

std::vector<lstm_config_params_t> parse_lstms(const std::vector<toml_table_t *>& sublayers) {
    std::vector<lstm_config_params_t> lstms;
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

conv_params_t parse_merge_conv(const std::vector<toml_table_t *>& sublayers) {
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

encoder_upsample_params_t parse_linear_upsample(const toml_table_t *segment) {
    encoder_upsample_params_t params;

    toml_datum_t d_model = toml_int_in(segment, "size");
    check_toml_datum(d_model);
    toml_datum_t scale_factor = toml_int_in(segment, "scale_factor");
    check_toml_datum(scale_factor);

    params.d_model = d_model.u.i;
    params.scale_factor = scale_factor.u.i;

    return params;
}

modules_params_t parse_modules_params(const toml_table_t *config_toml) {
    modules_params_t m;
    m.sequence_convs = parse_convs(get_layers(config_toml, "sequence_encoder"));
    m.signal_convs = parse_convs(get_layers(config_toml, "signal_encoder"));

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

int stride_product(const std::vector<conv_params_t>& cs) {
    return std::accumulate(cs.cbegin(), cs.cend(), 1,
                           [](const int s, const auto& c) { return s * c.stride; });
}

// --- modbase param free functions (behaviour split out of the now-inert structs) ---------------

int crf_outsize(const crf_encoder_params_t &p) {
    if (p.expand_blanks) return static_cast<int>(pow(p.n_base, p.state_len + 1));
    return (p.n_base + 1) * static_cast<int>(pow(p.n_base, p.state_len));
}
int crf_out_features(const crf_encoder_params_t &p) { return static_cast<int>(pow(p.n_base, p.state_len + 1)); }

int modules_stride_ratio(const modules_params_t &m) {
    // Signal is downsampled more than the sequence (kmer) input; the ratio is how many signal
    // samples map to one sequence position. e.g. RNA m6A v3: signal stride 6, sequence stride 1 -> 6.
    const int seq = stride_product(m.sequence_convs);
    const int sig = stride_product(m.signal_convs);
    assert(seq > 0 && sig >= seq && sig % seq == 0);
    return sig / seq;
}
int general_stride_ratio(const model_general_params_t &g) {
    return g.modules ? modules_stride_ratio(*g.modules) : 1;
}
int64_t context_normalise(int64_t v, int64_t stride) {
    const int64_t remainder = v % stride;
    return remainder == 0 ? v : v + stride - remainder;
}
context_params_t context_normalised(const context_params_t &c, int stride) {
    const int64_t sb = context_normalise(c.samples_before, stride);
    const int64_t sa = context_normalise(c.samples_after, stride);
    const int64_t cs = context_normalise(c.chunk_size, stride);
    return context_params_t{sb, sa, sb + sa, cs, c.bases_before, c.bases_after, c.kmer_len, c.reverse, c.base_start_justify};
}
bool is_chunked_input_model(const modbase_model_config_t &config) {
    return config.general.model_type == ModelType::CONV_LSTM_V2 ||
           config.general.model_type == ModelType::CONV_LSTM_V3;
}

// --- modbase context (was the ModBaseContext class + MotifMatcher) ------------------------------

// Expand an IUPAC nucleotide motif (e.g. "DRACH") into a POSIX ERE (e.g. "[AGT][AG]AC[ACT]").
// Plain ACGT pass through so exact motifs ("CG", "A") are unchanged. Sequences use the DNA
// alphabet (T, not U), matching the basecaller output.
static std::string iupac_motif_to_regex(const std::string &motif) {
    std::string re;
    for (char c : motif) {
        switch (c) {
            case 'A': case 'C': case 'G': case 'T': re += c; break;
            case 'U': re += 'T'; break;
            case 'R': re += "[AG]";   break;
            case 'Y': re += "[CT]";   break;
            case 'S': re += "[GC]";   break;
            case 'W': re += "[AT]";   break;
            case 'K': re += "[GT]";   break;
            case 'M': re += "[AC]";   break;
            case 'B': re += "[CGT]";  break;
            case 'D': re += "[AGT]";  break;
            case 'H': re += "[ACT]";  break;
            case 'V': re += "[ACG]";  break;
            case 'N': re += "[ACGT]"; break;
            default:  re += c;        break;  // pass through (already a regex char class, etc.)
        }
    }
    return re;
}

std::vector<size_t> modbase_motif_hits(const std::string &motif, size_t offset, const char *seq, size_t seqlen) {
    std::vector<size_t> context_hits;
    regex_t compiled;
    const std::string pattern = iupac_motif_to_regex(motif);
    if (regcomp(&compiled, pattern.c_str(), REG_EXTENDED) != 0) {
        return context_hits;
    }
    size_t pos = 0;
    while (pos < seqlen) {
        regmatch_t match;
        if (regexec(&compiled, seq + pos, 1, &match, 0) != 0) {
            break;
        }
        context_hits.push_back(pos + match.rm_so + offset);
        pos += match.rm_so + 1;
    }
    regfree(&compiled);
    return context_hits;
}

int mb_ubase_to_int(char c) { return 0b11 & ((c >> 2) ^ (c >> 1)); }

void mb_set_context(modbase_context_t &ctx, std::string motif, size_t offset) {
    if (motif.size() < 2) {
        return;  // empty motif, or just the canonical base
    }
    const int index = mb_ubase_to_int(motif.at(offset));
    ctx.motifs[index] = std::move(motif);
    ctx.offsets[index] = offset;
}

const std::string &mb_motif(const modbase_context_t &ctx, char base) {
    return ctx.motifs[mb_ubase_to_int(base)];
}

std::string mb_encode(const modbase_context_t &ctx) {
    std::ostringstream s;
    for (size_t i = 0; i < 4; ++i) {
        if (ctx.motifs[i].empty()) {
            s << '_';
        } else {
            auto m = ctx.motifs[i];
            m[ctx.offsets[i]] = 'X';
            s << m;
        }
        if (i < 3) s << ':';
    }
    return s.str();
}

bool mb_decode(modbase_context_t &ctx, const std::string &context_string) {
    std::vector<std::string> tokens;
    std::istringstream context_stream(context_string);
    std::string token;
    while (std::getline(context_stream, token, ':')) {
        tokens.push_back(token);
    }
    if (tokens.size() != 4) {
        return false;
    }
    const char *canonical = "ACGT";
    for (size_t i = 0; i < 4; ++i) {
        if (tokens[i] == "_") {
            ctx.motifs[i].clear();
            ctx.offsets[i] = 0;
        } else {
            auto x = tokens[i].find('X');
            if (x == std::string::npos) {
                return false;
            }
            ctx.motifs[i] = tokens[i];
            ctx.motifs[i][x] = canonical[i];
            ctx.offsets[i] = x;
        }
    }
    return true;
}

std::vector<bool> mb_get_sequence_mask(const modbase_context_t &ctx, const char *seq, size_t seqlen) {
    std::vector<bool> mask(seqlen, false);
    for (size_t i = 0; i < 4; ++i) {
        if (ctx.motifs[i].empty()) continue;
        for (auto hit : modbase_motif_hits(ctx.motifs[i], ctx.offsets[i], seq, seqlen)) {
            mask[hit] = true;
        }
    }
    return mask;
}

void mb_update_mask(const modbase_context_t &ctx,
                    std::vector<bool> &mask,
                    const std::string &sequence,
                    const std::vector<std::string> &modbase_alphabet,
                    const std::vector<uint8_t> &modbase_probs,
                    uint8_t threshold) {
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
            if (!ctx.motifs[mb_ubase_to_int(current_adjustment.cardinal_base)].empty()) {
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
        return;  // No bases to adjust, so nothing to do.
    }

    for (size_t base_idx = 0; base_idx < sequence.size(); ++base_idx) {
        bool requires_update = false;
        bool flag = false;
        for (const auto &adjustment : adjustments) {
            if (adjustment.cardinal_base == sequence[base_idx]) {
                requires_update = true;
                for (const auto channel_idx : adjustment.modified_channels) {
                    flag |= (modbase_probs[base_idx * num_channels + channel_idx] >= threshold);
                }
            }
        }
        if (requires_update) {
            mask[base_idx] = flag;
        }
    }
}

static void validate_general_params(const model_general_params_t &g) {
    if (g.model_type == ModelType::UNKNOWN) {
        ERROR("%s", "general params: 'model type is unknown'");
    }
    if (g.size < 1 || g.kmer_len < 1 || g.num_out < 1 || g.stride < 1) {
        ERROR("%s", "general params: 'negative or zero value'.");
    }
    if (g.kmer_len % 2 != 1) {
        ERROR("%s", "general params: 'kmer_length is not odd'");
    }
    if (g.modules) {
        const auto &m = *g.modules;
        if ((g.size != m.lstms.front().size) || (m.lstms.front().size != m.lstms.back().size)) {
            ERROR("%s", "Modbase model config lstm size mismatch");
        }
        if (g.stride != stride_product(m.signal_convs)) {
            ERROR("%s", "Modbase model config signal convolution stride mismatch");
        }
        if (g.sequence_stride != stride_product(m.sequence_convs)) {
            ERROR("%s", "Modbase model config sequence convolution stride mismatch");
        }
        if (g.num_out != m.linear.out_size) {
            ERROR("%s", "Modbase model config linear and num_out mismatch");
        }
    }
}

model_general_params_t parse_general_params(const toml_table_t *config_toml) {
    const auto type_datum = toml_string_fallback(config_toml, {"general", "model"});
    check_toml_datum(type_datum);
    ModelType model_type = model_type_from_string(type_datum.u.s);
    free(type_datum.u.s);

    std_optional<modules_params_t> modules = model_type == ModelType::CONV_LSTM_V3 ? std_optional<modules_params_t>(parse_modules_params(config_toml)) : STD_NULLOPT;
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

    model_general_params_t params{model_type, size, kmer_len, num_out, stride, sequence_stride, modules};
    validate_general_params(params);
    return params;
}

char get_canonical_base_name(const std::string& motif, size_t motif_offset) {
    if (motif.size() < motif_offset) {
        ERROR("%s", "mods params: 'invalid motif offset'.");
    }

    // Assert a canonical base is at motif[motif_offset]
    const std::string canonical_bases = "ACGT";
    std::string motif_base = motif.substr(motif_offset, 1);
    if (canonical_bases.find(motif_base) == std::string::npos) {
        ERROR("%s", "mods params: 'invalid motif base'");
    }

    return motif_base[0];
}

static bool validate_bam_tag_code(const std::string& bam_name) {
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

static void validate_modification_params(const modification_params_t &p) {
    if (p.codes.empty()) {
        ERROR("%s", "mods params: 'empty modifications.");
    }
    if (p.long_names.empty()) {
        ERROR("%s", "mods params: 'empty long names.");
    }
    if (p.codes.size() != p.long_names.size()) {
        ERROR("%s", "mods params: 'mods and names size mismatch.");
    }
    for (const auto &code : p.codes) {
        if (!validate_bam_tag_code(code)) {
            ERROR("%s", "mods params: 'invalid mod code ");
        }
    }
}

modification_params_t parse_modification_params(const toml_table_t *config_toml) {
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

    const char base = get_canonical_base_name(motif_string, motif_offset);
    const size_t count = codes.size();
    modification_params_t mp{std::move(codes), std::move(long_names), count, motif_string, motif_offset,
                          base, BASE_IDS[base], {}};
    validate_modification_params(mp);
    return mp;
}

static void validate_context_params(const context_params_t &c) {
    if (c.samples_before < 0 || c.samples_after < 0) {
        ERROR("%s", "context params: 'negative context samples'.");
    }
    if (c.chunk_size < c.samples) {
        ERROR("%s", "mods params: 'context params: 'chunk size < context size'.");
    }
    if (c.bases_before < 1 || c.bases_after < 1) {
        ERROR("%s", "mods params: 'context params: 'negative or zero context bases'.");
    }
}

context_params_t parse_context_params(const toml_table_t *config_toml) {
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

    context_params_t cp{context_before, context_after, context_before + context_after, chunk_size,
                     bases_before, bases_after, bases_before + bases_after + 1,
                     reverse, base_start_justify};
    validate_context_params(cp);
    return cp;
}

refinement_params_t parse_refinement_params(const toml_table_t *config_toml) {
    if (!toml_key_exists(config_toml, "refinement")) {
        return refinement_params_t{};
    }

    const auto segment = toml_table_in(config_toml, "refinement");
    check_toml_table(segment);

    const auto do_rough_rescale = toml_int_in(segment, "refine_do_rough_rescale");
    if (do_rough_rescale.u.i != 1) {
        return refinement_params_t{};
    }

    const int center_index = get_int_in_range(segment, "refine_kmer_center_idx", 0, 19, REQUIRED);
    return refinement_params_t{true, static_cast<size_t>(center_index)};
}

std::vector<float> load_kmer_refinement_levels(const modbase_model_config_t& config) {
    std::vector<float> levels;
    if (!config.refine.do_rough_rescale) {
        return levels;
    }

    std::vector<torch::Tensor> tensors = load_tensors(config.model_path, {"refine_kmer_levels.tensor"});
    if (tensors.empty()) {
        ERROR("%s", "failed to load modbase refinement tensors");
        exit(EXIT_FAILURE);
    }
    auto& t = tensors.front();
    t.contiguous();
    levels.reserve(t.numel());
    std::copy(t.data_ptr<float>(), t.data_ptr<float>() + t.numel(), std::back_inserter(levels));
    return levels;
}

modbase_model_config_t load_modbase_model_config(const char *path) {
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
    

    modbase_model_config_t ret{
        path, parse_general_params(config_toml), parse_modification_params(config_toml),
        parse_context_params(config_toml), parse_refinement_params(config_toml)
    };

    // v2 models normalise the context to the model stride (was done in the old ctor).
    if (ret.general.model_type == ModelType::CONV_LSTM_V2) {
        ret.context = context_normalised(ret.context, ret.general.stride);
    }
    // Kmer length is duplicated in modbase configs - check they match.
    if (ret.general.kmer_len != ret.context.kmer_len) {
        ERROR("%s", "config: 'inconsistent kmer_len'");
    }

    ret.mods.kmer_levels = load_kmer_refinement_levels(ret);

    toml_free(config_toml);
    free(cpath);

    return ret;
}
SampleType get_sample_type_from_model_name(const std::string& model_name) {
    if (model_name.find("rna004") != std::string::npos) {
        return SampleType::RNA004;
    } else if (model_name.find("rna002") != std::string::npos) {
        return SampleType::RNA002;
    } else if (model_name.find("dna") != std::string::npos) {
        return SampleType::DNA;
    } else {
        return SampleType::UNKNOWN;
    }
}

bool is_rna(SampleType sample_type) {
    return (sample_type == SampleType::RNA002 || sample_type == SampleType::RNA004);
}

modbase_info_t get_modbase_info(std::vector<modbase_model_config_t>& base_mod_params) {
    struct ModelInfo {
        std::vector<std::string> long_names;
        std::vector<std::string> alphabet;
        std::string motif;
        int motif_offset;
        size_t base_counts = 1;
    };

    const std::string allowed_bases = "ACGT";
    std::array<ModelInfo, 4> model_info;
    for (int b = 0; b < 4; ++b) {
        model_info[b].alphabet.emplace_back(1, allowed_bases[b]);
    }

    for (const auto& params_ref : base_mod_params) {
        const auto& params = params_ref.mods;
        auto base = params.motif[params.motif_offset];
        if (allowed_bases.find(base) == std::string::npos) {
            ERROR("%s", "Invalid base in modbase model metadata.");
        }
        auto& map_entry = model_info[BASE_IDS[base]];
        map_entry.long_names = params.long_names;
        map_entry.alphabet.insert(map_entry.alphabet.end(), params.codes.begin(),
                                  params.codes.end());
        map_entry.base_counts = params.count + 1;
    }

    modbase_info_t result;
    size_t index = 0;
    for (const auto& info : model_info) {
        for (const auto& name : info.long_names) {
            if (!result.long_names.empty()) {
                result.long_names += ' ';
            }
            result.long_names += name;
        }
        result.alphabet.insert(result.alphabet.end(), info.alphabet.begin(), info.alphabet.end());
        result.base_counts[index++] = info.base_counts;
    }

    std::array<size_t, 4> offsets;
    offsets[0] = 0;
    offsets[1] = result.base_counts[0];
    offsets[2] = offsets[1] + result.base_counts[1];
    offsets[3] = offsets[2] + result.base_counts[2];

    result.base_probs_offsets = offsets;
    return result;
}