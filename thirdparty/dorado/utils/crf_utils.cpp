#include "crf_utils.h"
#include "toml.h"
#include "error.h"

#include "../nn/CRFModel.h"
#include "../utils/tensor_utils.h"

#include <algorithm>
#include <thread>

using namespace torch::nn;

enum SublayerType { CLAMP, CONVOLUTION, LINEAR, LINEAR_CRF_ENCODER, LSTM, PERMUTE, UNRECOGNISED };

SublayerType sublayer_type(const char *type) {
    if (strcmp(type, "clamp") == 0)             return SublayerType::CLAMP;
    if (strcmp(type, "convolution") == 0)       return SublayerType::CONVOLUTION;
    if (strcmp(type, "linear") == 0)            return SublayerType::LINEAR;
    if (strcmp(type, "linearcrfencoder") == 0)  return SublayerType::LINEAR_CRF_ENCODER;
    if (strcmp(type, "lstm") == 0)              return SublayerType::LSTM;
    if (strcmp(type, "permute") == 0)           return SublayerType::PERMUTE;
    return SublayerType::UNRECOGNISED;
}

// Parse sublayer extracting convolution parameters. This is for use on v4+ models only
ConvParams parse_conv_params(toml_table_t *segment, bool clamp) {
    ConvParams params;
    
    params.insize = toml_int_in(segment, "insize").u.i;
    params.size = toml_int_in(segment, "size").u.i;
    params.winlen = toml_int_in(segment, "winlen").u.i;
    params.stride = toml_int_in(segment, "stride").u.i;

    char *activation = toml_string_in(segment, "activation").u.s;
    if (strcmp(activation, "swish") == 0) {
        params.activation = clamp ? Activation::SWISH_CLAMP : Activation::SWISH;
    } else if (strcmp(activation, "tanh") == 0) {
        params.activation = Activation::TANH;
    } else {
        ERROR("Unknown activation: `%s` in model config, expected `swish` or `tanh`", activation);
    }

    return params;
}

// Parse sublayers extracting convolution parameters. This is for use on v4+ models only
std::vector<ConvParams> parse_convs(toml_array_t *sublayers) {
    std::vector<ConvParams> convs;

    for (int i = 0; ; i++) {
        toml_table_t *segment = toml_table_at(sublayers, i);
        if (!segment) break;

        char *type_str = toml_string_in(segment, "type").u.s;
        SublayerType type = sublayer_type(type_str);

        // If the sublayer after a convolution is a clamp, the activation function may have
        // a fused implementation
        if (type == SublayerType::CONVOLUTION) {
            bool has_clamp_next = false;
            toml_table_t *next_segment = toml_table_at(sublayers, i+1);

            if (next_segment != NULL) {
                char *next_type_str = toml_string_in(next_segment, "type").u.s;
                SublayerType next_type = sublayer_type(next_type_str);
                has_clamp_next = next_type == SublayerType::CLAMP;
                free(next_type_str);
            }

            ConvParams conv = parse_conv_params(segment, has_clamp_next);
            convs.push_back(conv);
        }
        
        free(type_str);
    }
    return convs;
}

// Parse a the config.toml to resolve the scaling parameters.
// SignalNormalisationParams parse_signal_normalisation_params(const toml::value &config_toml,
//                                                             const std::string &model_name) {
//     SignalNormalisationParams params;

//     // med_mad scaling set based on filename for r9.4.1 models (~v3)
//     if (model_name.rfind("dna_r9.4.1", 0) == 0) {
//         params.strategy = ScalingStrategy::MED_MAD;
//     }

//     // scaling.strategy introduced with v4.3 models
//     if (config_toml.contains("scaling")) {
//         const auto &scaling = toml::find(config_toml, "scaling");
//         params.strategy =
//                 scaling_strategy_from_string(toml::find<std::string>(scaling, "strategy"));
//     }

//     if (config_toml.contains("normalisation")) {
//         const auto &norm = toml::find(config_toml, "normalisation");
//         params.quantile.quantile_a = toml::find<float>(norm, "quantile_a");
//         params.quantile.quantile_b = toml::find<float>(norm, "quantile_b");
//         params.quantile.shift_multiplier = toml::find<float>(norm, "shift_multiplier");
//         params.quantile.scale_multiplier = toml::find<float>(norm, "scale_multiplier");

//         if (params.strategy != ScalingStrategy::QUANTILE) {
//             spdlog::warn(
//                     "Normalisation parameters are only used when `scaling.strategy = quantile`");
//         }
//     }

//     if (config_toml.contains("standardisation")) {
//         const auto &norm = toml::find(config_toml, "standardisation");
//         params.standarisation.standardise = toml_int_in(norm, "standardise") > 0;
//         if (params.standarisation.standardise) {
//             params.standarisation.mean = toml::find<float>(norm, "mean");
//             params.standarisation.stdev = toml::find<float>(norm, "stdev");
//         }

//         if (params.standarisation.standardise && params.strategy != ScalingStrategy::PA) {
//             throw std::runtime_error(
//                     "Signal standardisation is implemented only for `scaling.strategy = pa`");
//         }

//         if (params.standarisation.stdev <= 0.0f) {
//             throw std::runtime_error(
//                     "Config error: `standardisation.stdev` must be greater than 0, got: " +
//                     std::to_string(params.standarisation.stdev));
//         }
//     }

//     return params;
// }

CRFModelConfig load_crf_model_config(const std::string &path) {
    FILE* fp;
    char errbuf[200];

    fp = fopen((path + "/config.toml").c_str(), "r");
    if (!fp) {
        ERROR("cannot open toml - %s", (path + "/config.toml").c_str());
    }

    toml_table_t *config_toml = toml_parse_file(fp, errbuf, sizeof(errbuf));
    fclose(fp);

    if (!config_toml) {
        ERROR("cannot parse - %s", errbuf);
    }

    CRFModelConfig config;
    config.qscale = 1.0f;
    config.qbias = 0.0f;
    config.lstm_size = 0;

    if (toml_key_exists(config_toml, "qscore")) {
        toml_table_t *qscore = toml_table_in(config_toml, "qscore");
        config.qbias = (float)toml_double_in(qscore, "bias").u.d;
        config.qscale = (float)toml_double_in(qscore, "scale").u.d;
    } else {
        // no qscore calibration found
    }

    config.insize = 0;
    config.stride = 1;
    config.bias = false;
    config.clamp = false;
    config.decomposition = false;

    // The encoder scale only appears in pre-v4 models.  In v4 models
    // the value of 1 is used.
    config.scale = 1.0f;

    toml_table_t *input = toml_table_in(config_toml, "input");
    config.num_features = toml_int_in(input, "features").u.i;

    toml_table_t *encoder = toml_table_in(config_toml, "encoder");
    if (toml_key_exists(encoder, "type")) {
        // v4-type model
        toml_array_t *sublayers = toml_array_in(encoder, "sublayers");
        config.convs = parse_convs(sublayers);
        for (const auto &cv : config.convs) {
            config.stride *= cv.stride;
        }
        config.lstm_size = config.convs.back().size;

        for (int i = 0; ; i++) {
            toml_table_t *segment = toml_table_at(sublayers, i);
            if (!segment) break;

            char *type = toml_string_in(segment, "type").u.s;
            if (strcmp(type, "lstm") == 0) {
                config.insize = toml_int_in(segment, "size").u.i;
            } else if (strcmp(type, "linear") == 0) {
                // Specifying out_features implies a decomposition of the linear layer matrix
                // multiply with a bottleneck before the final feature size.
                try {
                    config.out_features = toml_int_in(segment, "out_features").u.i;
                    config.decomposition = true;
                } catch (std::out_of_range e) {
                    config.decomposition = false;
                }
                config.bias = config.lstm_size > 128;
            } else if (strcmp(type, "clamp") == 0) {
                config.clamp = true;
            } else if (strcmp(type, "linearcrfencoder") == 0) {
                config.blank_score = (float)toml_double_in(segment, "blank_score").u.d;
            }

            free(type);
        }
    } else {
        // pre-v4 model
        config.stride = toml_int_in(encoder, "stride").u.i;
        config.insize = toml_int_in(encoder, "features").u.i;
        config.blank_score = (float)toml_double_in(encoder, "blank_score").u.d;
        config.scale = (float)toml_double_in(encoder, "scale").u.d;

        int first_conv = 4;
        if (toml_key_exists(encoder, "first_conv_size")) {
            first_conv = toml_int_in(encoder, "first_conv_size").u.i;
        }

        config.convs.push_back(ConvParams{config.num_features, first_conv, 5, 1, Activation::SWISH});
        config.convs.push_back(ConvParams{first_conv, 16, 5, 1, Activation::SWISH});
        config.convs.push_back(ConvParams{16, config.lstm_size, 19, config.stride, Activation::SWISH});
    }

    toml_table_t *global_norm = toml_table_in(config_toml, "global_norm");
    // Note that in v4 files state_len appears twice: under global_norm and under
    // linearcrfencoder.  We are ignoring the latter.
    config.state_len = toml_int_in(global_norm, "state_len").u.i;

    // CUDA and CPU paths do not output explicit stay scores from the NN.
    config.outsize = pow(4, config.state_len + 1);

    toml_free(config_toml);

    return config;
}

template <class Model>
ModuleHolder<AnyModule> populate_model(Model &&model,
                                       const std::string &path,
                                       const torch::TensorOptions &options,
                                       bool decomposition,
                                       bool bias) {
    auto state_dict = load_crf_model_weights(path, decomposition, bias);
    model->load_state_dict(state_dict);
    model->to(options.dtype_opt().value().toScalarType());
    model->to(options.device_opt().value());
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);
    return holder;
}


ModuleHolder<AnyModule> load_crf_model(const std::string &model_path,
                                       const CRFModelConfig &model_config,
                                       const int batch_size,
                                       const int chunk_size,
                                       const torch::TensorOptions &options) {
    auto model = CRFModel(model_config);
    auto state_dict = load_crf_model_weights(
            model_path, model_config.decomposition, model_config.bias);
    model->load_state_dict(state_dict);
    model->to(options.dtype().toScalarType());
    model->to(options.device());
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);
    return holder;
}

std::vector<torch::Tensor> load_crf_model_weights(const std::string &dir,
                                                  bool decomposition,
                                                  bool bias) {
    auto tensors = std::vector<std::string>{
            "0.conv.weight.tensor",      "0.conv.bias.tensor",

            "1.conv.weight.tensor",      "1.conv.bias.tensor",

            "2.conv.weight.tensor",      "2.conv.bias.tensor",

            "4.rnn.weight_ih_l0.tensor", "4.rnn.weight_hh_l0.tensor",
            "4.rnn.bias_ih_l0.tensor",   "4.rnn.bias_hh_l0.tensor",

            "5.rnn.weight_ih_l0.tensor", "5.rnn.weight_hh_l0.tensor",
            "5.rnn.bias_ih_l0.tensor",   "5.rnn.bias_hh_l0.tensor",

            "6.rnn.weight_ih_l0.tensor", "6.rnn.weight_hh_l0.tensor",
            "6.rnn.bias_ih_l0.tensor",   "6.rnn.bias_hh_l0.tensor",

            "7.rnn.weight_ih_l0.tensor", "7.rnn.weight_hh_l0.tensor",
            "7.rnn.bias_ih_l0.tensor",   "7.rnn.bias_hh_l0.tensor",

            "8.rnn.weight_ih_l0.tensor", "8.rnn.weight_hh_l0.tensor",
            "8.rnn.bias_ih_l0.tensor",   "8.rnn.bias_hh_l0.tensor",

            "9.linear.weight.tensor"};

    if (bias) {
        tensors.push_back("9.linear.bias.tensor");
    }

    if (decomposition) {
        tensors.push_back("10.linear.weight.tensor");
    }

    return load_tensors(dir, tensors);
}
