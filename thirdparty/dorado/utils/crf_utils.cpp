#include "crf_utils.h"
#include "toml.h"
#include "error.h"

#include "../nn/CRFModel.h"
#include "../utils/tensor_utils.h"

#include <algorithm>
#include <thread>

using namespace torch::nn;

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
    config.num_features = toml_int_in(input, "features").u.i;

    toml_table_t *encoder = toml_table_in(config_toml, "encoder");
    if (toml_key_exists(encoder, "type")) {
        // v4-type model
        toml_array_t *sublayers = toml_array_in(encoder, "sublayers");
        for (int i = 0; ; i++) {
            toml_table_t *segment = toml_table_at(sublayers, i);
            if (!segment) break;

            char *type = toml_string_in(segment, "type").u.s;
            if (strcmp(type, "convolution") == 0) {
                // Overall stride is the product of all conv layers' strides.
                config.stride *= toml_int_in(segment, "stride").u.i;
            } else if (strcmp(type, "lstm") == 0) {
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
            } else if (strcmp(type, "clamp") == 0) {
                config.clamp = true;
            } else if (strcmp(type, "linearcrfencoder") == 0) {
                config.blank_score = (float)toml_double_in(segment, "blank_score").u.d;
            }

            free(type);
        }

        config.conv = 16;
        config.bias = config.insize > 128;
    } else {
        // pre-v4 model
        config.stride = toml_int_in(encoder, "stride").u.i;
        config.insize = toml_int_in(encoder, "features").u.i;
        config.blank_score = (float)toml_double_in(encoder, "blank_score").u.d;
        config.scale = (float)toml_double_in(encoder, "scale").u.d;

        if (toml_key_exists(encoder, "first_conv_size")) {
            config.conv = toml_int_in(encoder, "first_conv_size").u.i;
        }
    }

    toml_table_t *global_norm = toml_table_in(config_toml, "global_norm");
    // Note that in v4 files state_len appears twice: under global_norm and under
    // linearcrfencoder.  We are ignoring the latter.
    config.state_len = toml_int_in(global_norm, "state_len").u.i;

    // CUDA and CPU paths do not output explicit stay scores from the NN.
    config.outsize = pow(4, config.state_len) * 4;

    toml_free(config_toml);

    return config;
}