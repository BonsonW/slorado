#include "CRFModel.h"

#include "../utils/tensor_utils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <math.h>
#include <toml.hpp>
#include <torch/torch.h>

#include <string>

using namespace torch::nn;
namespace F = torch::nn::functional;
using Slice = torch::indexing::Slice;
using quantized_lstm = std::function<int(void *, void *, void *, void *, void *, void *, int)>;

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

struct ConvolutionImpl : Module {
    ConvolutionImpl(int size, int outsize, int k, int stride_, bool to_lstm_ = false)
            : in_size(size), out_size(outsize), window_size(k), stride(stride_), to_lstm(to_lstm_) {
        conv = register_module(
                "conv", Conv1d(Conv1dOptions(size, outsize, k).stride(stride).padding(k / 2)));
        activation = register_module("activation", SiLU());
    }

    torch::Tensor forward(torch::Tensor x) {
        // Input x is [N, C_in, T_in], contiguity optional
        if (to_lstm) {
            // Output is [N, T_out, C_out], non-contiguous
            return activation(conv(x)).transpose(1, 2);
        }
        // Output is [N, C_out, T_out], contiguous
        return activation(conv(x));
    }

    Conv1d conv{nullptr};
    SiLU activation{nullptr};
    int in_size;
    int out_size;
    int window_size;
    int stride;
    const bool to_lstm;
};

struct LinearCRFImpl : Module {
    LinearCRFImpl(int insize, int outsize) : scale(5), blank_score(2.0), expand_blanks(false) {
        linear = register_module("linear", Linear(insize, outsize));
        activation = register_module("activation", Tanh());
    };

    torch::Tensor forward(torch::Tensor x) {
        // Input x is [N, T, C], contiguity optional
        auto N = x.size(0);
        auto T = x.size(1);

        torch::Tensor scores;
        scores = activation(linear(x)) * scale;

        if (expand_blanks == true) {
            scores = scores.contiguous();
            int C = scores.size(2);
            scores = F::pad(scores.view({N, T, C / 4, 4}),
                            F::PadFuncOptions({1, 0, 0, 0, 0, 0, 0, 0}).value(blank_score))
                             .view({N, T, -1});
        }

        if (x.device() == torch::kCPU) {
            // Output is [T, N, C]
            return scores.transpose(0, 1);
        }

        // Output is [N, T, C], contiguous
        return scores;
    }

    int scale;
    int blank_score;
    bool expand_blanks;
    Linear linear{nullptr};
    Tanh activation{nullptr};
};

struct LSTMStackImpl : Module {
    LSTMStackImpl(int size, int batchsize, int chunksize) {
        // torch::nn::LSTM expects/produces [N, T, C] with batch_first == true
        rnn1 = register_module("rnn1", LSTM(LSTMOptions(size, size).batch_first(true)));
        rnn2 = register_module("rnn2", LSTM(LSTMOptions(size, size).batch_first(true)));
        rnn3 = register_module("rnn3", LSTM(LSTMOptions(size, size).batch_first(true)));
        rnn4 = register_module("rnn4", LSTM(LSTMOptions(size, size).batch_first(true)));
        rnn5 = register_module("rnn5", LSTM(LSTMOptions(size, size).batch_first(true)));
    };

    torch::Tensor forward(torch::Tensor x) {
        // Input is [N, T, C], contiguity optional

        // auto [y1, h1] = rnn1(x.flip(1));
        // auto [y2, h2] = rnn2(y1.flip(1));
        // auto [y3, h3] = rnn3(y2.flip(1));
        // auto [y4, h4] = rnn4(y3.flip(1));
        // auto [y5, h5] = rnn5(y4.flip(1));

        x = x.flip(1);

        // rnn1
        auto t1 = rnn1(x);
        auto y1 = std::get<0>(t1);
        auto h1 = std::get<1>(t1);


        x = y1.flip(1);

        // rnn2
        auto t2 = rnn2(x);
        auto y2 = std::get<0>(t2);
        auto h2 = std::get<1>(t2);

        x = y2;

        // rnn3
        x = x.flip(1);
        auto t3 = rnn3(x);
        auto y3 = std::get<0>(t3);
        auto h3 = std::get<1>(t3);

        x = y3.flip(1);

        // rnn4
        auto t4 = rnn4(x);
        auto y4 = std::get<0>(t4);
        auto h4 = std::get<1>(t4);
        x = y4;

        x = x.flip(1);

        // rnn5
        auto t5 = rnn5(x);
        auto y5 = std::get<0>(t5);
        auto h5 = std::get<1>(t5);
        
        x = y5.flip(1);

        // Output is [N, T, C], non-contiguous
        return x;
    }

    LSTM rnn1{nullptr}, rnn2{nullptr}, rnn3{nullptr}, rnn4{nullptr}, rnn5{nullptr};
};

struct ClampImpl : Module {
    ClampImpl(float _min, float _max, bool _active) : min(_min), max(_max), active(_active){};

    torch::Tensor forward(torch::Tensor x) {
        if (active) {
            return x.clamp(min, max);
        } else {
            return x;
        }
    }

    bool active;
    float min, max;
};

TORCH_MODULE(LSTMStack);
TORCH_MODULE(LinearCRF);
TORCH_MODULE(Convolution);
TORCH_MODULE(Clamp);

template <class LSTMStackType>
struct CRFModelImpl : Module {
    CRFModelImpl(const CRFModelConfig &config, bool expand_blanks, int batch_size, int chunk_size) {
        if (config.insize == 128) {
            conv1 = register_module("conv1", Convolution(config.num_features, 16, 5, 1));
            clamp1 = Clamp(-0.5, 3.5, config.clamp);
            conv2 = register_module("conv2", Convolution(16, 16, 5, 1));
            clamp2 = Clamp(-0.5, 3.5, config.clamp);
            conv3 = register_module("conv3",
                                    Convolution(16, config.insize, 19, config.stride, true));
            clamp3 = Clamp(-0.5, 3.5, config.clamp);
        } else {
            conv1 = register_module("conv1", Convolution(config.num_features, config.conv, 5, 1));
            clamp1 = Clamp(-0.5, 3.5, config.clamp);
            conv2 = register_module("conv2", Convolution(config.conv, 16, 5, 1));
            clamp2 = Clamp(-0.5, 3.5, config.clamp);
            conv3 = register_module("conv3",
                                    Convolution(16, config.insize, 19, config.stride, true));
            clamp3 = Clamp(-0.5, 3.5, config.clamp);
        }

        rnns = register_module(
                "rnns", LSTMStackType(config.insize, batch_size, chunk_size / config.stride));

        if (config.out_features != -1) {
            // The linear layer is decomposed into 2 matmuls.
            const int decomposition = config.out_features;
            linear1 = register_module("linear1", Linear(config.insize, decomposition));
            linear2 = register_module(
                    "linear2", Linear(LinearOptions(decomposition, config.outsize).bias(false)));
            clamp4 = Clamp(-5.0, 5.0, config.clamp);
            encoder = Sequential(conv1, clamp1, conv2, clamp2, conv3, clamp3, rnns, linear1,
                                 linear2, clamp4);
        } else if ((config.conv == 16) && (config.num_features == 1)) {
            linear1 = register_module(
                    "linear1", Linear(LinearOptions(config.insize, config.outsize).bias(false)));
            clamp4 = Clamp(-5.0, 5.0, config.clamp);
            encoder =
                    Sequential(conv1, clamp1, conv2, clamp2, conv3, clamp3, rnns, linear1, clamp4);
        } else {
            linear = register_module("linear1", LinearCRF(config.insize, config.outsize));
            encoder = Sequential(conv1, conv2, conv3, rnns, linear);
        }
    }

    void load_state_dict(const std::vector<torch::Tensor> &weights) {
        module_load_state_dict(*this, weights);
    }

    torch::Tensor forward(torch::Tensor x) {
        // nvtx3::scoped_range loop{"nn_forward"};
        return encoder->forward(x);
    }

    LSTMStackType rnns{nullptr};
    LinearCRF linear{nullptr};
    Linear linear1{nullptr}, linear2{nullptr};
    Sequential encoder{nullptr};
    Convolution conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    Clamp clamp1{nullptr}, clamp2{nullptr}, clamp3{nullptr}, clamp4{nullptr};
};

using CpuCRFModelImpl = CRFModelImpl<LSTMStack>;
TORCH_MODULE(CpuCRFModel);

CRFModelConfig load_crf_model_config(const std::string &path) {
    const auto config_toml = toml::parse(path + "/config.toml");

    CRFModelConfig config;
    config.qscale = 1.0f;
    config.qbias = 0.0f;

    if (config_toml.contains("qscore")) {
        const auto &qscore = toml::find(config_toml, "qscore");
        config.qbias = toml::find<float>(qscore, "bias");
        config.qscale = toml::find<float>(qscore, "scale");
    } else {
        // spdlog::debug("> no qscore calibration found");
    }

    config.conv = 4;
    config.insize = 0;
    config.stride = 1;
    config.bias = true;
    config.clamp = false;

    // The encoder scale only appears in pre-v4 models.  In v4 models
    // the value of 1 is used.
    config.scale = 1.0f;

    const auto &input = toml::find(config_toml, "input");
    config.num_features = toml::find<int>(input, "features");

    const auto &encoder = toml::find(config_toml, "encoder");
    if (encoder.contains("type")) {
        // v4-type model
        for (const auto &segment : toml::find(config_toml, "encoder", "sublayers").as_array()) {
            const auto type = toml::find<std::string>(segment, "type");
            if (type.compare("convolution") == 0) {
                // Overall stride is the product of all conv layers' strides.
                config.stride *= toml::find<int>(segment, "stride");
            } else if (type.compare("lstm") == 0) {
                config.insize = toml::find<int>(segment, "size");
            } else if (type.compare("linear") == 0) {
                // Specifying out_features implies a decomposition of the linear layer matrix
                // multiply with a bottleneck before the final feature size.
                config.out_features = toml::find_or<int>(segment, "out_features", -1); // I have no idea if -1 is an invalid number but will have to use for now without C++17 in place of std::optional
            } else if (type.compare("clamp") == 0) {
                config.clamp = true;
            } else if (type.compare("linearcrfencoder") == 0) {
                config.blank_score = toml::find<float>(segment, "blank_score");
            }
        }
        config.conv = 16;
        config.bias = config.insize > 128;
    } else {
        // pre-v4 model
        config.stride = toml::find<int>(encoder, "stride");
        config.insize = toml::find<int>(encoder, "features");
        config.blank_score = toml::find<float>(encoder, "blank_score");
        config.scale = toml::find<float>(encoder, "scale");

        if (encoder.contains("first_conv_size")) {
            config.conv = toml::find<int>(encoder, "first_conv_size");
        }
    }

    const auto &global_norm = toml::find(config_toml, "global_norm");
    // Note that in v4 files state_len appears twice: under global_norm and under
    // linearcrfencoder.  We are ignoring the latter.
    config.state_len = toml::find<int>(global_norm, "state_len");

    // CUDA and CPU paths do not output explicit stay scores from the NN.
    config.outsize = pow(4, config.state_len) * 4;

    return config;
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

ModuleHolder<AnyModule> load_crf_model(const std::string &path,
                                       const CRFModelConfig &model_config,
                                       const int batch_size,
                                       const int chunk_size,
                                       const torch::TensorOptions &options) {
    const bool expand_blanks = true;
    auto model = CpuCRFModel(model_config, expand_blanks, batch_size, chunk_size);
    return populate_model(model, path, options, model_config.out_features,
                          model_config.bias);
}