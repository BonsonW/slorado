#ifndef CRF_MODEL_H
#define CRF_MODEL_H

#include <torch/torch.h>

#include <vector>

#include "model_config.h"

using namespace torch::nn;

ModuleHolder<AnyModule> load_lstm_model(const CRFModelConfig &model_config, const torch::TensorOptions &options);

struct ConvStackImpl : torch::nn::Module {
    explicit ConvStackImpl(const std::vector<ConvParams> &layer_params);

    torch::Tensor forward(torch::Tensor x);

    struct ConvLayer {
        explicit ConvLayer(const ConvParams &params);
        const ConvParams params;
        torch::nn::Conv1d conv{nullptr};
    };

    std::vector<ConvLayer> layers;
};

struct LinearCRFImpl : torch::nn::Module {
    LinearCRFImpl(int insize, int outsize, bool bias_, bool tanh_and_scale);
    torch::Tensor forward(const torch::Tensor &x);

    bool bias;
    static constexpr int scale = 5;
    torch::nn::Linear linear{nullptr};
    torch::nn::Tanh activation{nullptr};
};

struct LSTMStackImpl : torch::nn::Module {
    LSTMStackImpl(int num_layers, int size);
    torch::Tensor forward(torch::Tensor x);
    int layer_size;
    std::vector<torch::nn::LSTM> rnns;
};

struct ClampImpl : torch::nn::Module {
    ClampImpl(float _min, float _max, bool _active);
    torch::Tensor forward(torch::Tensor x);
    bool active;
    float min, max;
};

TORCH_MODULE(LSTMStack);
TORCH_MODULE(LinearCRF);
TORCH_MODULE(ConvStack);
TORCH_MODULE(Clamp);

struct CRFModelImpl : torch::nn::Module {
    explicit CRFModelImpl(const CRFModelConfig &config);
    void load_state_dict(const std::vector<torch::Tensor> &weights);

    torch::Tensor forward(const torch::Tensor &x);
    ConvStack convs{nullptr};
    LSTMStack rnns{nullptr};
    LinearCRF linear1{nullptr}, linear2{nullptr};
    Clamp clamp1{nullptr};
    torch::nn::Sequential encoder{nullptr};
};

TORCH_MODULE(CRFModel);

#endif