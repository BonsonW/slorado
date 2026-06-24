#ifndef CRF_MODEL_H
#define CRF_MODEL_H

#include <torch/torch.h>

#include <vector>

#include "model_config.h"
#include "tensor_chunk_utils.h"
#include "calib.h"
#include "quant.h"

using namespace torch::nn;

ModuleHolder<AnyModule> load_lstm_model(const CRFModelConfig &model_config, const torch::TensorOptions &options, lstm_stats_t *model_stats);

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

struct FLSTMLayerImpl : torch::nn::Module {
    FLSTMLayerImpl(int C, int K, lstm_stats_t *model_stats, const std::string &name_prefix = "");
    torch::Tensor forward(torch::Tensor x);
private:
    int C_, K_;
    lstm_stats_t *model_stats_;
    torch::Tensor dn_weight_ih_, dn_weight_hh_;
    torch::Tensor up_weight_ih_, up_weight_hh_;
    torch::Tensor up_bias_ih_,   up_bias_hh_;

    // calibration (null when --calibrate not set)
    calib_stats_t *calib_stats_ = nullptr;
    std::string calib_prefix_;
    calib_layer_t *cl_dn_ih_ = nullptr, *cl_up_ih_ = nullptr;
    calib_layer_t *cl_dn_hh_ = nullptr, *cl_up_hh_ = nullptr;
};

struct FLSTMStackImpl : torch::nn::Module {
    FLSTMStackImpl(int num_layers, int C, int K, lstm_stats_t *model_stats);
    torch::Tensor forward(torch::Tensor x);
    std::vector<torch::nn::ModuleHolder<FLSTMLayerImpl>> layers_;
};

struct ClampImpl : torch::nn::Module {
    ClampImpl(float _min, float _max, bool _active);
    torch::Tensor forward(torch::Tensor x);
    bool active;
    float min, max;
};

TORCH_MODULE(LSTMStack);
TORCH_MODULE(FLSTMLayer);
TORCH_MODULE(FLSTMStack);
TORCH_MODULE(LinearCRF);
TORCH_MODULE(ConvStack);
TORCH_MODULE(Clamp);

struct CRFModelImpl : torch::nn::Module {
    explicit CRFModelImpl(const CRFModelConfig &config, lstm_stats_t *model_stats);
    void load_state_dict(const std::vector<torch::Tensor> &weights);

    torch::Tensor forward(const torch::Tensor &x);
    ConvStack convs{nullptr};
    LSTMStack rnns{nullptr};
    FLSTMStack flstm_rnns{nullptr};
    LinearCRF linear1{nullptr}, linear2{nullptr};
    Clamp clamp1{nullptr};
    lstm_stats_t *model_stats_;
};

TORCH_MODULE(CRFModel);

#endif