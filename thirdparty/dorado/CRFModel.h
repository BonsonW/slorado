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

// Procedural (C-style) plain-LSTM/CRF model — e.g. fast/hac v5. Weight structs + free-function
// forward; the heavy math stays on at:: ops (conv1d / cuDNN lstm / linear). FLSTM (v6) and TX
// remain on the torch::nn path for now.
typedef struct {
    at::Tensor w;   // conv weight [out, in, winlen]
    at::Tensor b;   // conv bias [out]
    int stride;
    int padding;
    Activation activation;
} conv_layer_t;

typedef struct {
    at::Tensor w_ih, w_hh, b_ih, b_hh;   // single-layer batch_first LSTM params
} lstm_layer_t;

typedef struct {
    std::vector<conv_layer_t> convs;
    std::vector<lstm_layer_t> lstms;   // bidirectional-alternating (flip per layer)
    at::Tensor linear_w;               // CRF linear weight [outsize, lstm_size]
    at::Tensor linear_b;               // undefined => no bias
    int lstm_size;
    bool clamp;
    float clamp_min, clamp_max;
    lstm_stats_t *stats;
} lstm_model_t;

lstm_model_t *load_lstm_model_proc(const CRFModelConfig &config, const torch::TensorOptions &options, lstm_stats_t *model_stats);
at::Tensor lstm_model_forward(const lstm_model_t *m, at::Tensor x);
void free_lstm_model(lstm_model_t *m);

// Conv stack forward: [N, C_in, T] -> [N, T, C_out]. Shared by the procedural LSTM/FLSTM/TX models.
at::Tensor conv_stack_forward(const std::vector<conv_layer_t> &convs, at::Tensor x);

// Procedural factored-LSTM (FLSTM) model — hac/fast v6. Down/up-projected LSTM with a per-timestep
// recurrence (openfish_flstm_step_gpu on GPU), decomposed linear1 + tanh-scaled linear2 CRF, no
// clamp. Weights are fake-quantised inline; calib hooks are registered with the real loaded weights.
typedef struct {
    at::Tensor dn_w_ih, dn_w_hh, up_w_ih, up_w_hh, up_b_ih, up_b_hh;
    std::string prefix;                        // quant_methods lookup key
    calib_layer_t *cl_dn_ih = nullptr, *cl_up_ih = nullptr;
    calib_layer_t *cl_dn_hh = nullptr, *cl_up_hh = nullptr;
} flstm_layer_t;

typedef struct {
    std::vector<conv_layer_t> convs;
    std::vector<flstm_layer_t> flstms;         // bidirectional-alternating
    int C, K;
    at::Tensor linear1_w, linear1_b;           // decomposed CRF linear (bias usually undefined)
    at::Tensor linear2_w;                       // tanh(scores) * scale
    lstm_stats_t *stats;
} flstm_model_t;

flstm_model_t *load_flstm_model_proc(const CRFModelConfig &config, const torch::TensorOptions &options, lstm_stats_t *model_stats);
at::Tensor flstm_model_forward(const flstm_model_t *m, at::Tensor x);
void free_flstm_model(flstm_model_t *m);

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
    void update_calib_weights();
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