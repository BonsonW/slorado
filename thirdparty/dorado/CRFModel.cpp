#include <math.h>
#include <string>

#include "CRFModel.h"
#include "error.h"
#include "misc.h"
#include "tensor_chunk_utils.h"
#include "quant.h"

#ifdef USE_GPU
#include <openfish/openfish.h>
#endif

using namespace torch::nn;

ConvStackImpl::ConvStackImpl(const std::vector<ConvParams> &layer_params) {
    for (size_t i = 0; i < layer_params.size(); ++i) {
        layers.emplace_back(layer_params[i]);
        auto &layer = layers.back();
        auto opts = Conv1dOptions(layer.params.insize, layer.params.size, layer.params.winlen)
            .stride(layer.params.stride)
            .padding(layer.params.winlen / 2);
        layer.conv = register_module(std::string("conv") + std::to_string(i + 1), Conv1d(opts));
    }
}

torch::Tensor ConvStackImpl::forward(torch::Tensor x) {
    // Input x is [N, C_in, T_in], contiguity optional
    for (auto &layer : layers) {
        x = layer.conv(x);
        if (layer.params.activation == Activation::SWISH) {
            torch::silu_(x);
        } else if (layer.params.activation == Activation::SWISH_CLAMP) {
            torch::silu_(x).clamp_(c10::nullopt, 3.5f);
        } else if (layer.params.activation == Activation::TANH) {
            x.tanh_();
        } else {
            ERROR("%s", "Unrecognised activation function id.");
        }
    }
    // Output is [N, T_out, C_out], non-contiguous
    return x.transpose(1, 2);
}

ConvStackImpl::ConvLayer::ConvLayer(const ConvParams &conv_params) : params(conv_params) {}

LinearCRFImpl::LinearCRFImpl(int insize, int outsize, bool bias_, bool tanh_and_scale) : bias(bias_) {
    linear = register_module("linear", Linear(LinearOptions(insize, outsize).bias(bias)));
    if (tanh_and_scale) {
        activation = register_module("activation", Tanh());
    }
};

void LinearCRFImpl::set_calib(const std::string &name, calib_stats_t *calib) {
    calib_stats_ = calib;
    calib_layer_ = calib->register_layer(name, linear->weight);
}

torch::Tensor LinearCRFImpl::forward(const torch::Tensor &x) {
    // Input x is [N, T, C], contiguity optional
    if (calib_layer_) calib_stats_->accumulate(calib_layer_, x);
    auto W = fake_quant(linear->weight, qm_);
    auto scores = at::linear(fake_quant(x, qm_), W, linear->bias);
    if (activation) {
        scores = activation(scores) * scale;
    }

    // Output is [N, T, C], contiguous
    return scores;
}

LSTMStackImpl::LSTMStackImpl(int num_layers, int size) : layer_size(size) {
    // torch::nn::LSTM expects/produces [N, T, C] with batch_first == true
    const auto lstm_opts = LSTMOptions(size, size).batch_first(true);
    for (int i = 0; i < num_layers; ++i) {
        auto label = std::string("rnn") + std::to_string(i + 1);
        rnns.emplace_back(register_module(label, LSTM(lstm_opts)));
    }
};

torch::Tensor LSTMStackImpl::forward(torch::Tensor x) {
    // Input is [N, T, C], contiguity optional
    for (auto &rnn : rnns) {
        x = std::get<0>(rnn(x.flip(1)));
    }

    // Output is [N, T, C], contiguous
    return (rnns.size() & 1) ? x.flip(1) : x;
}

FLSTMLayerImpl::FLSTMLayerImpl(int C, int K, lstm_stats_t *model_stats, const std::string &name_prefix) : C_(C), model_stats_(model_stats) {
    dn_weight_ih_ = register_parameter("dn_weight_ih", torch::empty({K, C}));
    dn_weight_hh_ = register_parameter("dn_weight_hh", torch::empty({K, C}));
    up_weight_ih_ = register_parameter("up_weight_ih", torch::empty({4 * C, K}));
    up_weight_hh_ = register_parameter("up_weight_hh", torch::empty({4 * C, K}));
    up_bias_ih_   = register_parameter("up_bias_ih",   torch::empty({4 * C}));
    up_bias_hh_   = register_parameter("up_bias_hh",   torch::empty({4 * C}));

    if (!name_prefix.empty() && model_stats) {
        calib_prefix_ = name_prefix;
        if (model_stats->calib_stats) {
            calib_stats_ = model_stats->calib_stats;
        }
    }
}

void FLSTMLayerImpl::fuse_weights() {
    // Fuse two sequential matmuls into one: x @ dn.T @ up.T == x @ W_fused
    // W_fused = dn.T @ up.T, shape (C, 4*C)
    W_ih_fused_ = torch::matmul(dn_weight_ih_.t(), up_weight_ih_.t()).contiguous();
    W_hh_fused_ = torch::matmul(dn_weight_hh_.t(), up_weight_hh_.t()).contiguous();

    // Register fused matrices with calibration if active (these are what get quantized).
    // Transpose to (out, in) = (4*C, C) so compute_weight_stats sees the standard convention.
    if (calib_stats_) {
        cl_ih_fused_ = calib_stats_->register_layer(calib_prefix_ + ".ih_fused", W_ih_fused_.t().contiguous());
        cl_hh_fused_ = calib_stats_->register_layer(calib_prefix_ + ".hh_fused", W_hh_fused_.t().contiguous());
    }

    // Look up quant methods for the fused matrices.
    if (model_stats_->quant_config) {
        const auto &cfg = *model_stats_->quant_config;
        auto it = cfg.find(calib_prefix_ + ".ih_fused");
        if (it != cfg.end()) qm_ih_fused_ = it->second;
        it = cfg.find(calib_prefix_ + ".hh_fused");
        if (it != cfg.end()) qm_hh_fused_ = it->second;
        // Optional .act suffix overrides activation granularity independently from weight.
        it = cfg.find(calib_prefix_ + ".ih_fused.act");
        qm_ih_fused_act_ = (it != cfg.end()) ? it->second : qm_ih_fused_;
        it = cfg.find(calib_prefix_ + ".hh_fused.act");
        qm_hh_fused_act_ = (it != cfg.end()) ? it->second : qm_hh_fused_;
    }
}

torch::Tensor FLSTMLayerImpl::forward(torch::Tensor x) {
    // x is [N, T, C]
    x = x.transpose(0, 1).contiguous();  // [T, N, C]
    const int T = x.size(0);
    const int N = x.size(1);
    const bool on_gpu = !x.device().is_cpu();
    double a, b;

    // ih precompute: one addmm over full sequence [T*N, C] @ [C, 4*C] -> [T, N, 4*C]
    a = realtime();
    auto x_flat = x.view({T * N, x.size(2)});
    // x is (T, N, C); transpose to (N, T, C) for standard (..., T, C) layout.
    if (cl_ih_fused_) calib_stats_->accumulate(cl_ih_fused_, x.transpose(0, 1));
    auto W_ih = fake_quant(W_ih_fused_, qm_ih_fused_, /*transposed=*/true);
    auto ih = torch::addmm(up_bias_ih_, fake_quant(x_flat, qm_ih_fused_act_), W_ih).view({T, N, 4 * C_});
    if (on_gpu) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats_->time_flstm_precompute += b - a;

    auto hh = torch::empty({T + 1, N, C_}, x.options());
    hh[0].zero_();
    auto c       = torch::zeros({N, C_},     x.options());
    auto scratch = torch::empty({N, 4 * C_}, x.options());

    a = realtime();
    auto W_hh = fake_quant(W_hh_fused_, qm_hh_fused_, /*transposed=*/true);
    for (int t = 0; t < T; ++t) {
        // One addmm per step: scratch = up_bias_hh_ + hh[t] @ W_hh_fused_
        // a = realtime();
        torch::addmm_out(scratch, up_bias_hh_, fake_quant(hh[t], qm_hh_fused_act_), W_hh);
        // if (on_gpu) torch::cuda::synchronize(x.device().index());
        // b = realtime();
        // model_stats_->time_flstm_linear2 += b - a;

        // Fused epilogue: scratch + ih[t] -> gates -> cell update -> hh[t+1]
        
#ifdef USE_GPU
        openfish_flstm_step_gpu(scratch.data_ptr(), ih[t].data_ptr(),
                                c.data_ptr(), hh[t + 1].data_ptr(), N, C_);
#else
        auto gates = scratch.add(ih[t]).chunk(4, 1);
        auto i = gates[0].mul_(0.2f).add_(0.5f).clamp_(0.f, 1.f);
        auto f = gates[1].mul_(0.2f).add_(0.5f).clamp_(0.f, 1.f);
        auto g = gates[2].clamp_(-1.f, 1.f);
        auto o = gates[3].mul_(0.2f).add_(0.5f).clamp_(0.f, 1.f);
        c = (f * c) + (i * g);
        hh[t + 1] = o * torch::tanh(c);
#endif
        // if (on_gpu) torch::cuda::synchronize(x.device().index());
        
    }
    b = realtime();
    model_stats_->time_flstm_recurrence += b - a;

    using namespace torch::indexing;

    // Accumulate hh calib stats once post-loop with the full [N, T, C] batch.
    // hh[0..T-1] are the hidden-state inputs fed into W_hh_fused_ each step.
    if (cl_hh_fused_) {
        calib_stats_->accumulate(cl_hh_fused_,
            hh.index({Slice(0, T)}).transpose(0, 1).contiguous());  // (N, T, C)
    }

    // Return [N, T, C]
    return hh.index({Slice(1, None)}).transpose(0, 1).contiguous();
}

FLSTMStackImpl::FLSTMStackImpl(int num_layers, int C, int K, lstm_stats_t *model_stats) {
    for (int i = 0; i < num_layers; ++i) {
        auto label = std::string("rnn") + std::to_string(i + 1);
        auto prefix = std::string("rnns.") + label;
        layers_.emplace_back(register_module(label, FLSTMLayer(C, K, model_stats, prefix)));
    }
}

torch::Tensor FLSTMStackImpl::forward(torch::Tensor x) {
    // Flip before every layer (alternating directions); odd layer count → flip output
    for (auto &layer : layers_) {
        x = layer->forward(x.flip(1));
    }
    return (layers_.size() & 1) ? x.flip(1) : x;
}

void FLSTMStackImpl::fuse_weights() {
    for (auto &layer : layers_) {
        layer->fuse_weights();
    }
}

ClampImpl::ClampImpl(float _min, float _max, bool _active)
        : active(_active), min(_min), max(_max) {}

torch::Tensor ClampImpl::forward(torch::Tensor x) {
    if (active) {
        x.clamp_(min, max);
    }
    return x;
}

CRFModelImpl::CRFModelImpl(const CRFModelConfig &config, lstm_stats_t *model_stats) : model_stats_(model_stats) {
    const auto cv = config.convs;
    const auto lstm_size = config.lstm_size;
    convs = register_module("convs", ConvStack(cv));

    if (config.lstm_inner_dim >= 0) {
        // v6.0+ FLSTM model: decomposed linear + tanh in CRF encoder, no clamp
        flstm_rnns = register_module("rnns", FLSTMStack(config.lstm_layers, lstm_size, config.lstm_inner_dim, model_stats));
        const int decomposition = config.out_features;
        linear1 = register_module("linear1", LinearCRF(lstm_size, decomposition, config.bias, false));
        linear2 = register_module("linear2", LinearCRF(decomposition, config.outsize, false, true));
    } else if (config.has_out_features) {
        // v4.x model with linear decomposition
        rnns = register_module("rnns", LSTMStack(config.lstm_layers, lstm_size));
        const int decomposition = config.out_features;
        linear1 = register_module("linear1", LinearCRF(lstm_size, decomposition, true, false));
        linear2 = register_module("linear2", LinearCRF(decomposition, config.outsize, false, false));
        clamp1 = Clamp(-5.0, 5.0, config.clamp);
    } else if ((config.convs[0].size > 4) && (config.num_features == 1)) {
        // v4.x / v5.x model without linear decomposition
        rnns = register_module("rnns", LSTMStack(config.lstm_layers, lstm_size));
        linear1 = register_module("linear1", LinearCRF(lstm_size, config.outsize, false, false));
        clamp1 = Clamp(-5.0, 5.0, config.clamp);
    } else {
        // Pre-v4 model
        rnns = register_module("rnns", LSTMStack(config.lstm_layers, lstm_size));
        linear1 = register_module("linear1", LinearCRF(lstm_size, config.outsize, true, true));
    }

    if (model_stats && model_stats->calib_stats) {
        linear1->set_calib("linear1", model_stats->calib_stats);
        if (linear2) linear2->set_calib("linear2", model_stats->calib_stats);
    }
    if (model_stats && model_stats->quant_config) {
        const auto &cfg = *model_stats->quant_config;
        auto it = cfg.find("linear1");
        if (it != cfg.end()) linear1->set_quant_method(it->second);
        if (linear2) {
            it = cfg.find("linear2");
            if (it != cfg.end()) linear2->set_quant_method(it->second);
        }
    }
}

void CRFModelImpl::load_state_dict(const std::vector<torch::Tensor> &weights) {
    module_load_state_dict(*this, weights);
}

void CRFModelImpl::fuse_weights() {
    if (flstm_rnns) {
        flstm_rnns->fuse_weights();
    }
}

torch::Tensor CRFModelImpl::forward(const torch::Tensor &x) {
    const bool on_gpu = !x.device().is_cpu();
    double a, b;
    torch::Tensor h;

    a = realtime();
    h = convs->forward(x);
    if (on_gpu) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats_->time_conv_stack += b - a;

    a = realtime();
    if (flstm_rnns) {
        h = flstm_rnns->forward(h);
    } else {
        h = rnns->forward(h);
    }
    if (on_gpu) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats_->time_rnns += b - a;

    a = realtime();
    h = linear1->forward(h);
    if (on_gpu) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats_->time_crf_1 += b - a;

    if (linear2) {
        a = realtime();
        h = linear2->forward(h);
        if (on_gpu) torch::cuda::synchronize(x.device().index());
        b = realtime();
        model_stats_->time_crf_2 += b - a;
    }

    if (clamp1) {
        a = realtime();
        h = clamp1->forward(h);
        if (on_gpu) torch::cuda::synchronize(x.device().index());
        b = realtime();
        model_stats_->time_clamp += b - a;
    }

    // Output is [N, T, C]
    return h;
}

std::vector<torch::Tensor> load_lstm_model_weights(const CRFModelConfig &config) {
    const auto &dir = config.model_path;
    auto tensors = std::vector<std::string>{
        "0.conv.weight.tensor", "0.conv.bias.tensor",
        "1.conv.weight.tensor", "1.conv.bias.tensor",
        "2.conv.weight.tensor", "2.conv.bias.tensor",
    };

    // RNN layers start at sublayer index 4 (after 3 convs + 1 permute)
    const int rnn_start = 4;

    if (config.lstm_inner_dim >= 0) {
        // FLSTM: 6 parameters per layer (dn_weight_ih/hh, up_weight_ih/hh, up_bias_ih/hh)
        for (int i = 0; i < config.lstm_layers; ++i) {
            auto p = std::to_string(rnn_start + i) + ".rnn.";
            tensors.push_back(p + "dn_weight_ih.tensor");
            tensors.push_back(p + "dn_weight_hh.tensor");
            tensors.push_back(p + "up_weight_ih.tensor");
            tensors.push_back(p + "up_weight_hh.tensor");
            tensors.push_back(p + "up_bias_ih.tensor");
            tensors.push_back(p + "up_bias_hh.tensor");
        }
        // Intermediate linear then CRF linear (both bias=false)
        const int lin_idx = rnn_start + config.lstm_layers;
        tensors.push_back(std::to_string(lin_idx)     + ".linear.weight.tensor");
        tensors.push_back(std::to_string(lin_idx + 1) + ".linear.weight.tensor");
    } else {
        // Standard LSTM: 4 parameters per layer (weight_ih/hh, bias_ih/hh)
        for (int i = 0; i < config.lstm_layers; ++i) {
            auto p = std::to_string(rnn_start + i) + ".rnn.";
            tensors.push_back(p + "weight_ih_l0.tensor");
            tensors.push_back(p + "weight_hh_l0.tensor");
            tensors.push_back(p + "bias_ih_l0.tensor");
            tensors.push_back(p + "bias_hh_l0.tensor");
        }
        const int lin_idx = rnn_start + config.lstm_layers;
        tensors.push_back(std::to_string(lin_idx) + ".linear.weight.tensor");
        if (config.bias) {
            tensors.push_back(std::to_string(lin_idx) + ".linear.bias.tensor");
        }
        if (config.has_out_features) {
            tensors.push_back(std::to_string(lin_idx + 1) + ".linear.weight.tensor");
        }
    }

    return load_tensors(dir, tensors);
}

ModuleHolder<AnyModule> load_lstm_model(const CRFModelConfig &model_config, const torch::TensorOptions &options, lstm_stats_t *model_stats) {
    auto model = CRFModel(model_config, model_stats);
    auto state_dict = load_lstm_model_weights(model_config);
    model->load_state_dict(state_dict);
    model->to(options.dtype().toScalarType());
    model->to(options.device());
    model->eval();

    // Fuse FLSTM projection matrices after weights are loaded and on the correct device.
    model->fuse_weights();

    // Register weight-only stats for standard LSTM layers (no activation hooks possible).
    if (model_stats && model_stats->calib_stats) {
        for (const auto &named : model->named_parameters()) {
            const auto &n = named.key();
            if (n.find("weight_ih_l0") != std::string::npos ||
                n.find("weight_hh_l0") != std::string::npos) {
                model_stats->calib_stats->register_layer(n, named.value());
            }
        }
    }

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);
    return holder;
}