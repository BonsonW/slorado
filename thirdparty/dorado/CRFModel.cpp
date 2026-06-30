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

torch::Tensor LinearCRFImpl::forward(const torch::Tensor &x) {
    // Input x is [N, T, C], contiguity optional
    auto scores = at::linear(x, linear->weight, linear->bias);
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

FLSTMLayerImpl::FLSTMLayerImpl(int C, int K, lstm_stats_t *model_stats, const std::string &name_prefix) : C_(C), K_(K), model_stats_(model_stats) {
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
            // dn weights are [K, C] = [out, in]; up weights are [4*C, K] = [out, in]
            cl_dn_ih_ = calib_stats_->register_layer(name_prefix + ".dn_ih", dn_weight_ih_);
            cl_up_ih_ = calib_stats_->register_layer(name_prefix + ".up_ih", up_weight_ih_);
            cl_dn_hh_ = calib_stats_->register_layer(name_prefix + ".dn_hh", dn_weight_hh_);
            cl_up_hh_ = calib_stats_->register_layer(name_prefix + ".up_hh", up_weight_hh_);
        }
    }
}

torch::Tensor FLSTMLayerImpl::forward(torch::Tensor x) {
    // x is [N, T, C]
    x = x.transpose(0, 1).contiguous();  // [T, N, C]
    const int T = x.size(0);
    const int N = x.size(1);
    const bool on_gpu = !x.device().is_cpu();
    double a, b;

    static const layer_quant_t k_empty_lq;
    auto lq = [&](const char *suffix) -> const layer_quant_t& {
        if (!calib_prefix_.empty() && !model_stats_->quant_methods.empty()) {
            auto it = model_stats_->quant_methods.find(calib_prefix_ + suffix);
            if (it != model_stats_->quant_methods.end()) return it->second;
        }
        return k_empty_lq;
    };
    const auto &lq_dn_ih = lq(".dn_ih");
    const auto &lq_up_ih = lq(".up_ih");
    const auto &lq_dn_hh = lq(".dn_hh");
    const auto &lq_up_hh = lq(".up_hh");

    // Cache (possibly fake-quantized) weights once outside the loop.
    auto dn_w_ih = fake_quant(dn_weight_ih_, lq_dn_ih.weight);  // [K, C]
    auto up_w_ih = fake_quant(up_weight_ih_, lq_up_ih.weight);  // [4*C, K]
    auto dn_w_hh = fake_quant(dn_weight_hh_, lq_dn_hh.weight);  // [K, C]
    auto up_w_hh_t = fake_quant(up_weight_hh_, lq_up_hh.weight).t().contiguous();  // [K, 4*C]

    // --- IH precompute: two matmuls over the full sequence ---
    a = realtime();
    auto x_flat = x.view({T * N, x.size(2)});
    if (cl_dn_ih_) calib_stats_->accumulate(cl_dn_ih_, x.transpose(0, 1));  // (N, T, C)

    // dn_ih: [T*N, C] @ [C, K] -> [T*N, K]
    auto dn_ih = at::linear(fake_quant(x_flat, lq_dn_ih.act), dn_w_ih);
    if (cl_up_ih_) calib_stats_->accumulate(cl_up_ih_, dn_ih.view({T, N, K_}).transpose(0, 1).contiguous());

    // ih: [T*N, K] @ [K, 4*C] + bias -> [T, N, 4*C]
    auto ih = at::linear(fake_quant(dn_ih, lq_up_ih.act), up_w_ih, up_bias_ih_).view({T, N, 4 * C_});
    if (on_gpu) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats_->time_flstm_precompute += b - a;

    auto hh = torch::empty({T + 1, N, C_}, x.options());
    hh[0].zero_();
    auto c        = torch::zeros({N, C_},     x.options());
    auto scratch  = torch::empty({N, 4 * C_}, x.options());
    auto dn_hh_buf = torch::empty({N, K_},   x.options());
    // Collect dn_hh intermediates for up_hh calib (only when calibrating).
    torch::Tensor dn_hh_all;
    if (cl_up_hh_) dn_hh_all = torch::empty({T, N, K_}, x.options());

    a = realtime();
    for (int t = 0; t < T; ++t) {
        // Down project hh: [N, C] @ [C, K] -> [N, K]
        torch::mm_out(dn_hh_buf, fake_quant(hh[t], lq_dn_hh.act), dn_w_hh.t());
        if (dn_hh_all.defined()) dn_hh_all[t] = dn_hh_buf;
        // Up project hh: [N, K] @ [K, 4*C] + bias -> [N, 4*C]
        torch::addmm_out(scratch, up_bias_hh_, fake_quant(dn_hh_buf, lq_up_hh.act), up_w_hh_t);

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
    }
    b = realtime();
    model_stats_->time_flstm_recurrence += b - a;

    using namespace torch::indexing;

    // Post-loop calib accumulation.
    if (cl_dn_hh_) {
        // hh[0..T-1] are the hidden states fed into dn_weight_hh_ each step.
        calib_stats_->accumulate(cl_dn_hh_, hh.index({Slice(0, T)}).transpose(0, 1).contiguous());
    }
    if (cl_up_hh_ && dn_hh_all.defined()) {
        calib_stats_->accumulate(cl_up_hh_, dn_hh_all.transpose(0, 1).contiguous());
    }

    // Return [N, T, C]
    return hh.index({Slice(1, None)}).transpose(0, 1).contiguous();
}

void FLSTMLayerImpl::update_calib_weights() {
    if (!calib_stats_) return;
    calib_stats_->update_weight(cl_dn_ih_, dn_weight_ih_);
    calib_stats_->update_weight(cl_up_ih_, up_weight_ih_);
    calib_stats_->update_weight(cl_dn_hh_, dn_weight_hh_);
    calib_stats_->update_weight(cl_up_hh_, up_weight_hh_);
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

}

void CRFModelImpl::load_state_dict(const std::vector<torch::Tensor> &weights) {
    module_load_state_dict(*this, weights);
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
    if (model_stats && model_stats->quant_config) {
        build_quant_methods(model_stats->quant_methods, *model_stats->quant_config);
    }
    auto model = CRFModel(model_config, model_stats);
    auto state_dict = load_lstm_model_weights(model_config);
    model->load_state_dict(state_dict);
    model->to(options.dtype().toScalarType());
    model->to(options.device());
    model->eval();

    // Update FLSTM calib weight stats now that real weights are loaded.
    // (register_layer is called during construction with torch::empty() placeholders.)
    if (model_stats && model_stats->calib_stats && model->flstm_rnns) {
        for (auto &layer : model->flstm_rnns->layers_) {
            layer->update_calib_weights();
        }
    }

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);
    return holder;
}