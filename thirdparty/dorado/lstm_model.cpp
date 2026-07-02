#include <math.h>
#include <string>

#include "lstm_model.h"
#include "error.h"
#include "misc.h"
#include "tensor_chunk_utils.h"
#include "quant.h"

#ifdef USE_GPU
#include <fluke/fluke.h>
#endif

using namespace torch::nn;

void flatten_lstm_weights(lstm_layer_t &l, int input_size, int hidden, bool batch_first) {
#if defined(USE_GPU)
    // ATen op; dispatches to cuDNN on CUDA and MIOpen on ROCm (both need flattened RNN weights).
    if (l.w_ih.device().is_cpu()) return;
    l.w_ih = l.w_ih.contiguous();
    l.w_hh = l.w_hh.contiguous();
    l.b_ih = l.b_ih.contiguous();
    l.b_hh = l.b_hh.contiguous();
    std::vector<at::Tensor> ws = {l.w_ih, l.w_hh, l.b_ih, l.b_hh};
    // Packs ws (in place, via set_) into one contiguous buffer laid out for cuDNN; the returned
    // tensor owns the storage the w_* now view, so keep it alive on the layer.
    l.flat = at::_cudnn_rnn_flatten_weight(ws, /*weight_stride0=*/4, input_size,
                                           /*mode=CUDNN_LSTM*/ 2, hidden, /*proj_size=*/0,
                                           /*num_layers=*/1, batch_first, /*bidirectional=*/false);
#else
    (void)l; (void)input_size; (void)hidden; (void)batch_first;
#endif
}

// --- procedural model shared helpers ------------------------------------------------------------

// conv stack: [N, C_in, T] -> [N, T, C_out] (no timing/sync; caller wraps).
at::Tensor conv_stack_forward(const std::vector<conv_layer_t> &convs, at::Tensor x) {
    for (const auto &c : convs) {
        x = at::conv1d(x, c.w, c.b, c.stride, c.padding);
        if (c.activation == Activation::SWISH) {
            torch::silu_(x);
        } else if (c.activation == Activation::SWISH_CLAMP) {
            torch::silu_(x).clamp_(c10::nullopt, 3.5f);
        } else if (c.activation == Activation::TANH) {
            x.tanh_();
        } else {
            ERROR("%s", "Unrecognised activation function id.");
        }
    }
    return x.transpose(1, 2);
}

// Load the conv weight/bias tensors (files "<i>.conv.weight/bias.tensor") into conv_layer_t.
static void load_conv_layers(const model_config_t &config, const std::string &dir,
                             const torch::TensorOptions &options, std::vector<conv_layer_t> &out) {
    std::vector<std::string> names;
    for (size_t i = 0; i < config.convs.size(); ++i) {
        names.push_back(std::to_string(i) + ".conv.weight.tensor");
        names.push_back(std::to_string(i) + ".conv.bias.tensor");
    }
    auto tensors = load_tensors(dir, names);
    const auto dtype = options.dtype().toScalarType();
    const auto dev = options.device();
    size_t idx = 0;
    for (size_t i = 0; i < config.convs.size(); ++i) {
        conv_layer_t c;
        c.w = tensors[idx++].to(dtype).to(dev);
        c.b = tensors[idx++].to(dtype).to(dev);
        c.stride = config.convs[i].stride;
        c.padding = config.convs[i].winlen / 2;
        c.activation = config.convs[i].activation;
        out.push_back(c);
    }
}

// --- procedural plain-LSTM model (fast/hac v5) --------------------------------------------------

lstm_model_t *load_lstm_model_proc(const model_config_t &config, const torch::TensorOptions &options, lstm_stats_t *model_stats) {
    lstm_model_t *m = new lstm_model_t();
    m->stats = model_stats;
    m->lstm_size = config.lstm_size;
    m->clamp = config.clamp;
    m->clamp_min = -5.0f;
    m->clamp_max = 5.0f;

    // Build the weight-file name list in load order (mirrors load_lstm_model_weights, non-FLSTM).
    std::vector<std::string> names;
    for (size_t i = 0; i < config.convs.size(); ++i) {
        names.push_back(std::to_string(i) + ".conv.weight.tensor");
        names.push_back(std::to_string(i) + ".conv.bias.tensor");
    }
    const int rnn_start = 4;  // 3 convs + 1 permute
    for (int i = 0; i < config.lstm_layers; ++i) {
        auto p = std::to_string(rnn_start + i) + ".rnn.";
        names.push_back(p + "weight_ih_l0.tensor");
        names.push_back(p + "weight_hh_l0.tensor");
        names.push_back(p + "bias_ih_l0.tensor");
        names.push_back(p + "bias_hh_l0.tensor");
    }
    const int lin_idx = rnn_start + config.lstm_layers;
    names.push_back(std::to_string(lin_idx) + ".linear.weight.tensor");
    const bool has_lin_bias = config.bias;
    if (has_lin_bias) names.push_back(std::to_string(lin_idx) + ".linear.bias.tensor");

    auto tensors = load_tensors(config.model_path, names);
    const auto dtype = options.dtype().toScalarType();
    const auto dev = options.device();
    auto to_dev = [&](const at::Tensor &t) { return t.to(dtype).to(dev); };

    size_t idx = 0;
    for (size_t i = 0; i < config.convs.size(); ++i) {
        conv_layer_t c;
        c.w = to_dev(tensors[idx++]);
        c.b = to_dev(tensors[idx++]);
        c.stride = config.convs[i].stride;
        c.padding = config.convs[i].winlen / 2;
        c.activation = config.convs[i].activation;
        m->convs.push_back(c);
    }
    for (int i = 0; i < config.lstm_layers; ++i) {
        lstm_layer_t l;
        l.w_ih = to_dev(tensors[idx++]);
        l.w_hh = to_dev(tensors[idx++]);
        l.b_ih = to_dev(tensors[idx++]);
        l.b_hh = to_dev(tensors[idx++]);
        flatten_lstm_weights(l, config.lstm_size, config.lstm_size, /*batch_first=*/true);
        m->lstms.push_back(l);
    }
    m->linear_w = to_dev(tensors[idx++]);
    if (has_lin_bias) m->linear_b = to_dev(tensors[idx++]);

    return m;
}

at::Tensor lstm_model_forward(const lstm_model_t *m, at::Tensor x) {
    const bool on_gpu = !x.device().is_cpu();
    const auto dev_idx = x.device().index();
    double a, b;

    // conv stack: [N, C_in, T] -> [N, T, C_out]
    a = realtime();
    x = conv_stack_forward(m->convs, x);
#ifdef USE_GPU
    if (on_gpu) torch::cuda::synchronize(dev_idx);
#endif
    b = realtime();
    m->stats->time_conv_stack += b - a;

    // bidirectional-alternating LSTM stack (flip time per layer, final flip if odd)
    a = realtime();
    for (const auto &l : m->lstms) {
        auto flipped = x.flip(1);
        const int64_t N = flipped.size(0);
        auto h0 = torch::zeros({1, N, m->lstm_size}, flipped.options());
        auto c0 = torch::zeros({1, N, m->lstm_size}, flipped.options());
        x = std::get<0>(torch::lstm(flipped, {h0, c0}, {l.w_ih, l.w_hh, l.b_ih, l.b_hh},
                                    /*has_biases*/ true, /*num_layers*/ 1, /*dropout*/ 0.0,
                                    /*train*/ false, /*bidirectional*/ false, /*batch_first*/ true));
    }
    if (m->lstms.size() & 1) x = x.flip(1);
#ifdef USE_GPU
    if (on_gpu) torch::cuda::synchronize(dev_idx);
#endif
    b = realtime();
    m->stats->time_rnns += b - a;

    // CRF linear
    a = realtime();
    x = at::linear(x, m->linear_w, m->linear_b);
#ifdef USE_GPU
    if (on_gpu) torch::cuda::synchronize(dev_idx);
#endif
    b = realtime();
    m->stats->time_crf_1 += b - a;

    if (m->clamp) {
        a = realtime();
        x.clamp_(m->clamp_min, m->clamp_max);
#ifdef USE_GPU
        if (on_gpu) torch::cuda::synchronize(dev_idx);
#endif
        b = realtime();
        m->stats->time_clamp += b - a;
    }

    return x;
}

void free_lstm_model(lstm_model_t *m) {
    delete m;
}

// --- procedural FLSTM model (hac/fast v6) -------------------------------------------------------

flstm_model_t *load_flstm_model_proc(const model_config_t &config, const torch::TensorOptions &options, lstm_stats_t *model_stats) {
    if (model_stats && model_stats->quant_config) {
        build_quant_methods(model_stats->quant_methods, *model_stats->quant_config);
    }

    flstm_model_t *m = new flstm_model_t();
    m->stats = model_stats;
    m->C = config.lstm_size;
    m->K = config.lstm_inner_dim;

    const auto dtype = options.dtype().toScalarType();
    const auto dev = options.device();
    auto to_dev = [&](const at::Tensor &t) { return t.to(dtype).to(dev); };

    load_conv_layers(config, config.model_path, options, m->convs);

    const int rnn_start = 4;  // 3 convs + 1 permute
    for (int i = 0; i < config.lstm_layers; ++i) {
        auto p = std::to_string(rnn_start + i) + ".rnn.";
        auto names = std::vector<std::string>{
            p + "dn_weight_ih.tensor", p + "dn_weight_hh.tensor",
            p + "up_weight_ih.tensor", p + "up_weight_hh.tensor",
            p + "up_bias_ih.tensor",   p + "up_bias_hh.tensor",
        };
        auto t = load_tensors(config.model_path, names);
        flstm_layer_t L;
        L.dn_w_ih = to_dev(t[0]);
        L.dn_w_hh = to_dev(t[1]);
        L.up_w_ih = to_dev(t[2]);
        L.up_w_hh = to_dev(t[3]);
        L.up_b_ih = to_dev(t[4]);
        L.up_b_hh = to_dev(t[5]);
        L.prefix = std::string("rnns.rnn") + std::to_string(i + 1);
        // Register calib layers with the real (loaded) weights — no placeholder/update dance.
        if (model_stats && model_stats->calib_stats) {
            calib_stats_t *cs = model_stats->calib_stats;
            L.cl_dn_ih = cs->register_layer(L.prefix + ".dn_ih", L.dn_w_ih);
            L.cl_up_ih = cs->register_layer(L.prefix + ".up_ih", L.up_w_ih);
            L.cl_dn_hh = cs->register_layer(L.prefix + ".dn_hh", L.dn_w_hh);
            L.cl_up_hh = cs->register_layer(L.prefix + ".up_hh", L.up_w_hh);
        }
        m->flstms.push_back(std::move(L));
    }

    // Two decomposed CRF linears (both bias=false in v6 configs).
    const int lin_idx = rnn_start + config.lstm_layers;
    auto lin = load_tensors(config.model_path, {
        std::to_string(lin_idx)     + ".linear.weight.tensor",
        std::to_string(lin_idx + 1) + ".linear.weight.tensor",
    });
    m->linear1_w = to_dev(lin[0]);
    m->linear2_w = to_dev(lin[1]);

    return m;
}

// Single FLSTM layer: input [N, T, C] (already flipped by caller), output [N, T, C].
static at::Tensor flstm_layer_forward(const flstm_layer_t *L, at::Tensor x, lstm_stats_t *stats, int C, int K) {
    x = x.transpose(0, 1).contiguous();  // [T, N, C]
    const int T = x.size(0);
    const int N = x.size(1);
    const bool on_gpu = !x.device().is_cpu();
    double a, b;
    calib_stats_t *cs = stats->calib_stats;

    static const layer_quant_t k_empty_lq;
    auto lq = [&](const char *suffix) -> const layer_quant_t& {
        if (!L->prefix.empty() && !stats->quant_methods.empty()) {
            auto it = stats->quant_methods.find(L->prefix + suffix);
            if (it != stats->quant_methods.end()) return it->second;
        }
        return k_empty_lq;
    };
    const auto &lq_dn_ih = lq(".dn_ih");
    const auto &lq_up_ih = lq(".up_ih");
    const auto &lq_dn_hh = lq(".dn_hh");
    const auto &lq_up_hh = lq(".up_hh");

    // Cache (possibly fake-quantized) weights once outside the loop.
    auto dn_w_ih = fake_quant(L->dn_w_ih, lq_dn_ih.weight);          // [K, C]
    auto up_w_ih = fake_quant(L->up_w_ih, lq_up_ih.weight);          // [4*C, K]
    auto dn_w_hh = fake_quant(L->dn_w_hh, lq_dn_hh.weight);          // [K, C]
    auto up_w_hh_t = fake_quant(L->up_w_hh, lq_up_hh.weight).t().contiguous();  // [K, 4*C]

    // --- IH precompute: two matmuls over the full sequence ---
    a = realtime();
    auto x_flat = x.view({T * N, x.size(2)});
    if (L->cl_dn_ih) cs->accumulate(L->cl_dn_ih, x.transpose(0, 1));  // (N, T, C)

    auto dn_ih = at::linear(fake_quant(x_flat, lq_dn_ih.act), dn_w_ih);
    if (L->cl_up_ih) cs->accumulate(L->cl_up_ih, dn_ih.view({T, N, K}).transpose(0, 1).contiguous());

    auto ih = at::linear(fake_quant(dn_ih, lq_up_ih.act), up_w_ih, L->up_b_ih).view({T, N, 4 * C});
    if (on_gpu) torch::cuda::synchronize(x.device().index());
    b = realtime();
    stats->time_flstm_precompute += b - a;

    auto hh = torch::empty({T + 1, N, C}, x.options());
    hh[0].zero_();
    auto c        = torch::zeros({N, C},     x.options());
    auto scratch  = torch::empty({N, 4 * C}, x.options());
    auto dn_hh_buf = torch::empty({N, K},    x.options());
    torch::Tensor dn_hh_all;
    if (L->cl_up_hh) dn_hh_all = torch::empty({T, N, K}, x.options());

    a = realtime();
    for (int t = 0; t < T; ++t) {
        torch::mm_out(dn_hh_buf, fake_quant(hh[t], lq_dn_hh.act), dn_w_hh.t());
        if (dn_hh_all.defined()) dn_hh_all[t] = dn_hh_buf;
        torch::addmm_out(scratch, L->up_b_hh, fake_quant(dn_hh_buf, lq_up_hh.act), up_w_hh_t);

#ifdef USE_GPU
        fluke_flstm_step_gpu(scratch.data_ptr(), ih[t].data_ptr(),
                                c.data_ptr(), hh[t + 1].data_ptr(), N, C);
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
    stats->time_flstm_recurrence += b - a;

    using namespace torch::indexing;
    if (L->cl_dn_hh) {
        cs->accumulate(L->cl_dn_hh, hh.index({Slice(0, T)}).transpose(0, 1).contiguous());
    }
    if (L->cl_up_hh && dn_hh_all.defined()) {
        cs->accumulate(L->cl_up_hh, dn_hh_all.transpose(0, 1).contiguous());
    }

    return hh.index({Slice(1, None)}).transpose(0, 1).contiguous();  // [N, T, C]
}

at::Tensor flstm_model_forward(const flstm_model_t *m, at::Tensor x) {
    const bool on_gpu = !x.device().is_cpu();
    const auto dev_idx = x.device().index();
    double a, b;

    a = realtime();
    x = conv_stack_forward(m->convs, x);
#ifdef USE_GPU
    if (on_gpu) torch::cuda::synchronize(dev_idx);
#endif
    b = realtime();
    m->stats->time_conv_stack += b - a;

    // bidirectional-alternating FLSTM stack (flip time per layer, final flip if odd)
    a = realtime();
    for (const auto &L : m->flstms) {
        x = flstm_layer_forward(&L, x.flip(1), m->stats, m->C, m->K);
    }
    if (m->flstms.size() & 1) x = x.flip(1);
#ifdef USE_GPU
    if (on_gpu) torch::cuda::synchronize(dev_idx);
#endif
    b = realtime();
    m->stats->time_rnns += b - a;

    // decomposed CRF: linear1 then tanh-scaled linear2 (scale = LinearCRFImpl::scale = 5)
    a = realtime();
    x = at::linear(x, m->linear1_w, m->linear1_b);
#ifdef USE_GPU
    if (on_gpu) torch::cuda::synchronize(dev_idx);
#endif
    b = realtime();
    m->stats->time_crf_1 += b - a;

    a = realtime();
    auto scores = at::linear(x, m->linear2_w);
    x = torch::tanh(scores) * 5;
#ifdef USE_GPU
    if (on_gpu) torch::cuda::synchronize(dev_idx);
#endif
    b = realtime();
    m->stats->time_crf_2 += b - a;

    return x;
}

void free_flstm_model(flstm_model_t *m) {
    delete m;
}