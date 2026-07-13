#include <math.h>
#include <string>

#include "lstm_model.h"
#include "error.h"
#include "misc.h"
#include "tensor_chunk_utils.h"
#include "quant.h"

#ifdef USE_GPU
#include <fluke/fluke.h>
#include <vector>
#endif

#if defined(HAVE_METAL)
#include "lstm_model_metal.h"

// Build the metal LSTM context for a plain-LSTM model: compile kernels and hand each layer its
// weights reordered into dorado's [3C+1, C, 4] tiled layout (U|W|W|bias, gates IFGO->GIFO, bias =
// bias_ih+bias_hh). Direction (and the dorado U/W swap) follow reverse_first alternation, matching
// slorado's flip-before-every-layer order. Returns nullptr if the config can't run on the kernel.
static void *build_metal_lstm(const lstm_model_t *m, int lstm_size) {
    metal_lstm_ctx_t *ctx = metal_lstm_create(lstm_size, (int)m->lstms.size(), /*reverse_first=*/1);
    if (!ctx) return nullptr;
    if (!metal_lstm_ok(ctx)) { metal_lstm_free(ctx); return nullptr; }
    const int C = lstm_size;
    for (size_t i = 0; i < m->lstms.size(); ++i) {
        const bool rev = metal_lstm_layer_reverse(ctx, (int)i) != 0;
        // Reorder on CPU in fp32 to avoid MPS view/op quirks; result fp16 [3C+1, C, 4].
        auto w_ih = m->lstms[i].w_ih.to(torch::kCPU).to(torch::kFloat32);
        auto w_hh = m->lstms[i].w_hh.to(torch::kCPU).to(torch::kFloat32);
        auto bias = (m->lstms[i].b_ih + m->lstms[i].b_hh).to(torch::kCPU).to(torch::kFloat32);
        auto t_w = (rev ? w_hh : w_ih).reshape({4, C, C}).transpose(1, 2);
        auto t_u = (rev ? w_ih : w_hh).reshape({4, C, C}).transpose(1, 2);
        auto t_b = bias.reshape({4, 1, C});
        auto comb = torch::cat({t_u, t_w, t_w, t_b}, 1);                       // [4, 3C+1, C]
        comb = torch::stack({comb[2], comb[0], comb[1], comb[3]}, 2);          // [3C+1, C, 4]
        comb = comb.to(torch::kFloat16).contiguous();
        metal_lstm_set_layer(ctx, (int)i, comb.data_ptr(), comb.numel() * sizeof(uint16_t));
    }

    // Conv stack: hand each conv layer its weights in dorado's padded [rows, out] layout so the whole
    // conv1->conv2->conv3 runs on the GPU (conv3 writes the LSTM layout). Only the exact fast/hac v5
    // shapes dorado's specialized kernels support; anything else leaves has_conv false -> ATen conv.
    if (m->convs.size() == 3) {
        // dorado's specialized kernels: conv1_in1_out16 (win5), conv2_in16_out16 (win5), conv3 (in16).
        auto shape = [](const conv_layer_t &c, int in, int out, int win) {
            return c.w.size(1) == in && c.w.size(0) == out && (win == 0 || c.w.size(2) == win);
        };
        bool shapes_ok = shape(m->convs[0], 1, 16, 5) && shape(m->convs[1], 16, 16, 5) &&
                         m->convs[2].w.size(1) == 16 && lstm_size == 96;   // verified: fast v5 only
        for (size_t i = 0; shapes_ok && i < 3; ++i) {
            const auto &cv = m->convs[i];
            const int out = (int)cv.w.size(0), in = (int)cv.w.size(1), win = (int)cv.w.size(2);
            const int pad_rows = (in == 1 && out == 16) ? 5 : (in == 4 && out == 16) ? 4 : 0;  // repeats=1
            const int rows = 2 * pad_rows + win * in + 1;
            auto tw = cv.w.to(torch::kCPU).to(torch::kFloat32).permute({2, 1, 0}).contiguous();   // [win,in,out]
            auto tb = torch::zeros({rows, out}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
            tb.narrow(0, pad_rows, win * in).view({win, in, out}).copy_(tw);
            tb.select(0, rows - 1).copy_(cv.b.to(torch::kCPU).to(torch::kFloat32));               // bias row
            auto tbf = tb.to(torch::kFloat16).contiguous();
            const int clamp = (cv.activation == Activation::SWISH_CLAMP) ? 1 : 0;
            metal_lstm_set_conv(ctx, (int)i + 1, in, out, win, cv.stride, clamp,
                                tbf.data_ptr(), tbf.numel() * sizeof(uint16_t));
        }
    }
    return ctx;
}
#endif

using namespace torch::nn;

void flatten_lstm_weights(lstm_layer_t &l, int input_size, int hidden, bool batch_first) {
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
    // ATen op; dispatches to cuDNN on CUDA and MIOpen on ROCm (both need flattened RNN weights).
    // Skipped on MPS (no cuDNN RNN backend): torch::lstm dispatches to _lstm_mps without it.
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

#if defined(HAVE_METAL)
    if (!dev.is_cpu()) {
        m->metal_ctx = build_metal_lstm(m, config.lstm_size);
        INFO("metal LSTM kernel %s", m->metal_ctx ? "enabled" : "unavailable (using ATen _lstm_mps)");
    }
#endif

    return m;
}

// conv + bidirectional LSTM stack; returns the LSTM output (pre-CRF). Full scores = lstm_model_crf().
at::Tensor lstm_model_forward_nocrf(const lstm_model_t *m, at::Tensor x) {
    const bool on_gpu = !x.device().is_cpu();
    const auto dev_idx = x.device().index();
    double a, b;

#if defined(HAVE_METAL)
    const bool fused_conv = m->metal_ctx && metal_lstm_has_conv((metal_lstm_ctx_t *)m->metal_ctx) && on_gpu;
#else
    const bool fused_conv = false;
#endif

    // conv stack: [N, C_in, T] -> [N, T, C_out]. Skipped when the Metal conv+LSTM block runs it on GPU.
    if (!fused_conv) {
        a = realtime();
        x = conv_stack_forward(m->convs, x);
        STAGE_SYNC(on_gpu, dev_idx);
        b = realtime();
        m->stats->time_conv_stack += b - a;
    }

    // bidirectional-alternating LSTM stack (flip time per layer, final flip if odd)
    a = realtime();
#if defined(HAVE_METAL)
    if (fused_conv) {
        // Fused conv+LSTM on GPU: x is the raw scaled signal [N,1,chunk]. Pass its MTLBuffer as
        // [N,chunk]; the metal block does conv1->conv2->conv3->lstm->reorder and returns [T,N,C].
        const int64_t N = x.size(0), chunk = x.size(2);
        const int64_t T = chunk / m->convs.back().stride;   // conv3 stride
        const int64_t C = m->lstm_size;
        const int64_t Npad = (N + 47) / 48 * 48;
        at::Tensor xp = x;                                  // [N,1,chunk]
        if (Npad != N) { xp = torch::zeros({Npad, 1, chunk}, x.options()); xp.narrow(0, 0, N).copy_(x); }
        auto sig = xp.reshape({Npad, chunk}).contiguous();  // [Npad,chunk] fp16 MPS, offset 0
        auto out = torch::empty({T, Npad, C}, x.options()); // [T,Npad,C] fp16 MPS
        torch::mps::synchronize();
        metal_lstm_run((metal_lstm_ctx_t *)m->metal_ctx, (int)Npad, (int)T,
                       sig.storage().data(), out.storage().data());
        auto y = out.transpose(0, 1).contiguous();          // [Npad,T,C]
        x = (Npad != N) ? y.narrow(0, 0, N).contiguous() : y;
    } else if (m->metal_ctx && metal_lstm_ok((metal_lstm_ctx_t *)m->metal_ctx) && on_gpu) {
        // Zero-copy Metal LSTM from the (ATen) conv output: hand the kernels the conv tensor's
        // MTLBuffer directly and write into another MPS tensor -- no host round-trip.
        const int64_t N = x.size(0), T = x.size(1), C = x.size(2);
        const int64_t Npad = (N + 47) / 48 * 48;
        at::Tensor xp = x;
        if (Npad != N) { xp = torch::zeros({Npad, T, C}, x.options()); xp.narrow(0, 0, N).copy_(x); }
        auto in = xp.transpose(0, 1).contiguous();          // [T,Npad,C] fp16 MPS, offset 0
        auto out = torch::empty({T, Npad, C}, xp.options());
        torch::mps::synchronize();
        metal_lstm_run((metal_lstm_ctx_t *)m->metal_ctx, (int)Npad, (int)T,
                       in.storage().data(), out.storage().data());
        auto y = out.transpose(0, 1).contiguous();
        x = (Npad != N) ? y.narrow(0, 0, N).contiguous() : y;
    } else
#endif
    {
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
    }
    STAGE_SYNC(on_gpu, dev_idx);
    b = realtime();
    m->stats->time_rnns += b - a;

    return x;   // LSTM output [N, T, lstm_size]; CRF applied by lstm_model_crf
}

at::Tensor lstm_model_forward(const lstm_model_t *m, at::Tensor x) {
    return lstm_model_crf(m, lstm_model_forward_nocrf(m, x));
}

// CRF linear (+ clamp): LSTM output [N,T,lstm_size] -> scores [N,T,outsize]. Split out from
// lstm_model_forward so the streaming pipeline can defer/overlap it with the next batch's conv+LSTM.
at::Tensor lstm_model_crf(const lstm_model_t *m, at::Tensor x) {
    const bool on_gpu = !x.device().is_cpu();
    const auto dev_idx = x.device().index();
    double a, b;

    a = realtime();
    x = at::linear(x, m->linear_w, m->linear_b);
    STAGE_SYNC(on_gpu, dev_idx);
    b = realtime();
    m->stats->time_crf_1 += b - a;

    if (m->clamp) {
        a = realtime();
        x.clamp_(m->clamp_min, m->clamp_max);
        STAGE_SYNC(on_gpu, dev_idx);
        b = realtime();
        m->stats->time_clamp += b - a;
    }
    return x;
}

void free_lstm_model(lstm_model_t *m) {
#if defined(HAVE_METAL)
    if (m && m->metal_ctx) metal_lstm_free((metal_lstm_ctx_t *)m->metal_ctx);
#endif
    delete m;
}

// --- procedural FLSTM model (hac/fast v6) -------------------------------------------------------

// The int8 FLSTM kernels live in fluke, which only builds on CUDA/ROCm; gate the whole quantized
// path on that (not USE_GPU) so the Metal/MPS build falls back to the fp16 ATen recurrence below.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
// Persistent per-(N,T) recurrence buffers, shared across all FLSTM layers. The recurrence itself —
// the T-step loop, the fused-vs-two-kernel choice, the hh|x concat, all per-step scratch, and
// CUDA-graph capture/replay — lives entirely in fluke (fluke_flstm_run_recurrence), so slorado holds
// only the model state: the int8 ring, the precomputed ih projection, and the cell. This keeps the
// layer device-agnostic (a HIP backend implements the same recurrence over hipGraph). H = hidden C,
// K = inner rank (K_hh == R).
typedef struct {
    int N, T;
    at::Tensor hh_all;     // [T+1, N, C] int8 ring (scale 1/127); boundary slot holds the zero state
    at::Tensor x_down;     // [T, N, K]  fp16 (ih down-projection, precomputed for the whole sequence)
    at::Tensor cell;       // [N, C]     fp32 (cell state, updated in place by the recurrence)
    at::Tensor x_scale;    // [T*N]      fp32 (const 1/127, for an int8-input layer's ih precompute)
    fluke_flstm_rec_t *rec;  // fluke-owned recurrence state (loop + graph + scratch); lazily created
} flstm_bufs_t;

struct flstm_qctx {
    std::vector<flstm_bufs_t> pool;  // keyed by (N,T); N varies only for a trailing partial batch
};

static flstm_bufs_t &get_flstm_bufs(flstm_qctx *qc, int N, int T, int C, int K, const at::TensorOptions &o) {
    for (auto &bf : qc->pool) if (bf.N == N && bf.T == T) return bf;
    flstm_bufs_t bf;
    bf.N = N; bf.T = T;
    bf.hh_all  = torch::empty({T + 1, N, C}, o.dtype(at::kChar));
    bf.x_down  = torch::empty({T, N, K},     o.dtype(at::kHalf));
    bf.cell    = torch::empty({N, C},        o.dtype(at::kFloat));
    bf.x_scale = torch::full({T * N}, 1.0f / 127.0f, o.dtype(at::kFloat));
    bf.rec     = nullptr;   // created on first use (needs the backend handle + layer count)
    qc->pool.push_back(bf);
    return qc->pool.back();
}
#endif

flstm_model_t *load_flstm_model_proc(const model_config_t &config, const torch::TensorOptions &options, lstm_stats_t *model_stats, const std::string &quant_mode) {
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

    // int8 kernel path: pre-quantize down weights, fuse the up-projection into per-gate weights.
    enum fluke_format_t quant_format = fluke_parse_format(quant_mode);
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
    if (quant_format != FLUKE_FORMAT_NONE && !dev.is_cpu()) {
        const int dev_idx = dev.index();
        const int H = m->C;      // hidden size
        const int R = m->K;      // input down-proj rank      (dn_w_ih: [R, H])
        const int K_hh = m->K;   // recurrent down-proj rank  (dn_w_hh: [K_hh, H])
        bool all_ok = true;
        for (auto &L : m->flstms) {
            L.backend = fluke_select_flstm(dev_idx, quant_format, H, K_hh, R);
            if (!L.backend) { all_ok = false; continue; }
            L.qw_dn_ih = quantize_tensor(L.dn_w_ih, 1);   // [R, H]    per-out-channel (R)
            L.qw_dn_hh = quantize_tensor(L.dn_w_hh, 1);   // [K_hh, H] per-out-channel (K_hh)
            // Fused step folds the fixed 1/127 activation dequant into the per-channel weight
            // scale so the kernel applies a single [K_hh] multiplier (host side, once per layer).
            L.hh_comb_scale = (L.qw_dn_hh.scale * (1.0f / 127.0f)).contiguous();  // [K_hh] f32
            // Fuse per gate g: gate_w[g] = [up_hh_g | up_ih_g] ([H, K_hh+R] fp16); the concat order
            // matches a_f16 = [hh_down | x_down]. gate_b[g] = up_b_ih_g + up_b_hh_g ([H] fp32).
            // Gate order i,f,g,o (matches the fp16 path's chunk(4)).
            for (int g = 0; g < 4; ++g) {
                auto hh_g = L.up_w_hh.slice(0, g * H, (g + 1) * H).contiguous();  // [H, K_hh]
                auto ih_g = L.up_w_ih.slice(0, g * H, (g + 1) * H).contiguous();  // [H, R]
                L.gate_w[g] = torch::cat({hh_g, ih_g}, 1).contiguous();          // [H, K_hh+R] fp16
                auto bias_g = L.up_b_ih.slice(0, g * H, (g + 1) * H)
                            + L.up_b_hh.slice(0, g * H, (g + 1) * H);
                L.gate_b[g] = bias_g.to(torch::kFloat32).contiguous();          // [H] fp32
            }
        }
        // All-or-nothing: if any layer lacks a kernel, fall back to fp16 for the whole stack (the
        // flip-free ring path assumes every layer is quantized).
        if (all_ok) {
            m->quant_ctx = new flstm_qctx();
        } else {
            for (auto &L : m->flstms) L.backend = nullptr;
        }
        INFO("quant '%s' FLSTM kernel path %s", quant_mode.c_str(), all_ok ? "enabled" : "unavailable (using fp16)");
    } else if (!quant_mode.empty() && quant_format == FLUKE_FORMAT_NONE) {
        WARNING("unknown quant mode '%s' — using fp16", quant_mode.c_str());
    }
#else
    if (quant_format != FLUKE_FORMAT_NONE) WARNING("quant mode '%s' requires a GPU build — using fp16", quant_mode.c_str());
#endif

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
    STAGE_SYNC(on_gpu, x.device().index());
    b = realtime();
    stats->time_flstm_precompute += b - a;

    torch::Tensor hh, dn_hh_all;

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
    // GPU path: fluke's per-step kernel writes directly into a persistent [T+1,N,C] ring buffer via
    // raw pointers, with mm_out/addmm_out reusing pre-allocated scratch to avoid per-step allocation.
    hh = torch::empty({T + 1, N, C}, x.options());
    hh[0].zero_();
    auto c        = torch::zeros({N, C},     x.options());
    auto scratch  = torch::empty({N, 4 * C}, x.options());
    auto dn_hh_buf = torch::empty({N, K},    x.options());
    if (L->cl_up_hh) dn_hh_all = torch::empty({T, N, K}, x.options());

    a = realtime();
    for (int t = 0; t < T; ++t) {
        torch::mm_out(dn_hh_buf, fake_quant(hh[t], lq_dn_hh.act), dn_w_hh.t());
        if (dn_hh_all.defined()) dn_hh_all[t] = dn_hh_buf;
        torch::addmm_out(scratch, L->up_b_hh, fake_quant(dn_hh_buf, lq_up_hh.act), up_w_hh_t);
        fluke_flstm_step_gpu(scratch.data_ptr(), ih[t].data_ptr(),
                                c.data_ptr(), hh[t + 1].data_ptr(), N, C);
    }
    b = realtime();
#else
    // CPU / MPS path: functional ops only. The GPU path's mm_out/addmm_out into persistent scratch
    // and the hh[t+1] = ... slice-assignment produce wrong results on the MPS backend (the recurrent
    // state does not propagate, collapsing the output to homopolymers). Build each step's hidden
    // state as an independent tensor and stack them at the end -- no in-place writes, no aliasing.
    std::vector<at::Tensor> hh_steps;
    hh_steps.reserve(T + 1);
    hh_steps.push_back(torch::zeros({N, C}, x.options()));  // hh[0]
    auto c = torch::zeros({N, C}, x.options());
    std::vector<at::Tensor> dn_hh_steps;
    if (L->cl_up_hh) dn_hh_steps.reserve(T);

    a = realtime();
    for (int t = 0; t < T; ++t) {
        auto dn_hh = torch::mm(fake_quant(hh_steps[t], lq_dn_hh.act), dn_w_hh.t());      // [N, K]
        if (L->cl_up_hh) dn_hh_steps.push_back(dn_hh);
        auto scratch = torch::addmm(L->up_b_hh, fake_quant(dn_hh, lq_up_hh.act), up_w_hh_t);  // [N, 4C]
        auto gates = scratch.add(ih[t]).chunk(4, 1);
        auto i = gates[0].mul(0.2f).add(0.5f).clamp(0.f, 1.f);
        auto f = gates[1].mul(0.2f).add(0.5f).clamp(0.f, 1.f);
        auto g = gates[2].clamp(-1.f, 1.f);
        auto o = gates[3].mul(0.2f).add(0.5f).clamp(0.f, 1.f);
        c = (f * c) + (i * g);
        hh_steps.push_back(o * torch::tanh(c));
    }
    hh = torch::stack(hh_steps, 0);                          // [T+1, N, C]
    if (L->cl_up_hh) dn_hh_all = torch::stack(dn_hh_steps, 0);  // [T, N, K]
    b = realtime();
#endif
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

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
// int8 FLSTM layer over the persistent ring buffers. input x [N, T, C] in NATURAL time order:
// fp16 for the first layer (quantized here per-token), else the previous layer's int8 hidden view
// (scale 1/127) — down-projected directly. No physical flips: even layers scan time in reverse via
// the ring indexing (matches the flip-based path's alternating direction). Output [N, T, C]:
// int8 view into the ring for intermediate layers (chained), fp16 for the last layer (fused dequant).
static at::Tensor flstm_layer_forward_quant(const flstm_model_t *m, const flstm_layer_t *L, int layer_idx, at::Tensor x, bool last) {
    lstm_stats_t *stats = m->stats;
    const int C = m->C, K = m->K;
    const int N = x.size(0), T = x.size(1);
    const int dev = x.device().index();
    const bool reverse = (layer_idx % 2 == 0);
    flstm_qctx *qc = (flstm_qctx *)m->quant_ctx;
    flstm_bufs_t &bufs = get_flstm_bufs(qc, N, T, C, K, x.options());
    double a, b;

    // ih precompute over the whole sequence, into the persistent x_down [T, N, K] buffer.
    a = realtime();
    if (x.scalar_type() == at::kChar) {
        // int8 input (chained hidden from the previous layer). The previous layer's output is a
        // [N,T,C] view over the ring's contiguous [T,N,C]; transpose(0,1) recovers that native
        // layout, so .contiguous() is a no-op (no copy). Then the int8 down-projection.
        auto x_flat = x.transpose(0, 1).contiguous().view({T * N, C});
        auto x_down_flat = bufs.x_down.view({T * N, K});
        fluke_flstm_down_proj_i8_into(L->backend, x_down_flat, x_flat, bufs.x_scale, L->qw_dn_ih);
    } else {
        // fp16 input (first layer = conv output, physically [N,C,T]). The old path did
        // x.transpose(0,1).contiguous() -> [T,N,C], but with C innermost at stride T that copy is
        // fully uncoalesced (~85ms/batch, ~90% of the whole precompute). Instead contract C on the
        // NATIVE [N,C,T] layout with a batched matmul (coalesced), then only the 8x-smaller [N,K,T]
        // output is reshaped to the recurrence's [T,N,K].
        auto x_down_nkt = at::matmul(L->dn_w_ih, x.transpose(1, 2));   // [K,C] x [N,C,T] -> [N,K,T]
        bufs.x_down.copy_(x_down_nkt.permute({2, 0, 1}));             // -> [T,N,K]
    }
    STAGE_SYNC(!x.device().is_cpu(), dev);
    b = realtime();
    stats->time_flstm_precompute += b - a;

    // Recurrence over the ring (direction by layer parity). Fully delegated to fluke: it runs the
    // T-step loop, picks fused vs two-kernel per device/N, does the hh|x concat, and captures/replays
    // a CUDA graph — all device-agnostic behind one call. Created lazily (needs the backend + layer
    // count); reused across batches (per-layer graph cache keyed inside the rec handle).
    a = realtime();
    if (bufs.rec == nullptr)
        bufs.rec = fluke_flstm_rec_create(L->backend, N, T, (int)m->flstms.size());
    fluke_flstm_run_recurrence(bufs.rec, layer_idx, bufs.hh_all, bufs.cell, bufs.x_down,
                               L->qw_dn_hh.tensor, L->hh_comb_scale, L->gate_w, L->gate_b, reverse);
    STAGE_SYNC(!x.device().is_cpu(), dev);
    b = realtime();
    stats->time_flstm_recurrence += b - a;

    // Natural-order hidden slice [T, N, C]. Chain int8 to the next layer; fused-dequant on the last.
    auto out_slice = reverse ? bufs.hh_all.slice(0, 0, T) : bufs.hh_all.slice(0, 1, T + 1);
    // Last layer: fused int8->fp16 dequant + transpose. A/B tested ~2% faster than manual ATen dequant.
    if (last) return fluke_dequant_int8_transpose(out_slice, 1.0f / 127.0f);  // [N, T, C] fp16
    return out_slice.permute({1, 0, 2});                                       // [N, T, C] int8 view
}
#endif

at::Tensor flstm_model_forward(const flstm_model_t *m, at::Tensor x) {
    const bool on_gpu = !x.device().is_cpu();
    const auto dev_idx = x.device().index();
    double a, b;

    a = realtime();
    x = conv_stack_forward(m->convs, x);
    STAGE_SYNC(on_gpu, dev_idx);
    b = realtime();
    m->stats->time_conv_stack += b - a;

    // bidirectional-alternating FLSTM stack. The int8 path is all-or-nothing (the loader either
    // engages every layer or none) and runs flip-free over a persistent ring (direction by parity);
    // the fp16 path flips per layer with a final flip if odd.
    a = realtime();
    const bool quant = !m->flstms.empty() && m->flstms[0].backend != nullptr;
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
    if (quant) {
        for (size_t i = 0; i < m->flstms.size(); ++i)
            x = flstm_layer_forward_quant(m, &m->flstms[i], (int)i, x, (i + 1 == m->flstms.size()));
    } else
#endif
    {
        for (size_t i = 0; i < m->flstms.size(); ++i) {
            x = flstm_layer_forward(&m->flstms[i], x.flip(1), m->stats, m->C, m->K);
        }
        if (m->flstms.size() & 1) x = x.flip(1);
    }
    STAGE_SYNC(on_gpu, dev_idx);
    b = realtime();
    m->stats->time_rnns += b - a;

    // decomposed CRF: linear1 then tanh-scaled linear2 (scale = LinearCRFImpl::scale = 5)
    a = realtime();
    x = at::linear(x, m->linear1_w, m->linear1_b);
    STAGE_SYNC(on_gpu, dev_idx);
    b = realtime();
    m->stats->time_crf_1 += b - a;

    a = realtime();
    auto scores = at::linear(x, m->linear2_w);
    if (g_scores_i8) {
        // int8 CRF emission scores (matches dorado): tanh is in [-1,1], so tanh*127 lands exactly in
        // the int8 range [-127,127] -- no clamp/calibration needed. Decoder rescales by 5/127.
        // In-place tanh_/mul_/round_ so only one fp16 buffer + the int8 output exist (a chained
        // tanh()*127.round().to() would materialize ~4 full fp16 score tensors and OOM at large N).
        x = scores.tanh_().mul_(127.0f).round_().to(torch::kChar);
    } else {
        x = torch::tanh(scores) * 5;
    }
    STAGE_SYNC(on_gpu, dev_idx);
    b = realtime();
    m->stats->time_crf_2 += b - a;

    return x;
}

// Release the per-(N,T) recurrence buffer pool (hh_all/x_down/cell + the fluke rec/graph).
// The pool is otherwise keep-forever, so it must be cleared between auto-batch trials -- each
// trial forwards at a different N and would otherwise leave that N's hh_all (GBs) resident,
// starving subsequent trials. The real run repopulates the pool lazily on first forward.
void free_flstm_bufs_pool(flstm_model_t *m) {
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
    if (m && m->quant_ctx) {
        flstm_qctx *qc = (flstm_qctx *)m->quant_ctx;
        for (auto &bf : qc->pool) if (bf.rec) fluke_flstm_rec_free(bf.rec);
        qc->pool.clear();
    }
#endif
}

void free_flstm_model(flstm_model_t *m) {
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
    if (m->quant_ctx) {
        flstm_qctx *qc = (flstm_qctx *)m->quant_ctx;
        for (auto &bf : qc->pool) if (bf.rec) fluke_flstm_rec_free(bf.rec);
        delete qc;
    }
#endif
    delete m;
}