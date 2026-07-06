#include "tx_model.h"
#include "misc.h"
#include "quant.h"

#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>
#include <c10/core/ScalarType.h>
#include <torch/nn/functional/padding.h>
#include <torch/nn/options/padding.h>
#include <torch/types.h>
#include <torch/version.h>

#include <cmath>
#include <ATen/ops/scaled_dot_product_attention.h>

#include <stdexcept>
#include <string>

using namespace torch::nn;
using Slice = torch::indexing::Slice;

// =============================== procedural transformer model ===================================

// Probe whether flash attention actually runs on this device/dtype/head_dim by executing a tiny
// forward pass and catching failures. Returns false on CPU, unsupported builds, or any throw.
static bool flash_attn_supported(int head_dim, const torch::TensorOptions &options) {
#if defined USE_GPU && ((TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 4) || TORCH_VERSION_MAJOR >= 3)
    if (options.device().is_cpu()) return false;
    try {
        auto probe_opts = options.dtype(torch::kHalf);
        auto q = torch::zeros({1, 8, 1, head_dim}, probe_opts);
        auto k = torch::zeros({1, 8, 1, head_dim}, probe_opts);
        auto v = torch::zeros({1, 8, 1, head_dim}, probe_opts);
        float softmax_scale = 1.0 / std::sqrt((double)head_dim);
        auto res = at::_flash_attention_forward(
            q, k, v, std::nullopt, std::nullopt, 8, 8,
            0.0, false, false, softmax_scale, -1, -1, std::nullopt, std::nullopt);
        std::get<0>(res).sum().item(); // force execution so lazy/async errors surface here
        return true;
    } catch (const std::exception &e) {
        INFO("flash attention probe failed, falling back to SDPA: %s", e.what());
        return false;
    }
#else
    (void)head_dim; (void)options;
    return false;
#endif
}

tx_model_t *load_tx_model_proc(const model_config_t &config, const torch::TensorOptions &options, tx_stats_t *model_stats, bool use_flash, const std::string &quant_mode, int nthreads) {
    tx_model_t *m = new tx_model_t();
    m->stats = model_stats;
    const auto &txp = config.tx.tx;
    const int depth = txp.depth;
    const int d_model = txp.d_model, nhead = txp.nhead, head_dim = d_model / nhead;

    // flash requested (default on) is validated by a one-shot probe against this device/dtype/head_dim.
    if (use_flash) use_flash = flash_attn_supported(head_dim, options);

    if (model_stats) {
        model_stats->use_flash = use_flash;
        model_stats->nthreads = nthreads;
        if (model_stats->quant_config) build_quant_methods(model_stats->quant_methods, *model_stats->quant_config);
    }

    const auto dtype = options.dtype().toScalarType();
    const auto dev = options.device();
    auto to_dev = [&](const at::Tensor &t) { return t.to(dtype).to(dev); };

    // conv stack (5 layers), file prefix "conv.<i>."
    {
        std::vector<std::string> names;
        for (size_t i = 0; i < config.convs.size(); ++i) {
            names.push_back("conv." + std::to_string(i) + ".conv.weight.tensor");
            names.push_back("conv." + std::to_string(i) + ".conv.bias.tensor");
        }
        auto t = load_tensors(config.model_path, names);
        size_t idx = 0;
        for (size_t i = 0; i < config.convs.size(); ++i) {
            conv_layer_t c;
            c.w = to_dev(t[idx++]);
            c.b = to_dev(t[idx++]);
            c.stride = config.convs[i].stride;
            c.padding = config.convs[i].winlen / 2;
            c.activation = config.convs[i].activation;
            m->convs.push_back(c);
        }
    }

    // rotary sin/cos (fp32, on device) — matches RotaryEmbeddingImpl ctor (theta=10000, max=2048)
    const float theta = 10000.0f;
    const int64_t max_seq_len = 2048;
    auto inv_freq = torch::pow(theta, torch::arange(0, head_dim, 2, options) / head_dim).reciprocal();
    auto freqs = torch::arange(max_seq_len, options).outer(inv_freq);
    at::Tensor rot_cos = torch::cos(freqs).to(torch::kFloat32).contiguous();
    at::Tensor rot_sin = torch::sin(freqs).to(torch::kFloat32).contiguous();

    calib_stats_t *cs = model_stats ? model_stats->calib_stats : nullptr;

    for (int i = 0; i < depth; ++i) {
        const std::string p = "transformer_encoder." + std::to_string(i) + ".";
        auto t = load_tensors(config.model_path, {
            p + "self_attn.Wqkv.weight.tensor",
            p + "self_attn.out_proj.weight.tensor",
            p + "self_attn.out_proj.bias.tensor",
            p + "ff.fc1.weight.tensor",
            p + "ff.fc2.weight.tensor",
            p + "norm1.weight.tensor",
            p + "norm2.weight.tensor",
        });
        tx_layer_t L;
        L.rot_sin = rot_sin;
        L.rot_cos = rot_cos;
        L.d_model = d_model;
        L.nhead = nhead;
        L.head_dim = head_dim;
        L.num_splits = 12;
        L.attn_window = txp.attn_window;
        L.wqkv_w     = to_dev(t[0]);
        L.out_proj_w = to_dev(t[1]);
        L.out_proj_b = to_dev(t[2]);
        L.fc1_w      = to_dev(t[3]);
        L.fc2_w      = to_dev(t[4]);
        L.norm1_w    = to_dev(t[5]);
        L.norm2_w    = to_dev(t[6]);
        L.hidden_features = txp.dim_feedforward;
        // Match the torch::nn model: deepnorm_alpha is a registered buffer, so model->to(dtype)
        // rounds it to the runner dtype (fp16 on GPU) before RMSNorm reads it.
        L.deepnorm_alpha = torch::tensor(txp.deepnorm_alpha).to(dtype).item<float>();
        L.attn_prefix = p + "self_attn";
        L.ff_prefix = p + "ff";
        if (cs) {
            L.cl_wqkv     = cs->register_layer(L.attn_prefix + ".wqkv",     L.wqkv_w);
            L.cl_out_proj = cs->register_layer(L.attn_prefix + ".out_proj", L.out_proj_w);
            L.cl_fc1      = cs->register_layer(L.ff_prefix + ".fc1",        L.fc1_w);
            L.cl_fc2      = cs->register_layer(L.ff_prefix + ".fc2",        L.fc2_w);
        }
        m->layers.push_back(std::move(L));
    }

    // upsample (tx_decoder) + CRF (pre-scale the weight at load, matching LinearScaledCRF)
    {
        auto t = load_tensors(config.model_path, {
            "upsample.linear.weight.tensor", "upsample.linear.bias.tensor", "crf.linear.weight.tensor",
        });
        m->up_w = to_dev(t[0]);
        m->up_b = to_dev(t[1]);
        m->scale_factor = config.tx.upsample.scale_factor;
        m->crf_w = to_dev(t[2]);
        m->crf_w = m->crf_w * config.tx.crf.scale;
    }

    // int8 kernel backends (engages only if a backend is available for every layer).
    enum fluke_format_t quant_format = fluke_parse_format(quant_mode);
#ifdef USE_GPU
    if (quant_format != FLUKE_FORMAT_NONE && !dev.is_cpu()) {
        fluke_dims_t dims{d_model, txp.dim_feedforward, nhead, head_dim, /*max_seq=*/1024};
        const int dev_idx = dev.index();
        const int64_t H = txp.dim_feedforward;
        bool all_ok = true;
        for (auto &L : m->layers) {
            L.attn_backend = fluke_select_backend(dev_idx, quant_format, dims);
            if (L.attn_backend) L.qw_wqkv = quantize_tensor(L.wqkv_w, 1);
            L.ff_backend = fluke_select_backend(dev_idx, quant_format, dims);
            if (L.ff_backend) {
                auto up_w   = L.fc1_w.slice(0, 0, H).contiguous();
                auto gate_w = L.fc1_w.slice(0, H, 2 * H).contiguous();
                L.qw_up   = quantize_tensor(up_w,   1);
                L.qw_gate = quantize_tensor(gate_w, 1);
            }
            if (!L.attn_backend || !L.ff_backend) all_ok = false;
        }
        m->quant_stream = all_ok;
        INFO("quant '%s' kernel path %s", quant_mode.c_str(), all_ok ? "enabled" : "unavailable (using fp16)");
    } else if (!quant_mode.empty() && quant_format == FLUKE_FORMAT_NONE) {
        WARNING("unknown quant mode '%s' — using fp16", quant_mode.c_str());
    }
#else
    if (quant_format != FLUKE_FORMAT_NONE) WARNING("quant mode '%s' requires a GPU build — using fp16", quant_mode.c_str());
#endif

    INFO("%s", use_flash ? "flash attention enabled" : "flash attention disabled");
    return m;
}

static const layer_quant_t *lq_lookup(tx_stats_t *stats, const std::string &prefix, const char *suffix) {
    if (stats && !prefix.empty() && !stats->quant_methods.empty()) {
        auto it = stats->quant_methods.find(prefix + suffix);
        if (it != stats->quant_methods.end()) return &it->second;
    }
    return nullptr;
}

static at::Tensor tx_rotary(const tx_layer_t *L, at::Tensor qkv, tx_stats_t *stats) {
    const int batch = qkv.size(0), seqlen = qkv.size(1), nheads = qkv.size(3), head_dim = qkv.size(4);
    const int rotary_dim = 32;
    const int sb = qkv.stride(0), ss = qkv.stride(1), sh = qkv.stride(3);
    auto ch = qkv.chunk(3, 2);
#ifdef USE_GPU
    if (!qkv.device().is_cpu()) {
        fluke_rotary_emb_gpu(ch[0].data_ptr(), L->rot_sin.data_ptr(), L->rot_cos.data_ptr(), batch, seqlen, nheads, head_dim, rotary_dim, sb, ss, sh);
        fluke_rotary_emb_gpu(ch[1].data_ptr(), L->rot_sin.data_ptr(), L->rot_cos.data_ptr(), batch, seqlen, nheads, head_dim, rotary_dim, sb, ss, sh);
    } else
#endif
    {
        fluke_rotary_emb_cpu(ch[0].data_ptr(), L->rot_sin.data_ptr(), L->rot_cos.data_ptr(), batch, seqlen, nheads, head_dim, rotary_dim, sb, ss, sh, stats->nthreads);
        fluke_rotary_emb_cpu(ch[1].data_ptr(), L->rot_sin.data_ptr(), L->rot_cos.data_ptr(), batch, seqlen, nheads, head_dim, rotary_dim, sb, ss, sh, stats->nthreads);
    }
    return qkv;
}

static at::Tensor tx_get_mask(tx_model_t *m, const tx_layer_t *L, int64_t size, const torch::Device &device) {
    const auto key = MaskKey{size, device};
    auto it = m->mask_cache.find(key);
    if (it != m->mask_cache.end()) return it->second;
    const auto win_upper = std::get<0>(L->attn_window);
    const auto win_lower = std::get<1>(L->attn_window);
    torch::Tensor mask = torch::ones({size, size}, device);
    mask.triu_(-win_upper).tril_(win_lower);
    mask = mask.to(torch::kBool);
    m->mask_cache[key] = mask;
    return mask;
}

// Shared fp16 attention core: SDPA / flash + out_proj (qlinear). qkv: fp16 [N,T,3,nhead,head_dim].
static at::Tensor tx_attn_tail(tx_model_t *m, const tx_layer_t *L, torch::Tensor qkv, const layer_quant_t *lq_op) {
    tx_stats_t *stats = m->stats;
    const int64_t N = qkv.size(0), T = qkv.size(1), C = L->d_model;
    const int head_dim = L->head_dim, nhead = L->nhead, num_splits = L->num_splits;
    const bool on_gpu = !qkv.device().is_cpu();
    double a, b;

    a = realtime();
    const auto win_upper = std::get<0>(L->attn_window);
    const auto win_lower = std::get<1>(L->attn_window);
    torch::Tensor attn_output_ntc;
#if defined USE_GPU && ((TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 4) || TORCH_VERSION_MAJOR >= 3)
    if (stats->use_flash) {
        float softmax_scale = 1.0 / std::sqrt(head_dim);
        auto qkv_chunks = qkv.chunk(3, 2);
        auto q = qkv_chunks[0].squeeze(2);
        auto k = qkv_chunks[1].squeeze(2);
        auto v = qkv_chunks[2].squeeze(2);
        auto flash_res = at::_flash_attention_forward(
            q, k, v, std::nullopt, std::nullopt, qkv.size(1), qkv.size(1),
            0.0, false, false, softmax_scale, win_lower, win_upper, std::nullopt, std::nullopt);
        attn_output_ntc = std::get<0>(flash_res).reshape({N, T, C});
    } else
#endif
    {
        qkv = qkv.permute({2, 0, 3, 1, 4});
        attn_output_ntc = torch::empty({N, T, C}, qkv.options());
        auto attn_window_mask = tx_get_mask(m, L, T, qkv.device());
        auto attn_output = attn_output_ntc.view({N, T, nhead, head_dim}).transpose(1, 2);
        const auto elems_per_split = pad_to(div_round_up(T, int64_t{num_splits}), int64_t{4});
        for (int i = 0; i < num_splits; ++i) {
            const auto qb = i * elems_per_split;
            if (qb >= T) break;
            const auto qe = std::min(T, qb + elems_per_split);
            const auto kvb = std::max<int64_t>(0, qb - win_lower);
            const auto kve = std::min<int64_t>(T, qe + win_upper);
            const auto q = qkv[0].slice(-2, qb, qe);
            const auto k = qkv[1].slice(-2, kvb, kve);
            const auto v = qkv[2].slice(-2, kvb, kve);
            const auto mask = attn_window_mask.index({Slice(qb, qe), Slice(kvb, kve)});
            c10::optional<torch::Tensor> opt_mask = mask;
            attn_output.slice(-2, qb, qe) = torch::scaled_dot_product_attention(q, k, v, opt_mask);
        }
    }
    STAGE_SYNC(on_gpu, qkv.device().index());
    b = realtime();
    stats->time_sdp_attn += b - a;

    a = realtime();
    auto out = qlinear({L->out_proj_w, L->out_proj_b, lq_op, stats->calib_stats, L->cl_out_proj}, attn_output_ntc);
    STAGE_SYNC(on_gpu, qkv.device().index());
    b = realtime();
    stats->time_out_proj += b - a;
    return out;
}

static at::Tensor tx_mha_forward(tx_model_t *m, const tx_layer_t *L, torch::Tensor x) {
    tx_stats_t *stats = m->stats;
    const int64_t N = x.size(0), T = x.size(1);
    const bool on_gpu = !x.device().is_cpu();
    double a, b;

    const layer_quant_t *lq_wqkv = lq_lookup(stats, L->attn_prefix, ".wqkv");
    const layer_quant_t *lq_op   = lq_lookup(stats, L->attn_prefix, ".out_proj");

    a = realtime();
    auto qkv = qlinear({L->wqkv_w, at::Tensor(), lq_wqkv, stats->calib_stats, L->cl_wqkv}, x)
                   .view({N, T, 3, L->nhead, L->head_dim});
    STAGE_SYNC(on_gpu, x.device().index());
    b = realtime();
    stats->time_mm += b - a;

    a = realtime();
    qkv = tx_rotary(L, qkv, stats);
    STAGE_SYNC(on_gpu, x.device().index());
    b = realtime();
    stats->time_rotary_emb += b - a;

    return tx_attn_tail(m, L, qkv, lq_op);
}

static at::Tensor tx_mha_forward_quant(tx_model_t *m, const tx_layer_t *L, const tensor_quant_t &x) {
    tx_stats_t *stats = m->stats;
    const layer_quant_t *lq_op = lq_lookup(stats, L->attn_prefix, ".out_proj");
    const bool on_gpu = !x.tensor.device().is_cpu();
    double a = realtime();
    auto qkv = fluke_qkv_rotary_i8(L->attn_backend, x, L->qw_wqkv, L->rot_sin, L->rot_cos);
    STAGE_SYNC(on_gpu, x.tensor.device().index());
    stats->time_mm += realtime() - a;
    return tx_attn_tail(m, L, qkv, lq_op);
}

static at::Tensor tx_gmlp_forward(const tx_layer_t *L, torch::Tensor x, tx_stats_t *stats) {
    const layer_quant_t *lq_fc1 = lq_lookup(stats, L->ff_prefix, ".fc1");
    const layer_quant_t *lq_fc2 = lq_lookup(stats, L->ff_prefix, ".fc2");
    calib_stats_t *cs = stats ? stats->calib_stats : nullptr;

    const bool on_gpu = !x.device().is_cpu();
    const int dev = x.device().index();
    double t0 = realtime();
    torch::Tensor t = qlinear({L->fc1_w, at::Tensor(), lq_fc1, cs, L->cl_fc1}, x);
#ifdef USE_GPU
    auto M = t.size(0) * t.size(1);
    auto K = t.size(2) / 2;
    auto silu_o = torch::empty({t.size(0), t.size(1), K}, t.options());
    fluke_silu_mul_gpu(t.data_ptr(), silu_o.data_ptr(), M, K);
    t = silu_o;
#else
    const auto chunks = t.chunk(2, -1);
    t = functional::silu(chunks[1]).mul_(chunks[0]);
#endif
    STAGE_SYNC(on_gpu, dev);
    if (stats) stats->time_ff_gmlp += realtime() - t0;

    t0 = realtime();
    auto out = qlinear({L->fc2_w, at::Tensor(), lq_fc2, cs, L->cl_fc2}, t);
    STAGE_SYNC(on_gpu, dev);
    if (stats) stats->time_ff_down += realtime() - t0;
    return out;
}

static at::Tensor tx_gmlp_forward_quant(const tx_layer_t *L, const tensor_quant_t &x, tx_stats_t *stats) {
    const int dev = x.tensor.device().index();
    double t0 = realtime();
    auto g = fluke_gated_mlp_i8(L->ff_backend, x, L->qw_gate, L->qw_up);
    STAGE_SYNC(true, dev);
    if (stats) stats->time_ff_gmlp += realtime() - t0;

    const layer_quant_t *lq_fc2 = lq_lookup(stats, L->ff_prefix, ".fc2");
    t0 = realtime();
    auto out = qlinear({L->fc2_w, at::Tensor(), lq_fc2, nullptr, nullptr}, g);
    STAGE_SYNC(true, dev);
    if (stats) stats->time_ff_down += realtime() - t0;
    return out;
}

static void tx_encoder_forward(tx_model_t *m, const tx_layer_t *L, torch::Tensor &x) {
    tx_stats_t *stats = m->stats;
    const float alpha = L->deepnorm_alpha;
    const float eps = 1e-5f;
    double a, b;

    auto run_norm = [&](const at::Tensor &norm_w, const torch::Tensor &in) {
#ifdef USE_GPU
        auto MN = in.size(0) * in.size(1);
        auto out = torch::empty({in.size(0), in.size(1), in.size(2)}, in.options());
        auto K = in.size(2);
        fluke_rmsnorm_gpu(in.contiguous().data_ptr(), x.contiguous().data_ptr(),
                             norm_w.contiguous().data_ptr(), out.data_ptr(), MN, K, alpha, eps);
        x = out;
#else
        auto k = in + (x * alpha);
        auto rstd = torch::rsqrt(k.square().mean(-1, true).add_(eps));
        x = k.mul_(rstd).mul_(norm_w);
#endif
    };

    a = realtime();
    auto attn = tx_mha_forward(m, L, x);
    STAGE_SYNC(!x.device().is_cpu(), x.device().index());
    b = realtime();
    stats->time_self_attn += b - a;

    a = realtime();
    run_norm(L->norm1_w, attn);
    STAGE_SYNC(!x.device().is_cpu(), x.device().index());
    b = realtime();
    stats->time_norm1 += b - a;

    a = realtime();
    auto f = tx_gmlp_forward(L, x, stats);
    STAGE_SYNC(!x.device().is_cpu(), x.device().index());
    b = realtime();
    stats->time_ff += b - a;

    a = realtime();
    run_norm(L->norm2_w, f);
    STAGE_SYNC(!x.device().is_cpu(), x.device().index());
    b = realtime();
    stats->time_norm2 += b - a;
}

#ifdef USE_GPU
static void tx_encoder_forward_quant(tx_model_t *m, const tx_layer_t *L, tensor_quant_t &a) {
    tx_stats_t *stats = m->stats;
    const float alpha = L->deepnorm_alpha;
    const float eps = 1e-5f;
    const int dev = a.tensor.device().index();
    double t0, t1;

    t0 = realtime();
    auto attn = tx_mha_forward_quant(m, L, a).contiguous();
    STAGE_SYNC(true, dev);
    t1 = realtime(); stats->time_self_attn += t1 - t0;

    const int n_tokens = attn.size(0) * attn.size(1);
    const int K = attn.size(2);
    t0 = realtime();
    fluke_rmsnorm_quant_int8_gpu(attn.data_ptr(), L->norm1_w.contiguous().data_ptr(),
                                    a.tensor.data_ptr(), a.scale.data_ptr(), n_tokens, K, alpha, eps);
    STAGE_SYNC(true, dev);
    t1 = realtime(); stats->time_norm1 += t1 - t0;

    t0 = realtime();
    auto f = tx_gmlp_forward_quant(L, a, stats).contiguous();
    STAGE_SYNC(true, dev);
    t1 = realtime(); stats->time_ff += t1 - t0;

    t0 = realtime();
    fluke_rmsnorm_quant_int8_gpu(f.data_ptr(), L->norm2_w.contiguous().data_ptr(),
                                    a.tensor.data_ptr(), a.scale.data_ptr(), n_tokens, K, alpha, eps);
    STAGE_SYNC(true, dev);
    t1 = realtime(); stats->time_norm2 += t1 - t0;
}
#endif

at::Tensor tx_model_forward(tx_model_t *m, at::Tensor x) {
    tx_stats_t *stats = m->stats;
    const bool on_gpu = !x.device().is_cpu();
    const auto dev_idx = x.device().index();
    double a, b;

    a = realtime();
    at::Tensor h = conv_stack_forward(m->convs, x);
    STAGE_SYNC(on_gpu, dev_idx);
    b = realtime();
    stats->time_conv_stack += b - a;

    a = realtime();
#ifdef USE_GPU
    if (m->quant_stream && on_gpu && h.size(1) <= 2048) {
        tensor_quant_t aq = quantize_tensor(h, -1);
        for (auto &L : m->layers) tx_encoder_forward_quant(m, &L, aq);
        h = (aq.tensor.to(at::kFloat) * aq.scale.unsqueeze(-1)).to(at::kHalf);
    } else
#endif
    {
        for (auto &L : m->layers) tx_encoder_forward(m, &L, h);
    }
    STAGE_SYNC(on_gpu, dev_idx);
    b = realtime();
    stats->time_tx_encoder += b - a;

    // upsample (tx_decoder). Use matmul+bias (F::linear semantics, as torch::nn::Linear) rather than
    // at::linear, which flattens contiguous 3D to a fused addmm and differs by ~1 ULP.
    a = realtime();
    {
        const int64_t N = h.size(0), T = h.size(1), C = h.size(2);
        h = (h.matmul(m->up_w.t()) + m->up_b).reshape({N, m->scale_factor * T, C});
    }
    STAGE_SYNC(on_gpu, dev_idx);
    b = realtime();
    stats->time_tx_decoder += b - a;

    // CRF (weight pre-scaled at load; matmul to match torch::nn::Linear, bias-free)
    a = realtime();
    h = h.matmul(m->crf_w.t());
    STAGE_SYNC(on_gpu, dev_idx);
    b = realtime();
    stats->time_crf += b - a;

    return h;
}

void free_tx_model(tx_model_t *m) {
    delete m;
}

