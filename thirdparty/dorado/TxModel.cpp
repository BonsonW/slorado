#include "TxModel.h"
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

// Empty quant method => fp16 passthrough; used as the default for unset per-layer lookups.
static const layer_quant_t k_empty_lq;

void apply_rounding(torch::Tensor &t, int remove_bits) {
    // Round Float16 tensor elements such that the last `remove_bits` of the mantissa are 0s.
    // TODO: this is slightly dangerous as it will turn numbers close to +/-65304 into +/-inf
    t.view(torch::kI16).add_(1 << (remove_bits - 1));
    t.view(torch::kI16).bitwise_and_(0x10000 - (1 << remove_bits));
}

torch::Tensor scaled_dot_product_attention_naive(
    const torch::Tensor &q,
    const torch::Tensor &k,
    const torch::Tensor &v,
    const torch::Tensor &mask
) {
    auto matmul_qk = torch::matmul(q, k.transpose(-2, -1));

    auto d_k = k.size(-1);
    matmul_qk = matmul_qk / std::sqrt(d_k);

    if (mask.defined()) {
        matmul_qk = matmul_qk + (mask.logical_not() * -1e9);
    }

    auto weights = torch::softmax(matmul_qk, -1);
    return torch::matmul(weights, v);
}

RMSNormImpl::RMSNormImpl(int hidden_size_) : hidden_size(hidden_size_) {
    weight = torch::ones({hidden_size});
    register_parameter("weight", weight, false);
}

torch::Tensor RMSNormImpl::forward(torch::Tensor x) {
    torch::Tensor rstd = torch::rsqrt(x.square().mean(-1, true).add_(eps));
    x.mul_(rstd).mul_(weight);
    return x;
}

GatedMLPImpl::GatedMLPImpl(int in_features_, int hidden_features_,
                           tx_stats_t *stats, const std::string &name_prefix)
    : in_features(in_features_), hidden_features(hidden_features_) {
    fc1 = register_module("fc1", Linear(LinearOptions(in_features, 2 * hidden_features).bias(false)));
    fc2 = register_module("fc2", Linear(LinearOptions(hidden_features, in_features).bias(false)));

    if (!name_prefix.empty() && stats) {
        stats_ = stats;
        prefix_ = name_prefix;
        if (stats->calib_stats) {
            cl_fc1_ = stats->calib_stats->register_layer(name_prefix + ".fc1", fc1->weight);
            cl_fc2_ = stats->calib_stats->register_layer(name_prefix + ".fc2", fc2->weight);
        }
    }
};

torch::Tensor GatedMLPImpl::forward(const torch::Tensor &x) {
    auto lq = [&](const char *suffix) -> const layer_quant_t* {
        if (stats_ && !stats_->quant_methods.empty()) {
            auto it = stats_->quant_methods.find(prefix_ + suffix);
            if (it != stats_->quant_methods.end()) return &it->second;
        }
        return nullptr;
    };
    calib_stats_t *cs = stats_ ? stats_->calib_stats : nullptr;

    torch::Tensor t = qlinear({fc1->weight, fc1->bias, lq(".fc1"), cs, cl_fc1_}, x);
#ifdef USE_GPU
    auto M = t.size(0) * t.size(1);
    auto K = t.size(2) / 2;
    auto silu_o = torch::empty({t.size(0), t.size(1), K}, t.options());
    openfish_silu_mul_gpu(t.data_ptr(), silu_o.data_ptr(), M, K);
    t = silu_o;
#else
    const auto chunks = t.chunk(2, -1);
    const auto &y = chunks[0];
    const auto &gate = chunks[1];
    t = functional::silu(gate).mul_(y);
#endif
    return qlinear({fc2->weight, fc2->bias, lq(".fc2"), cs, cl_fc2_}, t);
}

void GatedMLPImpl::update_calib_weights() {
    if (!stats_ || !stats_->calib_stats) return;
    stats_->calib_stats->update_weight(cl_fc1_, fc1->weight);
    stats_->calib_stats->update_weight(cl_fc2_, fc2->weight);
}

// Fused int8 path: dual GEMM (gate,up) + SiLU on an int8 activation, then fc2 (fp16).
torch::Tensor GatedMLPImpl::forward_quant(const tensor_quant_t &x) {
    auto g = fluke_gated_mlp_i8(backend_, x, qw_gate_, qw_up_);
    const layer_quant_t *lq_fc2 = &k_empty_lq;
    if (stats_ && !stats_->quant_methods.empty()) {
        auto it = stats_->quant_methods.find(prefix_ + ".fc2");
        if (it != stats_->quant_methods.end()) lq_fc2 = &it->second;
    }
    return at::linear(fake_quant(g, lq_fc2->act), fake_quant(fc2->weight, lq_fc2->weight), fc2->bias);
}

void GatedMLPImpl::setup_backend(const fluke_dims_t &dims, int device_index, enum fluke_format_t format) {
    backend_ = fluke_select_backend(device_index, format, dims);
    if (!backend_) return;
    // fc1->weight is [2*hidden, in]. The fp16 path splits the OUTPUT via chunk(2,-1):
    // chunks[0]=y (up), chunks[1]=gate. So rows [0:H]=up, [H:2H]=gate.
    const int64_t H = hidden_features;
    auto up_w   = fc1->weight.slice(0, 0, H).contiguous();
    auto gate_w = fc1->weight.slice(0, H, 2 * H).contiguous();
    qw_up_   = quantize_tensor(up_w,   /*dim=*/1);
    qw_gate_ = quantize_tensor(gate_w, /*dim=*/1);
}

RotaryEmbeddingImpl::RotaryEmbeddingImpl(
    int dim_,
    float theta_,
    int max_seq_len_,
    const torch::TensorOptions &options_,
    tx_stats_t *stats
) :
    dim(dim_),
    max_seq_len(max_seq_len_),
    theta(theta_),
    options(options_)
{
    stats_ = stats;
    auto inv_freq = torch::pow(theta, torch::arange(0, dim, 2, options) / dim).reciprocal();
    torch::Tensor freqs = torch::arange(max_seq_len, options).outer(inv_freq);

    auto cos = torch::cos(freqs).to(torch::kFloat32).contiguous();
    auto sin = torch::sin(freqs).to(torch::kFloat32).contiguous();
    cos_buf = cos;
    sin_buf = sin;
};

torch::Tensor RotaryEmbeddingImpl::forward(torch::Tensor &qkv) {
    assert_forward_dims(qkv);
    const int batch_size = qkv.size(0);
    const int seqlen = qkv.size(1);
    const int nheads = qkv.size(3);
    const int head_dim = qkv.size(4);
    const int rotary_dim = 32;
    const int stride_batch = qkv.stride(0);
    const int stride_seq = qkv.stride(1);
    const int stride_head = qkv.stride(3);

    auto qkv_chunks = qkv.chunk(3, 2);

#ifdef USE_GPU
    if (!qkv.device().is_cpu()) {
        openfish_rotary_emb_gpu(
            qkv_chunks[0].data_ptr(),
            sin_buf.data_ptr(),
            cos_buf.data_ptr(),
            batch_size,
            seqlen,
            nheads,
            head_dim,
            rotary_dim,
            stride_batch,
            stride_seq,
            stride_head
        );
        
        openfish_rotary_emb_gpu(
            qkv_chunks[1].data_ptr(),
            sin_buf.data_ptr(),
            cos_buf.data_ptr(),
            batch_size,
            seqlen,
            nheads,
            head_dim,
            rotary_dim,
            stride_batch,
            stride_seq,
            stride_head
        );
    } else
#endif
    {
        openfish_rotary_emb_cpu(
            qkv_chunks[0].data_ptr(),
            sin_buf.data_ptr(),
            cos_buf.data_ptr(),
            batch_size,
            seqlen,
            nheads,
            head_dim,
            rotary_dim,
            stride_batch,
            stride_seq,
            stride_head,
            stats_->nthreads
        );

        openfish_rotary_emb_cpu(
            qkv_chunks[1].data_ptr(),
            sin_buf.data_ptr(),
            cos_buf.data_ptr(),
            batch_size,
            seqlen,
            nheads,
            head_dim,
            rotary_dim,
            stride_batch,
            stride_seq,
            stride_head,
            stats_->nthreads
        );
    }
    
    return qkv;
}

void RotaryEmbeddingImpl::assert_forward_dims(const torch::Tensor &qkv) const {
    // Expected shape: N, seq_len, 3, nhead, head_dim
    const int64_t seq_len = qkv.size(1);
    const int64_t three = qkv.size(2);
    const int64_t head_dim = qkv.size(4);

    bool has_error = false;
    if (seq_len > max_seq_len) {
        has_error = true;
        ERROR("RotE - maximum sequence length exceeded - len:%ld max:%ld - Your chunksize may be too large", seq_len, max_seq_len);
    }
    if (three != 3) {
        has_error = true;
        ERROR("RotE - expected constant size:3 at dim:2 found:%ld", three);
    }
    if (head_dim != dim) {
        has_error = true;
        ERROR("RotE - expected head_dim size:%ld at dim:4 found:%ld", dim, head_dim);
    }
    if (has_error) {
        exit(EXIT_FAILURE);
    }
}

MultiHeadAttentionImpl::MultiHeadAttentionImpl(
    int d_model_,
    int nhead_,
    bool qkv_bias_,
    bool out_bias_,
    const std::pair<int, int> &attn_window_,
    const torch::TensorOptions &options_,
    tx_stats_t *_model_stats,
    int layer_idx
) :
    d_model(d_model_),
    nhead(nhead_),
    head_dim(d_model_ / nhead_),
    // TODO: this may benefit from fine-tuning. 8 gives good performance at chunk size 12k
    num_splits(12),
    attn_window(attn_window_),
    options(options_)
{
    wqkv = register_module("wqkv", Linear(LinearOptions(d_model, 3 * d_model).bias(qkv_bias_)));
    out_proj = register_module("out_proj", Linear(LinearOptions(d_model, d_model).bias(out_bias_)));
    const float theta = 10000.0f;
    const int64_t max_seq_len = 2048;
    rotary_emb = register_module("rotary_emb", RotaryEmbedding(head_dim, theta, max_seq_len, options, _model_stats));
    model_stats = _model_stats;

    if (layer_idx >= 0 && _model_stats) {
        attn_prefix_ = "transformer_encoder." + std::to_string(layer_idx) + ".self_attn";
        if (_model_stats->calib_stats) {
            cl_wqkv_     = _model_stats->calib_stats->register_layer(attn_prefix_ + ".wqkv",     wqkv->weight);
            cl_out_proj_ = _model_stats->calib_stats->register_layer(attn_prefix_ + ".out_proj", out_proj->weight);
        }
    }
};


torch::Tensor MultiHeadAttentionImpl::get_attn_window_mask(const int64_t size) {
    const auto key = MaskKey{size, options.device()};
    if (mask_cache.find(key) == mask_cache.end()) {
        mask_cache[key] = build_attn_window_mask(size);
    }
    return mask_cache.at(key);
}

torch::Tensor MultiHeadAttentionImpl::build_attn_window_mask(const int64_t size) const {
    const auto win_upper = std::get<0>(attn_window);
    const auto win_lower = std::get<1>(attn_window);
    torch::Tensor mask = torch::ones({size, size}, options.device());
    mask.triu_(-win_upper).tril_(win_lower);
    mask = mask.to(torch::kBool);
    return mask;
};

torch::Tensor MultiHeadAttentionImpl::forward(torch::Tensor x) {
    const int64_t N = x.size(0);
    const int64_t T = x.size(1);

    double a, b;

    a = realtime();
    auto lq = [&](const char *suffix) -> const layer_quant_t& {
        if (model_stats && !attn_prefix_.empty() && !model_stats->quant_methods.empty()) {
            auto it = model_stats->quant_methods.find(attn_prefix_ + suffix);
            if (it != model_stats->quant_methods.end()) return it->second;
        }
        return k_empty_lq;
    };
    const auto &lq_wqkv = lq(".wqkv");
    const auto &lq_op   = lq(".out_proj");

    if (cl_wqkv_) model_stats->calib_stats->accumulate(cl_wqkv_, x);
    auto qkv = at::linear(fake_quant(x, lq_wqkv.act), fake_quant(wqkv->weight, lq_wqkv.weight), wqkv->bias)
                   .view({N, T, 3, nhead, head_dim});
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_mm += b-a;

    a = realtime();
    qkv = rotary_emb(qkv);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_rotary_emb += b-a;

    return attn_tail(qkv, lq_op);
}

// int8 path: fused int8 wqkv GEMM + rotary produces fp16 qkv, then the shared attention tail.
torch::Tensor MultiHeadAttentionImpl::forward_quant(const tensor_quant_t &x) {
    const layer_quant_t *lq_op = &k_empty_lq;
    if (model_stats && !attn_prefix_.empty() && !model_stats->quant_methods.empty()) {
        auto it = model_stats->quant_methods.find(attn_prefix_ + ".out_proj");
        if (it != model_stats->quant_methods.end()) lq_op = &it->second;
    }
    const bool on_gpu = !x.tensor.device().is_cpu();
    double a = realtime();
    auto qkv = fluke_qkv_rotary_i8(backend_, x, qw_wqkv_, rotary_emb->sin_buf, rotary_emb->cos_buf);
    if (on_gpu) torch::cuda::synchronize(x.tensor.device().index());
    model_stats->time_mm += realtime() - a; // fused wqkv GEMM + rotary
    return attn_tail(qkv, *lq_op);
}

// Shared fp16 attention core: SDPA / flash + out_proj. Both forward paths call this.
torch::Tensor MultiHeadAttentionImpl::attn_tail(torch::Tensor qkv, const layer_quant_t &lq_op) {
    const int64_t N = qkv.size(0);
    const int64_t T = qkv.size(1);
    const int64_t C = d_model;

    double a, b;
    const bool on_gpu = !qkv.device().is_cpu();

    a = realtime();
    const auto win_upper = std::get<0>(attn_window);
    const auto win_lower = std::get<1>(attn_window);

    torch::Tensor attn_output_ntc;
#if defined USE_GPU && ((TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 4) || TORCH_VERSION_MAJOR >= 3)
    if (model_stats->use_flash) {
        float softmax_scale = 1.0 / std::sqrt(head_dim);

        auto qkv_chunks = qkv.chunk(3, 2);
        auto q = qkv_chunks[0].squeeze(2);
        auto k = qkv_chunks[1].squeeze(2);
        auto v = qkv_chunks[2].squeeze(2);
        
        auto flash_res = at::_flash_attention_forward(
            q, k, v,
            std::nullopt, std::nullopt, // cum_seq_qk
            qkv.size(1), qkv.size(1), // size_qk
            0.0, // dropout
            false, // casual
            false, // return debug mask
            softmax_scale,
            win_lower,
            win_upper,
            std::nullopt, // seqused k
            std::nullopt // alibi slopes
        );
        attn_output_ntc = std::get<0>(flash_res).reshape({N, T, C});
    } else
#endif
    {
        qkv = qkv.permute({2, 0, 3, 1, 4}); // N T 3 H D -> 3 N H T D
        attn_output_ntc = torch::empty({N, T, C}, qkv.options());
        auto attn_window_mask = get_attn_window_mask(T);
        auto attn_output = attn_output_ntc.view({N, T, nhead, head_dim}).transpose(1, 2);
        // // The MPS backend refuses to work on a span of the mask that doesn't have an
        // // alignment of 4 elements, so pad the amount we process each loop to that.
        const auto elems_per_split = pad_to(div_round_up(T, int64_t{num_splits}), int64_t{4});
        for (int i = 0; i < num_splits; ++i) {
            const auto qb = i * elems_per_split;
            if (qb >= T) {
                break;
            }
            const auto qe = std::min(T, qb + elems_per_split);
            const auto kvb = std::max<int64_t>(0, qb - win_lower);
            const auto kve = std::min<int64_t>(T, qe + win_upper);
            const auto q = qkv[0].slice(-2, qb, qe);
            const auto k = qkv[1].slice(-2, kvb, kve);
            const auto v = qkv[2].slice(-2, kvb, kve);
            const auto mask = attn_window_mask.index({Slice(qb, qe), Slice(kvb, kve)});
            c10::optional<torch::Tensor> opt_mask;
            // Not using the mask gets us significantly better performance, at the cost of some
            // accuracy. Accuracy loss is minimised by larger num_splits.
            opt_mask = mask;
            attn_output.slice(-2, qb, qe) = torch::scaled_dot_product_attention(q, k, v, opt_mask);
        }
    }

    if (on_gpu) torch::cuda::synchronize(qkv.device().index());
    b = realtime();
    model_stats->time_sdp_attn += b-a;

    a = realtime();
    if (cl_out_proj_) model_stats->calib_stats->accumulate(cl_out_proj_, attn_output_ntc);
    auto out = at::linear(fake_quant(attn_output_ntc, lq_op.act), fake_quant(out_proj->weight, lq_op.weight), out_proj->bias);
    if (on_gpu) torch::cuda::synchronize(qkv.device().index());
    b = realtime();
    model_stats->time_out_proj += b-a;

    return out;
};

void MultiHeadAttentionImpl::setup_backend(const fluke_dims_t &dims, int device_index, enum fluke_format_t format) {
    backend_ = fluke_select_backend(device_index, format, dims);
    if (!backend_) return;
    // wqkv->weight is [3*d_model, d_model]; one int8 scale per output channel (dim 0).
    qw_wqkv_ = quantize_tensor(wqkv->weight, /*dim=*/1);
}

void MultiHeadAttentionImpl::update_calib_weights() {
    if (!model_stats || !model_stats->calib_stats) return;
    model_stats->calib_stats->update_weight(cl_wqkv_, wqkv->weight);
    model_stats->calib_stats->update_weight(cl_out_proj_, out_proj->weight);
}

TxEncoderImpl::TxEncoderImpl(const TxEncoderParams &params_, const torch::TensorOptions &options, tx_stats_t *_model_stats, int layer_idx) : params(params_) {
    self_attn = register_module("self_attn", MultiHeadAttention(params.d_model, params.nhead, false, true, params.attn_window, options, _model_stats, layer_idx));
    const std::string ff_prefix = layer_idx >= 0
        ? "transformer_encoder." + std::to_string(layer_idx) + ".ff"
        : "";
    ff = register_module("ff", GatedMLP(params.d_model, params.dim_feedforward, _model_stats, ff_prefix));
    norm1 = register_module("norm1", RMSNorm(params.d_model));
    norm2 = register_module("norm2", RMSNorm(params.d_model));
    model_stats = _model_stats;

    const torch::Tensor deepnorm_alpha = torch::tensor(params.deepnorm_alpha);
    register_buffer("deepnorm_alpha", deepnorm_alpha);
};

torch::Tensor TxEncoderImpl::forward(torch::Tensor x) {
    torch::Tensor attn, f;
    const auto deepnorm_alpha = named_buffers()["deepnorm_alpha"];
    double a, b;

    auto run_norm = [&](RMSNorm &norm, const torch::Tensor &in, at::Tensor &weight) {
#if defined USE_GPU
        auto MN = in.size(0) * in.size(1);
        auto output = torch::empty({in.size(0), in.size(1), in.size(2)}, in.options());
        auto K = in.size(2);
        auto eps = 1e-5f;
        
        openfish_rmsnorm_gpu(
            in.contiguous().data_ptr(),
            x.contiguous().data_ptr(),
            weight.contiguous().data_ptr(),
            output.contiguous().data_ptr(),
            MN,
            K,
            deepnorm_alpha.flatten()[0].item<float>(),
            eps
        );
        x = output;
#else
        auto k = in + (x * deepnorm_alpha);
        x = norm(k);
#endif
    };

    a = realtime();
    attn = self_attn(x);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_self_attn += b-a;

    a = realtime();
    run_norm(norm1, attn, norm1->weight);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_norm1 += b-a;

    a = realtime();
    f = ff(x);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_ff += b-a;

    a = realtime();
    run_norm(norm2, f, norm2->weight);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_norm2 += b-a;

    return x;
}

// Fused int8 path: one encoder layer of the int8 residual stream, updating `a` in place.
// Each fused sublayer consumes the int8 activation directly; the fused RMSNorm re-quantizes
// (sublayer_out + dequant(a)*alpha) back into `a` as int8 + per-token scale.
void TxEncoderImpl::forward_quant(tensor_quant_t &a) {
#ifdef USE_GPU
    const float alpha = named_buffers()["deepnorm_alpha"].flatten()[0].item<float>();
    const float eps = 1e-5f;
    const int dev = a.tensor.device().index();
    double t0, t1;

    // self_attn wraps the fused qkv (time_mm) + SDPA (time_sdp_attn) + out_proj (time_out_proj).
    t0 = realtime();
    auto attn = self_attn->forward_quant(a).contiguous();
    torch::cuda::synchronize(dev);
    t1 = realtime(); model_stats->time_self_attn += t1 - t0;

    const int n_tokens = attn.size(0) * attn.size(1);
    const int K = attn.size(2);
    t0 = realtime();
    openfish_rmsnorm_quant_int8_gpu(
        attn.data_ptr(), norm1->weight.contiguous().data_ptr(),
        a.tensor.data_ptr(), a.scale.data_ptr(), n_tokens, K, alpha, eps);
    torch::cuda::synchronize(dev);
    t1 = realtime(); model_stats->time_norm1 += t1 - t0;

    t0 = realtime();
    auto f = ff->forward_quant(a).contiguous();
    torch::cuda::synchronize(dev);
    t1 = realtime(); model_stats->time_ff += t1 - t0;

    t0 = realtime();
    openfish_rmsnorm_quant_int8_gpu(
        f.data_ptr(), norm2->weight.contiguous().data_ptr(),
        a.tensor.data_ptr(), a.scale.data_ptr(), n_tokens, K, alpha, eps);
    torch::cuda::synchronize(dev);
    t1 = realtime(); model_stats->time_norm2 += t1 - t0;
#else
    (void)a; // fused int8 path is GPU-only; never reached on CPU (quant_stream_ stays false)
#endif
}

TxEncoderStackImpl::TxEncoderStackImpl(const TxEncoderParams &params, const torch::TensorOptions &options, tx_stats_t *model_stats) {
    stack = Sequential();
    for (int i = 0; i < params.depth; ++i) {
        TxEncoder encoder(params, options, model_stats, i);
        stack->push_back(register_module("transformer_encoder" + std::to_string(i), encoder));
        layer_vec.push_back(encoder);
    }
};

torch::Tensor TxEncoderStackImpl::forward(const torch::Tensor &x) {
#ifdef USE_GPU
    // Fused int8 path: quantize once at entry, carry an int8 residual stream through every layer
    // (re-quantized by each RMSNorm), dequantize once at exit. The rotary kernel takes the runtime
    // seqlen, so any T works up to the baked sin/cos table extent (S2048 => T<=2048, which also
    // matches the RoPE table's max_seq_len); larger T falls back to fp16.
    if (quant_stream_ && !x.device().is_cpu() && x.size(1) <= 2048) {
        tensor_quant_t a = quantize_tensor(x, -1); // per-token int8
        for (auto &enc : layer_vec) enc->forward_quant(a);
        return (a.tensor.to(at::kFloat) * a.scale.unsqueeze(-1)).to(at::kHalf);
    }
#endif
    return stack->forward(x);
}

LinearUpsampleImpl::LinearUpsampleImpl(const EncoderUpsampleParams &params) : scale_factor(params.scale_factor) {
    linear = register_module("linear", Linear(LinearOptions(params.d_model, scale_factor * params.d_model).bias(true)));
};

torch::Tensor LinearUpsampleImpl::forward(const torch::Tensor &x) {
    const int64_t N = x.size(0);
    const int64_t T = x.size(1);
    const int64_t C = x.size(2);
    torch::Tensor out = linear(x).reshape({N, scale_factor * T, C});
    return out;
};

LinearScaledCRFImpl::LinearScaledCRFImpl(const CRFEncoderParams &params) {
    m_params = params;
    linear = register_module("linear", Linear(LinearOptions(m_params.insize, m_params.outsize()).bias(false)));
};

torch::Tensor LinearScaledCRFImpl::forward(const torch::Tensor &x) {
    if (!scale_applied) {
        linear->weight *= m_params.scale;
        scale_applied = true;
    }
    return linear(x);
}

TxModelImpl::TxModelImpl(const CRFModelConfig &config, const torch::TensorOptions &options, tx_stats_t *_model_stats) : m_options(options) {
    convs = register_module("convs", ::ConvStack(config.convs));
    tx_encoder = register_module("transformer_encoder", TxEncoderStack(config.tx->tx, m_options, _model_stats));
    tx_decoder = register_module("transformer_decoder", LinearUpsample(config.tx->upsample));
    crf = register_module("crf", LinearScaledCRF(config.tx->crf));
    model_stats = _model_stats;

}

torch::Tensor TxModelImpl::forward(const torch::Tensor &x) {
    torch::Tensor h;
    double a, b;

    a = realtime();
    h = convs->forward(x);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_conv_stack += b-a;
    
    a = realtime();
    h = tx_encoder(h);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_tx_encoder += b-a;

    a = realtime();
    h = tx_decoder(h);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_tx_decoder += b-a;

    a = realtime();
    h = crf(h);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    model_stats->time_crf += b-a;

    // Returns: NTC
    return h;
}

std::vector<torch::Tensor> load_tx_model_weights(const std::string &dir, int depth) {
    auto tensors = std::vector<std::string>{
            // convs 0-4
            "conv.0.conv.weight.tensor",
            "conv.0.conv.bias.tensor",
            "conv.1.conv.weight.tensor",
            "conv.1.conv.bias.tensor",
            "conv.2.conv.weight.tensor",
            "conv.2.conv.bias.tensor",
            "conv.3.conv.weight.tensor",
            "conv.3.conv.bias.tensor",
            "conv.4.conv.weight.tensor",
            "conv.4.conv.bias.tensor",
    };

    // tx encoder layers 0..depth-1 (v5 sup: depth 18, v6 rna sup: depth 22, ...)
    for (int i = 0; i < depth; ++i) {
        const std::string p = "transformer_encoder." + std::to_string(i) + ".";
        tensors.push_back(p + "self_attn.Wqkv.weight.tensor");
        tensors.push_back(p + "self_attn.out_proj.weight.tensor");
        tensors.push_back(p + "self_attn.out_proj.bias.tensor");
        tensors.push_back(p + "ff.fc1.weight.tensor");
        tensors.push_back(p + "ff.fc2.weight.tensor");
        tensors.push_back(p + "norm1.weight.tensor");
        tensors.push_back(p + "norm2.weight.tensor");
    }

    // tx decoder
    tensors.push_back("upsample.linear.weight.tensor");
    tensors.push_back("upsample.linear.bias.tensor");

    // linear CRF
    tensors.push_back("crf.linear.weight.tensor");

    return load_tensors(dir, tensors);
}

ModuleHolder<AnyModule> load_tx_model(const CRFModelConfig &model_config, const torch::TensorOptions &options, tx_stats_t *model_stats, bool use_flash, const std::string &quant_mode, int nthreads) {
    if (model_stats) {
        model_stats->use_flash = use_flash;
        model_stats->nthreads = nthreads;
        if (model_stats->quant_config) {
            build_quant_methods(model_stats->quant_methods, *model_stats->quant_config);
        }
    }
    auto model = TxModel(model_config, options, model_stats);
    auto state_dict = load_tx_model_weights(model_config.model_path, model_config.tx->tx.depth);
    model->load_state_dict(state_dict);
    model->to(options.dtype().toScalarType());
    model->to(options.device());
    model->eval();

    // Update calib weight stats now that real weights are loaded.
    // (register_layer is called during construction with initial/empty weight tensors.)
    if (model_stats && model_stats->calib_stats && model->tx_encoder) {
        for (auto &enc : model->tx_encoder->layer_vec) {
            enc->self_attn->update_calib_weights();
            enc->ff->update_calib_weights();
        }
    }

    // Quantized inference path: parse the requested mode to a kernel format, then eagerly quantize
    // weights + detect a device backend for (arch, format). Engages only when a backend is available
    // for every layer; otherwise the model falls back to the fp16 path transparently.
    enum fluke_format_t quant_format = fluke_parse_format(quant_mode);
#ifdef USE_GPU
    if (quant_format != FLUKE_FORMAT_NONE && model->tx_encoder && !options.device().is_cpu()) {
        const auto &txp = model_config.tx->tx;
        fluke_dims_t dims{txp.d_model, txp.dim_feedforward, txp.nhead,
                        txp.d_model / txp.nhead, /*max_seq=*/1024};
        const int dev = options.device().index();
        bool all_ok = true;
        for (auto &enc : model->tx_encoder->layer_vec) {
            enc->self_attn->setup_backend(dims, dev, quant_format);
            enc->ff->setup_backend(dims, dev, quant_format);
            if (!enc->self_attn->backend_ || !enc->ff->backend_) all_ok = false;
        }
        model->tx_encoder->quant_stream_ = all_ok;
        INFO("quant '%s' kernel path %s", quant_mode.c_str(), all_ok ? "enabled" : "unavailable (using fp16)");
    } else if (!quant_mode.empty() && quant_format == FLUKE_FORMAT_NONE) {
        WARNING("unknown quant mode '%s' — using fp16", quant_mode.c_str());
    }
#else
    if (quant_format != FLUKE_FORMAT_NONE)
        WARNING("quant mode '%s' requires a GPU build — using fp16", quant_mode.c_str());
#endif

    if (use_flash) {
        INFO("%s", "flash attention enabled");
    } else {
        INFO("%s", "flash attention disabled");
    }

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);
    return holder;
}

// =============================== procedural transformer model ===================================

tx_model_t *load_tx_model_proc(const CRFModelConfig &config, const torch::TensorOptions &options, tx_stats_t *model_stats, bool use_flash, const std::string &quant_mode, int nthreads) {
    if (model_stats) {
        model_stats->use_flash = use_flash;
        model_stats->nthreads = nthreads;
        if (model_stats->quant_config) build_quant_methods(model_stats->quant_methods, *model_stats->quant_config);
    }

    tx_model_t *m = new tx_model_t();
    m->stats = model_stats;
    const auto &txp = config.tx->tx;
    const int depth = txp.depth;
    const int d_model = txp.d_model, nhead = txp.nhead, head_dim = d_model / nhead;

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
        m->scale_factor = config.tx->upsample.scale_factor;
        m->crf_w = to_dev(t[2]);
        m->crf_w = m->crf_w * config.tx->crf.scale;
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
        openfish_rotary_emb_gpu(ch[0].data_ptr(), L->rot_sin.data_ptr(), L->rot_cos.data_ptr(), batch, seqlen, nheads, head_dim, rotary_dim, sb, ss, sh);
        openfish_rotary_emb_gpu(ch[1].data_ptr(), L->rot_sin.data_ptr(), L->rot_cos.data_ptr(), batch, seqlen, nheads, head_dim, rotary_dim, sb, ss, sh);
    } else
#endif
    {
        openfish_rotary_emb_cpu(ch[0].data_ptr(), L->rot_sin.data_ptr(), L->rot_cos.data_ptr(), batch, seqlen, nheads, head_dim, rotary_dim, sb, ss, sh, stats->nthreads);
        openfish_rotary_emb_cpu(ch[1].data_ptr(), L->rot_sin.data_ptr(), L->rot_cos.data_ptr(), batch, seqlen, nheads, head_dim, rotary_dim, sb, ss, sh, stats->nthreads);
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
    if (on_gpu) torch::cuda::synchronize(qkv.device().index());
    b = realtime();
    stats->time_sdp_attn += b - a;

    a = realtime();
    auto out = qlinear({L->out_proj_w, L->out_proj_b, lq_op, stats->calib_stats, L->cl_out_proj}, attn_output_ntc);
    if (on_gpu) torch::cuda::synchronize(qkv.device().index());
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
    if (on_gpu) torch::cuda::synchronize(x.device().index());
    b = realtime();
    stats->time_mm += b - a;

    a = realtime();
    qkv = tx_rotary(L, qkv, stats);
    if (on_gpu) torch::cuda::synchronize(x.device().index());
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
    if (on_gpu) torch::cuda::synchronize(x.tensor.device().index());
    stats->time_mm += realtime() - a;
    return tx_attn_tail(m, L, qkv, lq_op);
}

static at::Tensor tx_gmlp_forward(const tx_layer_t *L, torch::Tensor x, tx_stats_t *stats) {
    const layer_quant_t *lq_fc1 = lq_lookup(stats, L->ff_prefix, ".fc1");
    const layer_quant_t *lq_fc2 = lq_lookup(stats, L->ff_prefix, ".fc2");
    calib_stats_t *cs = stats ? stats->calib_stats : nullptr;

    torch::Tensor t = qlinear({L->fc1_w, at::Tensor(), lq_fc1, cs, L->cl_fc1}, x);
#ifdef USE_GPU
    auto M = t.size(0) * t.size(1);
    auto K = t.size(2) / 2;
    auto silu_o = torch::empty({t.size(0), t.size(1), K}, t.options());
    openfish_silu_mul_gpu(t.data_ptr(), silu_o.data_ptr(), M, K);
    t = silu_o;
#else
    const auto chunks = t.chunk(2, -1);
    t = functional::silu(chunks[1]).mul_(chunks[0]);
#endif
    return qlinear({L->fc2_w, at::Tensor(), lq_fc2, cs, L->cl_fc2}, t);
}

static at::Tensor tx_gmlp_forward_quant(const tx_layer_t *L, const tensor_quant_t &x, tx_stats_t *stats) {
    auto g = fluke_gated_mlp_i8(L->ff_backend, x, L->qw_gate, L->qw_up);
    const layer_quant_t *lq_fc2 = lq_lookup(stats, L->ff_prefix, ".fc2");
    return qlinear({L->fc2_w, at::Tensor(), lq_fc2, nullptr, nullptr}, g);
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
        openfish_rmsnorm_gpu(in.contiguous().data_ptr(), x.contiguous().data_ptr(),
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
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    stats->time_self_attn += b - a;

    a = realtime();
    run_norm(L->norm1_w, attn);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    stats->time_norm1 += b - a;

    a = realtime();
    auto f = tx_gmlp_forward(L, x, stats);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
    b = realtime();
    stats->time_ff += b - a;

    a = realtime();
    run_norm(L->norm2_w, f);
    if (!x.device().is_cpu()) torch::cuda::synchronize(x.device().index());
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
    torch::cuda::synchronize(dev);
    t1 = realtime(); stats->time_self_attn += t1 - t0;

    const int n_tokens = attn.size(0) * attn.size(1);
    const int K = attn.size(2);
    t0 = realtime();
    openfish_rmsnorm_quant_int8_gpu(attn.data_ptr(), L->norm1_w.contiguous().data_ptr(),
                                    a.tensor.data_ptr(), a.scale.data_ptr(), n_tokens, K, alpha, eps);
    torch::cuda::synchronize(dev);
    t1 = realtime(); stats->time_norm1 += t1 - t0;

    t0 = realtime();
    auto f = tx_gmlp_forward_quant(L, a, stats).contiguous();
    torch::cuda::synchronize(dev);
    t1 = realtime(); stats->time_ff += t1 - t0;

    t0 = realtime();
    openfish_rmsnorm_quant_int8_gpu(f.data_ptr(), L->norm2_w.contiguous().data_ptr(),
                                    a.tensor.data_ptr(), a.scale.data_ptr(), n_tokens, K, alpha, eps);
    torch::cuda::synchronize(dev);
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
    if (on_gpu) torch::cuda::synchronize(dev_idx);
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
    if (on_gpu) torch::cuda::synchronize(dev_idx);
    b = realtime();
    stats->time_tx_encoder += b - a;

    // upsample (tx_decoder). Use matmul+bias (F::linear semantics, as torch::nn::Linear) rather than
    // at::linear, which flattens contiguous 3D to a fused addmm and differs by ~1 ULP.
    a = realtime();
    {
        const int64_t N = h.size(0), T = h.size(1), C = h.size(2);
        h = (h.matmul(m->up_w.t()) + m->up_b).reshape({N, m->scale_factor * T, C});
    }
    if (on_gpu) torch::cuda::synchronize(dev_idx);
    b = realtime();
    stats->time_tx_decoder += b - a;

    // CRF (weight pre-scaled at load; matmul to match torch::nn::Linear, bias-free)
    a = realtime();
    h = h.matmul(m->crf_w.t());
    if (on_gpu) torch::cuda::synchronize(dev_idx);
    b = realtime();
    stats->time_crf += b - a;

    return h;
}

void free_tx_model(tx_model_t *m) {
    delete m;
}

