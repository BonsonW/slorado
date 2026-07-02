#pragma once

#include "CRFModel.h"
#include "calib.h"
#include "error.h"
#include "quant.h"
#include "misc.h"
#include "tensor_chunk_utils.h"
#include "fluke/fluke.h"

#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <torch/nn.h>

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace torch::nn;

ModuleHolder<AnyModule> load_tx_model(const CRFModelConfig &model_config, const torch::TensorOptions &options, tx_stats_t *model_stats, bool use_flash, const std::string &quant_mode, int nthreads);

torch::Tensor scaled_dot_product_attention_naive(
    const torch::Tensor &q,
    const torch::Tensor &k,
    const torch::Tensor &v,
    const torch::Tensor &mask
);

struct RMSNormImpl : torch::nn::Module {
    RMSNormImpl(int hidden_size_);
    torch::Tensor forward(torch::Tensor x);

    torch::Tensor weight;
    const int hidden_size;
    const float eps{1e-5f};
};

TORCH_MODULE(RMSNorm);

struct GatedMLPImpl : torch::nn::Module {
    GatedMLPImpl(int in_features, int hidden_features,
                 tx_stats_t *stats = nullptr, const std::string &name_prefix = "");

    torch::Tensor forward(const torch::Tensor &x);
    // Fused int8 path: dual GEMM (gate,up) + SiLU on an int8 activation, then fc2 (fp16).
    torch::Tensor forward_quant(const tensor_quant_t &x);
    // Detect a kernel backend for (device, format) and eagerly quantize fc1's gate/up weights.
    void setup_backend(const fluke_dims_t &dims, int device_index, enum fluke_format_t format);
    void update_calib_weights();

    bool features_interleaved = false;
    int in_features;
    int hidden_features;
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    tx_stats_t *stats_ = nullptr;
    std::string prefix_;
    calib_layer_t *cl_fc1_ = nullptr, *cl_fc2_ = nullptr;

    fluke_backend_t *backend_ = nullptr; // shared, process-lifetime handle (not owned)
    tensor_quant_t qw_gate_, qw_up_; // eagerly-quantized int8 gate/up weights (+per-channel scale)
};

TORCH_MODULE(GatedMLP);

struct RotaryEmbeddingImpl : torch::nn::Module {
    RotaryEmbeddingImpl(
        int dim_,
        float theta_,
        int max_seq_len_,
        const torch::TensorOptions &options_,
        tx_stats_t *stats
    );

    torch::Tensor forward(torch::Tensor &qkv);
    void assert_forward_dims(const torch::Tensor &qkv) const;

    torch::Tensor cos_buf;
    torch::Tensor sin_buf;

    const int64_t dim, max_seq_len;
    const float theta;
    const torch::TensorOptions options;
    tx_stats_t *stats_ = nullptr;
};

TORCH_MODULE(RotaryEmbedding);

using MaskKey = std::pair<int64_t, torch::Device>;

// Hash function for std::pair<int64_t, int>
struct MaskKeyHash {
    std::size_t operator()(const MaskKey &key) const {
        auto hash1 = std::hash<int64_t>{}(key.first);
        auto hash2 = std::hash<torch::Device>{}(key.second);
        return hash1 ^ (hash2 << 1);
    }
};

struct MultiHeadAttentionImpl : torch::nn::Module {
    MultiHeadAttentionImpl(
        int d_model_,
        int nhead_,
        bool qkv_bias_,
        bool out_bias_,
        const std::pair<int, int> &attn_window_,
        const torch::TensorOptions &options_,
        tx_stats_t *_model_stats,
        int layer_idx = -1
    );

    torch::Tensor forward(torch::Tensor x);
    // Fused int8 path: fused wqkv GEMM + rotary on an int8 activation, then shared attn tail.
    torch::Tensor forward_quant(const tensor_quant_t &x);
    // Detect a kernel backend for (device, format) and eagerly quantize the wqkv weight.
    void setup_backend(const fluke_dims_t &dims, int device_index, enum fluke_format_t format);
    void update_calib_weights();

    torch::Tensor get_attn_window_mask(const int64_t size);
    torch::Tensor build_attn_window_mask(const int64_t size) const;

    const int d_model, nhead, head_dim, num_splits;
    const std::pair<int, int> attn_window;
    const torch::TensorOptions options;
    bool wqkv_transposed = false;

    std::unordered_map<MaskKey, torch::Tensor, MaskKeyHash> mask_cache{};

    torch::nn::Linear wqkv{nullptr}, out_proj{nullptr};
    RotaryEmbedding rotary_emb{nullptr};

    tx_stats_t *model_stats;

    std::string attn_prefix_;
    calib_layer_t *cl_wqkv_ = nullptr, *cl_out_proj_ = nullptr;

    fluke_backend_t *backend_ = nullptr; // shared, process-lifetime handle (not owned)
    tensor_quant_t qw_wqkv_;              // eagerly-quantized int8 wqkv weight (+per-channel scale)

private:
    // Shared fp16 attention core: SDPA / flash + out_proj. Both forward paths call this so the
    // bulky attention logic stays single-copy. qkv: fp16 [N,T,3,nhead,head_dim].
    torch::Tensor attn_tail(torch::Tensor qkv, const layer_quant_t &lq_op);
};

TORCH_MODULE(MultiHeadAttention);

struct TxEncoderImpl : torch::nn::Module {
    TxEncoderImpl(const TxEncoderParams &params, const torch::TensorOptions &options, tx_stats_t *model_stats, int layer_idx = -1);

    torch::Tensor forward(torch::Tensor x);
    // Fused int8 path: drives one layer of the int8 residual stream in-place on `a`.
    void forward_quant(tensor_quant_t &a);

    TxEncoderParams params;
    
    torch::Tensor sincos_bfr, proj_weight, proj_bias, t_res_weights, t_res2_weights, t_fc2_wts;

    MultiHeadAttention self_attn{nullptr};
    GatedMLP ff{nullptr};
    RMSNorm norm1{nullptr}, norm2{nullptr};

    tx_stats_t *model_stats;
};

TORCH_MODULE(TxEncoder);

struct TxEncoderStackImpl : torch::nn::Module {
    TxEncoderStackImpl(const TxEncoderParams &params, const torch::TensorOptions &options, tx_stats_t *model_stats);

    torch::Tensor forward(const torch::Tensor &x);

    torch::nn::Sequential stack{nullptr};
    std::vector<TxEncoder> layer_vec;

    // Set at load when the int8 kernel path is enabled AND available (flag on, arch/dims match).
    bool quant_stream_ = false;
};

TORCH_MODULE(TxEncoderStack);

struct LinearUpsampleImpl : torch::nn::Module {
    LinearUpsampleImpl(const EncoderUpsampleParams &params);

    torch::Tensor forward(const torch::Tensor &x);

    const int scale_factor;
    torch::nn::Linear linear{nullptr};
};

TORCH_MODULE(LinearUpsample);

struct LinearScaledCRFImpl : torch::nn::Module {
    LinearScaledCRFImpl(const CRFEncoderParams &params);

    torch::Tensor forward(const torch::Tensor &x);

    bool scale_applied = false;
    torch::nn::Linear linear{nullptr};
    CRFEncoderParams m_params;
};

TORCH_MODULE(LinearScaledCRF);

struct TxModelImpl : torch::nn::Module {
    explicit TxModelImpl(const CRFModelConfig &config, const torch::TensorOptions &options, tx_stats_t *_model_stats);

    void load_state_dict(const std::vector<torch::Tensor> &weights) {
        module_load_state_dict(*this, weights);
    }

    torch::Tensor forward(const torch::Tensor &chunk_NCT);

    ::ConvStack convs{nullptr};
    TxEncoderStack tx_encoder{nullptr};
    LinearUpsample tx_decoder{nullptr};
    LinearScaledCRF crf{nullptr};

    tx_stats_t *model_stats;

    const torch::TensorOptions m_options;
};

TORCH_MODULE(TxModel);
