#pragma once

#include "CRFModel.h"
#include "calib.h"
#include "error.h"
#include "misc.h"
#include "tensor_chunk_utils.h"

#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <torch/nn.h>

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace torch::nn;

ModuleHolder<AnyModule> load_tx_model(const CRFModelConfig &model_config, const torch::TensorOptions &options, tx_stats_t *model_stats, bool use_flash, int nthreads);

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

    bool features_interleaved = false;
    int in_features;
    int hidden_features;
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    calib_stats_t *calib_stats_ = nullptr;
    calib_layer_t *cl_fc1_ = nullptr, *cl_fc2_ = nullptr;
    std::string qm_fc1_, qm_fc2_;
    std::string qm_fc1_act_, qm_fc2_act_;
};

TORCH_MODULE(GatedMLP);

struct RotaryEmbeddingImpl : torch::nn::Module {
    RotaryEmbeddingImpl(
        int dim_,
        float theta_,
        int max_seq_len_,
        const torch::TensorOptions &options_,
        int nthreads_
    );

    torch::Tensor forward(torch::Tensor &qkv);
    void assert_forward_dims(const torch::Tensor &qkv) const;

    torch::Tensor cos_buf;
    torch::Tensor sin_buf;

    const int64_t dim, max_seq_len;
    const float theta;
    const torch::TensorOptions options;
    const int nthreads;
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
        bool use_flash_,
        int nthreads
    );

    torch::Tensor forward(torch::Tensor x);
    void set_calib(const std::string &name_prefix, calib_stats_t *calib);
    void set_quant_methods(const std::string &name_prefix,
                           const std::unordered_map<std::string, std::string> &cfg);

    torch::Tensor get_attn_window_mask(const int64_t size);
    torch::Tensor build_attn_window_mask(const int64_t size) const;

    bool use_flash;

    const int d_model, nhead, head_dim, num_splits;
    const std::pair<int, int> attn_window;
    const torch::TensorOptions options;
    bool wqkv_transposed = false;

    std::unordered_map<MaskKey, torch::Tensor, MaskKeyHash> mask_cache{};

    torch::nn::Linear wqkv{nullptr}, out_proj{nullptr};
    RotaryEmbedding rotary_emb{nullptr};

    tx_stats_t *model_stats;

    calib_stats_t *calib_stats_ = nullptr;
    calib_layer_t *cl_wqkv_ = nullptr, *cl_out_proj_ = nullptr;
    std::string qm_wqkv_, qm_out_proj_;
    std::string qm_wqkv_act_;
};

TORCH_MODULE(MultiHeadAttention);

struct TxEncoderImpl : torch::nn::Module {
    TxEncoderImpl(const TxEncoderParams &params, const torch::TensorOptions &options, tx_stats_t *model_stats, bool use_flash, int nthreads, int layer_idx = -1);

    torch::Tensor forward(torch::Tensor x);

    TxEncoderParams params;
    
    torch::Tensor sincos_bfr, proj_weight, proj_bias, t_res_weights, t_res2_weights, t_fc2_wts;

    MultiHeadAttention self_attn{nullptr};
    GatedMLP ff{nullptr};
    RMSNorm norm1{nullptr}, norm2{nullptr};

    tx_stats_t *model_stats;
    int device_idx;
};

TORCH_MODULE(TxEncoder);

struct TxEncoderStackImpl : torch::nn::Module {
    TxEncoderStackImpl(const TxEncoderParams &params, const torch::TensorOptions &options, tx_stats_t *model_stats, bool use_flash, int nthreads);

    torch::Tensor forward(const torch::Tensor &x);
    
    bool use_i8{false};
    torch::nn::Sequential stack{nullptr};
    std::vector<TxEncoder> layer_vec;
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
    explicit TxModelImpl(const CRFModelConfig &config, const torch::TensorOptions &options, tx_stats_t *_model_stats, bool use_flash, int nthreads);

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

    calib_stats_t *calib_stats_ = nullptr;
    calib_layer_t *cl_upsample_ = nullptr, *cl_crf_ = nullptr;
    std::string qm_upsample_, qm_crf_;
};

TORCH_MODULE(TxModel);
