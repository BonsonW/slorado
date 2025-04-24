#include "TxModel.h"

#include <ATen/Functions.h>
#include <ATen/TensorIndexing.h>
#include <c10/core/ScalarType.h>
#include <torch/nn/functional/padding.h>
#include <torch/nn/options/padding.h>
#include <torch/types.h>
#include <torch/version.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <ATen/ops/scaled_dot_product_attention.h>

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <string>

#include <openfish/openfish.h>

using namespace torch::nn;
using Slice = torch::indexing::Slice;

std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

// torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
//     // TODO: determine Bc, Br dynamically
//     const int Bc = 32; const int Br = 32;

//     const int B = Q.size(0); const int nh = Q.size(1);
//     const int N = Q.size(2); const int d = Q.size(3);

//     const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
//     const float softmax_scale = 1.0 / sqrt(d);

//     // Initialize O, l, m to HBM
//     auto O = torch::zeros_like(Q);
//     auto l = torch::zeros({B, nh, N});
//     auto m = torch::full({B, nh, N}, -INFINITY);
//     torch::Device device(torch::kCUDA);
//     l = l.to(device); m = m.to(device);

//     // Calculate SRAM size needed per block
//     const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
//     int max_sram_size;
//     cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
//     printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

//     dim3 grid_dim(B, nh);  // batch_size x num_heads
//     dim3 block_dim(Bc);  // Bc threads per block

//     forward_kernel<<<grid_dim, block_dim, sram_size>>>(
//         Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
//         N, d, Tc, Tr, Bc, Br, softmax_scale,
//         l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
//     );
    
//     return O;
// }

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

GatedMLPImpl::GatedMLPImpl(int in_features_, int hidden_features_) : in_features(in_features_), hidden_features(hidden_features_) {
    fc1 = register_module("fc1", Linear(LinearOptions(in_features, 2 * hidden_features).bias(false)));
    fc2 = register_module("fc2", Linear(LinearOptions(hidden_features, in_features).bias(false)));
};

torch::Tensor GatedMLPImpl::forward(const torch::Tensor &x) {
    torch::Tensor t;
    t = fc1(x);
    const auto chunks = t.chunk(2, -1);
    const auto &y = chunks[0];
    const auto &gate = chunks[1];
    t = functional::silu(gate).mul_(y);
    
    return fc2(t);
}

RotaryEmbeddingImpl::RotaryEmbeddingImpl(
    int dim_,
    float theta_,
    int max_seq_len_,
    const torch::TensorOptions &options_
) :
    dim(dim_),
    max_seq_len(max_seq_len_),
    theta(theta_),
    options(options_)
{
    auto inv_freq = torch::pow(theta, torch::arange(0, dim, 2, options) / dim).reciprocal();
    torch::Tensor freqs = torch::arange(max_seq_len, options).outer(inv_freq);
    // const torch::Tensor freqs = torch::arange(max_seq_len, options).reshape({max_seq_len, 1, 1, 1}) * inv_freq;

    // std::vector<char> f;
    // size_t numel;
    // torch::IValue ival;
    // torch::Tensor pickle;

    // f = get_the_bytes("../bonito/sin_cached.pt");
    // ival = torch::pickle_load(f);
    // auto sin = ival.toTensor().to("cuda:0");

    // f = get_the_bytes("../bonito/cos_cached.pt");
    // ival = torch::pickle_load(f);
    // auto cos = ival.toTensor().to("cuda:0");

    auto cos = torch::cos(freqs).to(torch::kFloat32).contiguous();
    auto sin = torch::sin(freqs).to(torch::kFloat32).contiguous();
    cos_buf = cos;
    sin_buf = sin;
    // register_buffer("cos_freqs", cos);
    // register_buffer("sin_freqs", sin);
    
    // sin
    // cos = cos.index({torch::indexing::Slice(0, 833), torch::indexing::Ellipsis});
    // fprintf(stderr, "cos: %zd %zd | %zd\n", cos.size(0), cos.size(1), cos.dim());
    // pickle = cos.contiguous().to(torch::kFloat).cpu();
    // numel = pickle.numel();
    // fp = fopen("cos_2.blob", "w");
    // F_CHK(fp, "cos_2.blob");
    // if (fwrite(pickle.data_ptr(), sizeof(float), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);

    // exit(0);
};

torch::Tensor RotaryEmbeddingImpl::forward(torch::Tensor &qkv) {
    // Input is NT3HD
    assert_forward_dims(qkv);
    const int batch_size = qkv.size(0);
    const int seqlen = qkv.size(1);
    const int nheads = qkv.size(3);
    const int head_dim = qkv.size(4);
    const int rotary_dim = 32;
    const int stride_batch = qkv.stride(0);
    const int stride_seq = qkv.stride(1);
    const int stride_head = qkv.stride(3);

    auto qkv_arr = qkv.chunk(3, 2);
    
    openfish_rotary_f16(
        qkv_arr[0].data_ptr(),
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
    
    openfish_rotary_f16(
        qkv_arr[1].data_ptr(),
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
    return qkv;
    
    // // Input is NT3HD
    // assert_forward_dims(qkv);
    // const int64_t N = qkv.size(0);
    // const int64_t T = qkv.size(1);
    // const int64_t H = qkv.size(3);
    // const int64_t D = qkv.size(4);

    // auto buffers = named_buffers();
    // const torch::Tensor cos_buf = buffers["cos_freqs"].narrow(0, 0, T);
    // const torch::Tensor sin_buf = buffers["sin_freqs"].narrow(0, 0, T);

    // auto qk_evens = qkv.slice(2, 0, 2).slice(4, 0, D / 2);
    // auto qk_odds = qkv.slice(2, 0, 2).slice(4, D / 2, D);

    // // Allocate output tensor with memory layout as consumed by attention: 3NHTD
    // auto output = torch::empty({3, N, H, T, D}, qkv.options());
    // // View as [3 (q|k|v), N, H, T, 2 (even|odd), D/2], narrow first dim to 2 (q|k), then
    // // permute as [2 (even|odd), N, T, 2 (q|k), H, D/2] which is compatible with assignment below
    // auto output_kv_even_odd = output.view({3, N, H, T, 2, D / 2}).slice(0, 0, 2).permute({4, 1, 3, 0, 2, 5});

    // // Apply rotary embedding to Q and K
    // output_kv_even_odd[0] = cos_buf * qk_evens - sin_buf * qk_odds;
    // output_kv_even_odd[1] = sin_buf * qk_evens + cos_buf * qk_odds;

    // // Copy V to output
    // output.select(0, 2).permute({0, 2, 1, 3}) = qkv.select(2, 2);

    // return output;
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
    tx_stats_t *_model_stats
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
    rotary_emb = register_module("rotary_emb", RotaryEmbedding(head_dim, theta, max_seq_len, options));
    model_stats = _model_stats;
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
    auto device_idx = options.device_index();

    c10::cuda::CUDAGuard device_guard(device_idx);

    FILE *fp;
    size_t numel;

    const int64_t N = x.size(0);
    const int64_t T = x.size(1);
    const int64_t C = x.size(2);

    torch::Tensor qkv;
    torch::Tensor attn_output_ntc;

    double a, b;
    
    // print tens
    // fprintf(stderr, "ntc: %zd %zd %zd | %zd\n", x.size(0), x.size(1), x.size(2), x.dim());
    // numel = x.numel();
    // fp = fopen("ntc.blob", "w");
    // F_CHK(fp, "ntc.blob");
    // if (fwrite(x.to("cpu").data_ptr(), sizeof(int16_t), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);

    std::vector<char> f;
    torch::IValue ival;
    torch::Tensor pickle;

    // q_ro
    // f = get_the_bytes("../bonito/q_ro.pt");
    // ival = torch::pickle_load(f);
    // pickle = ival.toTensor().to(torch::kFloat).contiguous().cpu();
    // numel = pickle.numel();
    // fp = fopen("q_ro_full.blob", "w");
    // F_CHK(fp, "q_ro_full.blob");
    // if (fwrite(pickle.data_ptr(), sizeof(float), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);
    // exit(0);
    
    a = realtime();
    auto _wqkv = wqkv(x);
    torch::cuda::synchronize(device_idx);
    b = realtime();
    model_stats->time_mm += b-a;

    // fprintf(stderr, "wqkv: %zd %zd %zd | %zd\n", _wqkv.size(0), _wqkv.size(1), _wqkv.size(2), _wqkv.dim());
    // exit(0);
    // numel = _wqkv.numel();
    // fp = fopen("wqkv.blob", "w");
    // F_CHK(fp, "wqkv.blob");
    // if (fwrite(_wqkv.to("cpu").data_ptr(), sizeof(int16_t), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);

    

    // qkv
    // f = get_the_bytes("../bonito/qkv.pt");
    // ival = torch::pickle_load(f);
    // pickle = ival.toTensor().to(torch::kFloat).contiguous().cpu();
    // numel = pickle.numel();
    // fp = fopen("qkv_full.blob", "w");
    // F_CHK(fp, "qkv_full.blob");
    // if (fwrite(pickle.data_ptr(), sizeof(float), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);
    // exit(0);

    // // qkv_out
    // f = get_the_bytes("../bonito/qkv_out.pt");
    // ival = torch::pickle_load(f);
    // pickle = ival.toTensor().cpu().to(torch::kFloat).contiguous();
    // numel = pickle.numel();
    // fp = fopen("qkv_out.blob", "w");
    // F_CHK(fp, "qkv_out.blob");
    // if (fwrite(pickle.data_ptr(), sizeof(float), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);

    // exit(0);

    // sin
    // f = get_the_bytes("../bonito/sin_cached.pt");
    // ival = torch::pickle_load(f);
    // pickle = ival.toTensor().cpu().to(torch::kFloat).contiguous();
    // numel = pickle.numel();
    // fp = fopen("sin.blob", "w");
    // F_CHK(fp, "sin.blob");
    // if (fwrite(pickle.data_ptr(), sizeof(float), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);

    // f = get_the_bytes("../bonito/bonito_qkv.pt");
    // ival = torch::pickle_load(f);
    // pickle = ival.toTensor().to("cuda:0").to(torch::kHalf).contiguous();
    // fprintf(stderr, "pickled: %zd %zd %zd %zd %zd | %zd\n", pickle.size(0), pickle.size(1), pickle.size(2), pickle.size(3), pickle.size(4), pickle.dim());
    // qkv = pickle;
    
    // numel = pickle.numel();
    // fp = fopen("qkv_full.blob", "w");
    // F_CHK(fp, "qkv_full.blob");
    // if (fwrite(pickle.cpu().to(torch::kFloat).contiguous().data_ptr(), sizeof(float), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);

    // exit(0);

    // auto qk = pickle.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 2)}).reshape({N, T, -1, 64});
    // fprintf(stderr, "qk: %zd %zd %zd %zd | %zd\n", qk.size(0), qk.size(1), qk.size(2), qk.size(3),  qk.dim());
    // auto pickled = torch::pickle_save(qk);
    // std::ofstream fout("qk.pt", std::ios::out | std::ios::binary);
    // fout.write(pickled.data(), pickled.size());
    // fout.close();
    // exit(0);

    // numel = qk.numel();
    // fp = fopen("qk.blob", "w");
    // F_CHK(fp, "qk.blob");
    // if (fwrite(qk.to("cpu").to(torch::kFloat).data_ptr(), sizeof(float), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);
    // exit(0);

    // print tens
    // fprintf(stderr, "qkv: %zd %zd %zd %zd %zd | %zd\n", qkv.size(0), qkv.size(1), qkv.size(2), qkv.size(3), qkv.size(4), qkv.dim());
    // numel = qkv.numel();
    // fp = fopen("qkv.blob", "w");
    // F_CHK(fp, "qkv.blob");
    // if (fwrite(qkv.to("cpu").data_ptr(), sizeof(int16_t), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);

    a = realtime();
    qkv = _wqkv.view({N, T, 3, nhead, head_dim});
    qkv = rotary_emb(qkv);
    // qkv = qkv.permute({1, 3, 0, 2, 4}).contiguous();
    // const int batch_size = qkv.size(0);
    // const int seqlen = qkv.size(1);
    // const int nheads = qkv.size(3);
    // const int head_dim = qkv.size(4);
    // const int rotary_dim = 32;
    // const int stride_rotary = 32;
    // const int stride_batch = qkv.stride(0);
    // const int stride_seq = qkv.stride(1);
    // const int stride_c = qkv.stride(2);
    // const int stride_head = qkv.stride(3);
    // const int stride_head_dim = qkv.stride(4);

    // openfish_rotary(
    //     qkv.data_ptr(),
    //     qkv.data_ptr(),
    //     sin.data_ptr(),
    //     cos.data_ptr(),
    //     batch_size,
    //     seqlen,
    //     nheads,
    //     head_dim,
    //     rotary_dim,
    //     stride_batch,
    //     stride_seq,
    //     stride_c,
    //     stride_head,
    //     stride_head_dim,
    //     stride_rotary
    // );
    torch::cuda::synchronize(device_idx);
    b = realtime();
    model_stats->time_rotary_emb += b-a;

    // print tens
    fprintf(stderr, "rotary_emb_qkv: %zd %zd %zd %zd %zd | %zd\n", qkv.size(0), qkv.size(1), qkv.size(2), qkv.size(3), qkv.size(4), qkv.dim());
    // exit(0);
    a = realtime();
    attn_output_ntc = torch::empty({N, T, C}, x.options()).contiguous();
    auto attn_output = attn_output_ntc.view({N, T, nhead, head_dim});
    const auto win_upper = std::get<0>(attn_window);
    const auto win_lower = std::get<1>(attn_window);

    // fprintf(stderr, "qkv: %zd %zd %zd %zd %zd | %zd\n", qkv.size(0), qkv.size(1), qkv.size(2), qkv.size(3), qkv.size(4), qkv.dim());
    // numel = qkv.numel();
    // fp = fopen("x.blob", "w");
    // F_CHK(fp, "x.blob");
    // if (fwrite(qkv.to("cpu").to(torch::kFloat32).contiguous().data_ptr(), sizeof(float), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);

    // numel = qkv.numel();
    // fp = fopen("qkv_slorado.blob", "w");
    // F_CHK(fp, "qkv_slorado.blob");
    // if (fwrite(qkv.to("cpu").to(torch::kFloat32).contiguous().data_ptr(), sizeof(float), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);

    // exit(0);
    
    // bonito qkv
    // f = get_the_bytes("../bonito/qkv.pt");
    // ival = torch::pickle_load(f);
    // pickle = ival.toTensor().cpu().to(torch::kFloat).contiguous();
    // numel = pickle.numel();
    // fp = fopen("bonito_qkv.blob", "w");
    // F_CHK(fp, "bonito_qkv.blob");
    // if (fwrite(pickle.data_ptr(), sizeof(float), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);

    // exit(0);
    
    openfish_flash_fwd(
        qkv.data_ptr(),
        attn_output.data_ptr(),
        attn_output.size(0),
        attn_output.size(1),
        attn_output.size(2),
        attn_output.size(3),
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(3),
        win_upper,
        win_lower
    );
    torch::cuda::synchronize(device_idx);
    b = realtime();
    model_stats->time_sdp_attn += b-a;

    // qkv
    // f = get_the_bytes("../bonito/attn_output.pt");
    // ival = torch::pickle_load(f);
    // pickle = ival.toTensor();
    // numel = pickle.numel();
    // fp = fopen("attn_ntc_bonito.blob", "w");
    // F_CHK(fp, "attn_ntc_bonito.blob");
    // fprintf(stderr, "attn_ntc: %zd %zd %zd | %zd\n", attn_output_ntc.size(0), attn_output_ntc.size(1), attn_output_ntc.size(2), attn_output_ntc.dim());
    // if (fwrite(pickle.to(torch::kFloat32).contiguous().cpu().data_ptr(), sizeof(float), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);
    // exit(0);

    // print tens
    // fprintf(stderr, "attn_ntc: %zd %zd %zd | %zd\n", attn_output_ntc.size(0), attn_output_ntc.size(1), attn_output_ntc.size(2), attn_output_ntc.dim());
    // numel = attn_output_ntc.numel();
    // fp = fopen("slorado_attn_ntc.blob", "w");
    // F_CHK(fp, "slorado_attn_ntc.blob");
    // if (fwrite(attn_output_ntc.to(torch::kFloat32).contiguous().to("cpu").data_ptr(), sizeof(float), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);
    // exit(0);

    // fprintf(stderr, "attn_ntc: %zd %zd %zd | %zd\n", attn_output_ntc.size(0), attn_output_ntc.size(1), attn_output_ntc.size(2), attn_output_ntc.dim());
    // numel = attn_output_ntc.numel();
    // fp = fopen("openfish_attn_ntc.blob", "w");
    // F_CHK(fp, "openfish_attn_ntc.blob");
    // if (fwrite(attn_output_ntc.to(torch::kFloat32).contiguous().to("cpu").data_ptr(), sizeof(float), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);
    // exit(0);

    // convert and save the bonito attn_output
    // f = get_the_bytes("../bonito/bonito_attn_output.pt");
    // ival = torch::pickle_load(f);
    // pickle = ival.toTensor();
    // fprintf(stderr, "pickled: %zd %zd %zd %zd | %zd\n", pickle.size(0), pickle.size(1), pickle.size(2), pickle.size(3), pickle.dim());
    // numel = pickle.numel();
    // fp = fopen("bonito_attn_ntc.blob", "w");
    // F_CHK(fp, "bonito_attn_ntc.blob");
    // if (fwrite(pickle.data_ptr(), sizeof(float), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);

    // exit(0);

    a = realtime();
    x = out_proj(attn_output_ntc);
    torch::cuda::synchronize(device_idx);
    b = realtime();
    model_stats->time_out_proj += b-a;

    // print tens
    // fprintf(stderr, "out_proj_ntc: %zd %zd %zd | %zd\n", x.size(0), x.size(1), x.size(2), x.dim());
    // numel = x.numel();
    // fp = fopen("out_proj_ntc.blob", "w");
    // F_CHK(fp, "out_proj_ntc.blob");
    // if (fwrite(x.to("cpu").data_ptr(), sizeof(int16_t), numel, fp) != numel) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);

    // // exit
    // exit(0);
    
    return x;
};

TxEncoderImpl::TxEncoderImpl(const TxEncoderParams &params_, const torch::TensorOptions &options, tx_stats_t *_model_stats) : params(params_) {
    self_attn = register_module("self_attn", MultiHeadAttention(params.d_model, params.nhead, false, true, params.attn_window, options, _model_stats));
    ff = register_module("ff", GatedMLP(params.d_model, params.dim_feedforward));
    norm1 = register_module("norm1", RMSNorm(params.d_model));
    norm2 = register_module("norm2", RMSNorm(params.d_model));
    device_idx = options.device_index();
    model_stats = _model_stats;

    const torch::Tensor deepnorm_alpha = torch::tensor(params.deepnorm_alpha);
    register_buffer("deepnorm_alpha", deepnorm_alpha);
};

torch::Tensor TxEncoderImpl::forward(torch::Tensor x) {
    torch::Tensor attn, f;
    const auto deepnorm_alpha = named_buffers()["deepnorm_alpha"];

    double a, b;

    auto run_norm = [&](RMSNorm norm, const torch::Tensor &in) {
        x = norm(in + (x * deepnorm_alpha));
    };

    // print tens

    a = realtime();
    attn = self_attn(x);
    torch::cuda::synchronize(device_idx);
    b = realtime();
    model_stats->time_self_attn += b-a;

    // print tens

    a = realtime();
    run_norm(norm1, attn);
    torch::cuda::synchronize(device_idx);
    b = realtime();
    model_stats->time_norm1 += b-a;

    a = realtime();
    f = ff(x);
    torch::cuda::synchronize(device_idx);
    b = realtime();
    model_stats->time_ff += b-a;

    // print tens

    a = realtime();
    run_norm(norm2, f);
    torch::cuda::synchronize(device_idx);
    b = realtime();
    model_stats->time_norm2 += b-a;

    // print tens

    // exit
    
    return x;
}

TxEncoderStackImpl::TxEncoderStackImpl(const TxEncoderParams &params, const torch::TensorOptions &options, tx_stats_t *model_stats) {
    stack = Sequential();
    for (int i = 0; i < params.depth; ++i) {
        TxEncoder encoder(params, options, model_stats);
        stack->push_back(register_module("transformer_encoder" + std::to_string(i), encoder));
        layer_vec.push_back(encoder);
    }
    use_i8 = false;
};

torch::Tensor TxEncoderStackImpl::forward(const torch::Tensor &x) {
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

torch::Tensor TxModelImpl::forward(const torch::Tensor &chunk_NCT) {
    torch::Tensor h;
    double a, b;
    auto device_idx = m_options.device_index();
    
    a = realtime();
    h = convs->forward(chunk_NCT);
    torch::cuda::synchronize(device_idx);
    b = realtime();
    model_stats->time_conv_stack += b-a;
    
    a = realtime();
    h = tx_encoder(h);
    torch::cuda::synchronize(device_idx);
    b = realtime();
    model_stats->time_tx_encoder += b-a;

    a = realtime();
    h = tx_decoder(h);
    torch::cuda::synchronize(device_idx);
    b = realtime();
    model_stats->time_tx_decoder += b-a;

    a = realtime();
    h = crf(h);
    torch::cuda::synchronize(device_idx);
    b = realtime();
    model_stats->time_crf += b-a;

    // Returns: NTC
    return h;
}

std::vector<torch::Tensor> load_tx_model_weights(const std::string &dir) {
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

            // tx encoder layer 0
            "transformer_encoder.0.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.0.self_attn.out_proj.weight.tensor",
            "transformer_encoder.0.self_attn.out_proj.bias.tensor",
            "transformer_encoder.0.ff.fc1.weight.tensor",
            "transformer_encoder.0.ff.fc2.weight.tensor",
            "transformer_encoder.0.norm1.weight.tensor",
            "transformer_encoder.0.norm2.weight.tensor",

            // tx encoder layer 1
            "transformer_encoder.1.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.1.self_attn.out_proj.weight.tensor",
            "transformer_encoder.1.self_attn.out_proj.bias.tensor",
            "transformer_encoder.1.ff.fc1.weight.tensor",
            "transformer_encoder.1.ff.fc2.weight.tensor",
            "transformer_encoder.1.norm1.weight.tensor",
            "transformer_encoder.1.norm2.weight.tensor",

            // tx encoder layer 2
            "transformer_encoder.2.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.2.self_attn.out_proj.weight.tensor",
            "transformer_encoder.2.self_attn.out_proj.bias.tensor",
            "transformer_encoder.2.ff.fc1.weight.tensor",
            "transformer_encoder.2.ff.fc2.weight.tensor",
            "transformer_encoder.2.norm1.weight.tensor",
            "transformer_encoder.2.norm2.weight.tensor",

            // tx encoder layer 3
            "transformer_encoder.3.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.3.self_attn.out_proj.weight.tensor",
            "transformer_encoder.3.self_attn.out_proj.bias.tensor",
            "transformer_encoder.3.ff.fc1.weight.tensor",
            "transformer_encoder.3.ff.fc2.weight.tensor",
            "transformer_encoder.3.norm1.weight.tensor",
            "transformer_encoder.3.norm2.weight.tensor",

            // tx encoder layer 4
            "transformer_encoder.4.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.4.self_attn.out_proj.weight.tensor",
            "transformer_encoder.4.self_attn.out_proj.bias.tensor",
            "transformer_encoder.4.ff.fc1.weight.tensor",
            "transformer_encoder.4.ff.fc2.weight.tensor",
            "transformer_encoder.4.norm1.weight.tensor",
            "transformer_encoder.4.norm2.weight.tensor",

            // tx encoder layer 5
            "transformer_encoder.5.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.5.self_attn.out_proj.weight.tensor",
            "transformer_encoder.5.self_attn.out_proj.bias.tensor",
            "transformer_encoder.5.ff.fc1.weight.tensor",
            "transformer_encoder.5.ff.fc2.weight.tensor",
            "transformer_encoder.5.norm1.weight.tensor",
            "transformer_encoder.5.norm2.weight.tensor",

            // tx encoder layer 6
            "transformer_encoder.6.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.6.self_attn.out_proj.weight.tensor",
            "transformer_encoder.6.self_attn.out_proj.bias.tensor",
            "transformer_encoder.6.ff.fc1.weight.tensor",
            "transformer_encoder.6.ff.fc2.weight.tensor",
            "transformer_encoder.6.norm1.weight.tensor",
            "transformer_encoder.6.norm2.weight.tensor",

            // tx encoder layer 7
            "transformer_encoder.7.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.7.self_attn.out_proj.weight.tensor",
            "transformer_encoder.7.self_attn.out_proj.bias.tensor",
            "transformer_encoder.7.ff.fc1.weight.tensor",
            "transformer_encoder.7.ff.fc2.weight.tensor",
            "transformer_encoder.7.norm1.weight.tensor",
            "transformer_encoder.7.norm2.weight.tensor",

            // tx encoder layer 8
            "transformer_encoder.8.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.8.self_attn.out_proj.weight.tensor",
            "transformer_encoder.8.self_attn.out_proj.bias.tensor",
            "transformer_encoder.8.ff.fc1.weight.tensor",
            "transformer_encoder.8.ff.fc2.weight.tensor",
            "transformer_encoder.8.norm1.weight.tensor",
            "transformer_encoder.8.norm2.weight.tensor",

            // tx encoder layer 9
            "transformer_encoder.9.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.9.self_attn.out_proj.weight.tensor",
            "transformer_encoder.9.self_attn.out_proj.bias.tensor",
            "transformer_encoder.9.ff.fc1.weight.tensor",
            "transformer_encoder.9.ff.fc2.weight.tensor",
            "transformer_encoder.9.norm1.weight.tensor",
            "transformer_encoder.9.norm2.weight.tensor",

            // tx encoder layer 10
            "transformer_encoder.10.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.10.self_attn.out_proj.weight.tensor",
            "transformer_encoder.10.self_attn.out_proj.bias.tensor",
            "transformer_encoder.10.ff.fc1.weight.tensor",
            "transformer_encoder.10.ff.fc2.weight.tensor",
            "transformer_encoder.10.norm1.weight.tensor",
            "transformer_encoder.10.norm2.weight.tensor",

            // tx encoder layer 11
            "transformer_encoder.11.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.11.self_attn.out_proj.weight.tensor",
            "transformer_encoder.11.self_attn.out_proj.bias.tensor",
            "transformer_encoder.11.ff.fc1.weight.tensor",
            "transformer_encoder.11.ff.fc2.weight.tensor",
            "transformer_encoder.11.norm1.weight.tensor",
            "transformer_encoder.11.norm2.weight.tensor",

            // tx encoder layer 12
            "transformer_encoder.12.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.12.self_attn.out_proj.weight.tensor",
            "transformer_encoder.12.self_attn.out_proj.bias.tensor",
            "transformer_encoder.12.ff.fc1.weight.tensor",
            "transformer_encoder.12.ff.fc2.weight.tensor",
            "transformer_encoder.12.norm1.weight.tensor",
            "transformer_encoder.12.norm2.weight.tensor",

            // tx encoder layer 13
            "transformer_encoder.13.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.13.self_attn.out_proj.weight.tensor",
            "transformer_encoder.13.self_attn.out_proj.bias.tensor",
            "transformer_encoder.13.ff.fc1.weight.tensor",
            "transformer_encoder.13.ff.fc2.weight.tensor",
            "transformer_encoder.13.norm1.weight.tensor",
            "transformer_encoder.13.norm2.weight.tensor",

            // tx encoder layer 14
            "transformer_encoder.14.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.14.self_attn.out_proj.weight.tensor",
            "transformer_encoder.14.self_attn.out_proj.bias.tensor",
            "transformer_encoder.14.ff.fc1.weight.tensor",
            "transformer_encoder.14.ff.fc2.weight.tensor",
            "transformer_encoder.14.norm1.weight.tensor",
            "transformer_encoder.14.norm2.weight.tensor",

            // tx encoder layer 15
            "transformer_encoder.15.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.15.self_attn.out_proj.weight.tensor",
            "transformer_encoder.15.self_attn.out_proj.bias.tensor",
            "transformer_encoder.15.ff.fc1.weight.tensor",
            "transformer_encoder.15.ff.fc2.weight.tensor",
            "transformer_encoder.15.norm1.weight.tensor",
            "transformer_encoder.15.norm2.weight.tensor",

            // tx encoder layer 16
            "transformer_encoder.16.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.16.self_attn.out_proj.weight.tensor",
            "transformer_encoder.16.self_attn.out_proj.bias.tensor",
            "transformer_encoder.16.ff.fc1.weight.tensor",
            "transformer_encoder.16.ff.fc2.weight.tensor",
            "transformer_encoder.16.norm1.weight.tensor",
            "transformer_encoder.16.norm2.weight.tensor",

            // tx encoder layer 17
            "transformer_encoder.17.self_attn.Wqkv.weight.tensor",
            "transformer_encoder.17.self_attn.out_proj.weight.tensor",
            "transformer_encoder.17.self_attn.out_proj.bias.tensor",
            "transformer_encoder.17.ff.fc1.weight.tensor",
            "transformer_encoder.17.ff.fc2.weight.tensor",
            "transformer_encoder.17.norm1.weight.tensor",
            "transformer_encoder.17.norm2.weight.tensor",

            // tx decoder
            "upsample.linear.weight.tensor",
            "upsample.linear.bias.tensor",

            // linear CRF
            "crf.linear.weight.tensor",
    };

    return load_tensors(dir, tensors);
}

ModuleHolder<AnyModule> load_tx_model(const CRFModelConfig &model_config, const torch::TensorOptions &options, tx_stats_t *model_stats) {
    auto model = TxModel(model_config, options, model_stats);
    auto state_dict = load_tx_model_weights(model_config.model_path);
    model->load_state_dict(state_dict);
    model->to(options.dtype().toScalarType());
    model->to(options.device());
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);
    return holder;
}
