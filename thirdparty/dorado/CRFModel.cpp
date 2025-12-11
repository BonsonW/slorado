#include <math.h>
#include <string>

#include "CRFModel.h"
#include "error.h"
#include "tensor_chunk_utils.h"
#include <ATen/cuda/CUDAContext.h>

using namespace torch::indexing;
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
    auto scores = linear(x);
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
#ifdef HAVE_CUDA
    auto training = false;
    auto batch_size = 1;
    auto hidden_size = size;
    auto num_heads = 4;
    fused_rnn_0 = new FlashRNNFuncFused(training, batch_size, hidden_size, num_heads);
#endif
};

LSTMStackImpl::~LSTMStackImpl() {
#ifdef HAVE_CUDA
    delete fused_rnn_0;
#endif
}

int round_to_multiple(int n, int m) {
    return ((n + m - 1) / m) * m;
}

torch::Tensor LSTMStackImpl::forward(torch::Tensor x) {
    // Input is [N, T, C], contiguity optional
    for (uint i = 0; i < rnns.size(); ++i) {
        auto rnn = rnns[i];
#ifdef HAVE_CUDA
        if (i == 0) {
            auto batch_size = x.size(0);
            auto seqlen = x.size(1);
            auto nheads = 1;
            auto ngates = 4;
            auto num_states = 2;
            auto head_dim = rnn->options.hidden_size();
            fprintf(stderr, "batch_size: %ld\n", batch_size);
            fprintf(stderr, "seqlen: %ld\n", seqlen);
            fprintf(stderr, "nheads: %d\n", nheads);
            fprintf(stderr, "ngates: %d\n", ngates);
            fprintf(stderr, "num_states: %d\n", num_states);
            fprintf(stderr, "head_dim: %ld\n", head_dim);

            // B = batch, T time/sequence dim, N num heads, S state dimension, P previous D dimension (R matrix)
            // D head dim or hidden dim, G gates
            // x = x.matmul(rnn->named_parameters()["weight_ih_l0"]) + rnn->named_parameters()["bias_ih_l0"];
            // elif self.backend == "cuda_fused":
            //     self._internal_input_shape = "TBHDG"
            //     self._internal_recurrent_shape = "HDGP"
            //     self._internal_bias_shape = "HDG"
            //     self._internal_output_shape = "STBHD"

            x = x.flip(1);

            auto bih = rnn->named_parameters()["bias_ih_l0"];
            auto wih = rnn->named_parameters()["weight_ih_l0"];

            auto linear = torch::nn::Linear(head_dim, head_dim);
            // You MUST clone + detach when overwriting parameters
            linear->weight = wih.clone().detach();
            linear->bias   = bih.clone().detach();

            auto torch_out = linear->forward(x);

            // int64_t in_f  = x.size(2);
            // TORCH_CHECK(wih.size(1) == in_f, "Weight in_features mismatch");
            // // Flatten batch & seq for matmul: [batch*seq, in_features]
            // auto x2d = x.reshape({batch_size * seqlen, in_f});
            // auto my_out = at::addmm(bih, x2d, wih.t());
            
            // auto diff = torch::max(torch::abs(torch_out.flatten() - my_out.flatten())).item<float>();
            // std::cerr << "Max abs diff: " << diff << "\n";
            // exit(0);
            
            x = torch_out.view({batch_size, seqlen, ngates, nheads, head_dim}).permute({0, 1, 3, 4, 2}).to(torch::kBFloat16).contiguous();
            // fprintf(stderr, "0: %ld, 1: %ld, 2: %ld, 3: %ld, 4: %ld\n", x.size(0), x.size(1), x.size(2), x.size(3), x.size(4));
            auto recurrent_kernel = rnn->named_parameters()["weight_hh_l0"].view({ngates, nheads, head_dim, head_dim}).permute({1, 2, 0, 3}).to(torch::kBFloat16).contiguous();
            auto bias = rnn->named_parameters()["bias_hh_l0"].view({ngates, nheads, head_dim}).permute({1, 2, 0}).to(torch::kBFloat16).contiguous();
            
            auto options = x.options();

            torch::Tensor s0 = torch::zeros({num_states, batch_size, 1, nheads, head_dim}, options.dtype(torch::kBFloat16)).contiguous();
            s0 = s0.index({Slice(), 0});
            torch::Tensor states = torch::empty({FLASHRNN_NUM_STATES, seqlen + 1, batch_size, nheads, head_dim}, options.dtype(torch::kBFloat16)).contiguous();
            torch::Tensor gate_cache_i = torch::empty({}, options.dtype(torch::kBFloat16)).contiguous();
            torch::Tensor gate_cache_r = torch::empty({seqlen, batch_size, nheads, head_dim, FLASHRNN_NUM_GATES_R}, options.dtype(torch::kBFloat16)).contiguous();
            torch::Tensor gate_buffer = torch::ones({
                FLASHRNN_FORWARD_RECURRENT_TILING_COUNT_HIDDEN *
                FLASHRNN_FORWARD_BLOCK_TILING_COUNT_BATCH *
                FLASHRNN_FORWARD_WARP_TILING_COUNT_BATCH *
                FLASHRNN_FORWARD_WARP_LOOPING_COUNT_BATCH *
                FLASHRNN_FORWARD_WARP_TILING_DIM_BATCH,
                FLASHRNN_NUM_GATES_R * head_dim
            }, options.dtype(torch::kFloat32)).contiguous();

            for (uint i = 0; i < FLASHRNN_NUM_STATES; i++) {
                states[i][0] = s0[i];
            }

            cudaStream_t stream = at::cuda::getCurrentCUDAStream();
            cublasHandle_t cublas_handle;
            cublasCreate(&cublas_handle);
            cublasSetStream(cublas_handle, stream);
            fused_rnn_0->forward(
                false, // training
                x.data_ptr(), // W_ih * x + b_ih
                s0.data_ptr(),
                recurrent_kernel.data_ptr(), // W_hh
                bias.data_ptr(), // b_hh
                states.data_ptr(),
                gate_cache_r.data_ptr(),
                gate_cache_i.data_ptr(),
                gate_buffer.data_ptr(),
                seqlen,
                batch_size,
                nheads,
                head_dim,
                (void *)cublas_handle
            );

            cublasDestroy(cublas_handle);

            x = states
                .index({Slice(), Slice(1, None)})[0]
                .permute({1, 0, 2, 3})
                .reshape({batch_size, seqlen, head_dim})
                .contiguous()
                .to(torch::kFloat16);
        } else
#endif  
        {
            x = std::get<0>(rnn(x.flip(1)));
        }
    }

    // Output is [N, T, C], contiguous
    return (rnns.size() & 1) ? x.flip(1) : x;
}

ClampImpl::ClampImpl(float _min, float _max, bool _active)
        : active(_active), min(_min), max(_max) {}

torch::Tensor ClampImpl::forward(torch::Tensor x) {
    if (active) {
        x.clamp_(min, max);
    }
    return x;
}

CRFModelImpl::CRFModelImpl(const CRFModelConfig &config) {
    const auto cv = config.convs;
    const auto lstm_size = config.lstm_size;
    convs = register_module("convs", ConvStack(cv));
    rnns = register_module("rnns", LSTMStack(5, lstm_size));

    if (config.has_out_features) {
        // The linear layer is decomposed into 2 matmuls.
        const int decomposition = config.out_features;
        linear1 = register_module("linear1", LinearCRF(lstm_size, decomposition, true, false));
        linear2 = register_module("linear2", LinearCRF(decomposition, config.outsize, false, false));
        clamp1 = Clamp(-5.0, 5.0, config.clamp);
        encoder = Sequential(convs, rnns, linear1, linear2, clamp1);
    } else if ((config.convs[0].size > 4) && (config.num_features == 1)) {
        // v4.x model without linear decomposition
        linear1 = register_module("linear1", LinearCRF(lstm_size, config.outsize, false, false));
        clamp1 = Clamp(-5.0, 5.0, config.clamp);
        encoder = Sequential(convs, rnns, linear1, clamp1);
    } else {
        // Pre v4 model
        linear1 = register_module("linear1", LinearCRF(lstm_size, config.outsize, true, true));
        encoder = Sequential(convs, rnns, linear1);
    }
}

void CRFModelImpl::load_state_dict(const std::vector<torch::Tensor> &weights) {
    module_load_state_dict(*this, weights);
}

torch::Tensor CRFModelImpl::forward(const torch::Tensor &x) {
    // Output is [N, T, C]
    return encoder->forward(x);
}

std::vector<torch::Tensor> load_lstm_model_weights(const std::string &dir,
                                                  bool decomposition,
                                                  bool bias) {
    auto tensors = std::vector<std::string>{
        "0.conv.weight.tensor",      "0.conv.bias.tensor",

        "1.conv.weight.tensor",      "1.conv.bias.tensor",

        "2.conv.weight.tensor",      "2.conv.bias.tensor",

        "4.rnn.weight_ih_l0.tensor", "4.rnn.weight_hh_l0.tensor",
        "4.rnn.bias_ih_l0.tensor",   "4.rnn.bias_hh_l0.tensor",

        "5.rnn.weight_ih_l0.tensor", "5.rnn.weight_hh_l0.tensor",
        "5.rnn.bias_ih_l0.tensor",   "5.rnn.bias_hh_l0.tensor",

        "6.rnn.weight_ih_l0.tensor", "6.rnn.weight_hh_l0.tensor",
        "6.rnn.bias_ih_l0.tensor",   "6.rnn.bias_hh_l0.tensor",

        "7.rnn.weight_ih_l0.tensor", "7.rnn.weight_hh_l0.tensor",
        "7.rnn.bias_ih_l0.tensor",   "7.rnn.bias_hh_l0.tensor",

        "8.rnn.weight_ih_l0.tensor", "8.rnn.weight_hh_l0.tensor",
        "8.rnn.bias_ih_l0.tensor",   "8.rnn.bias_hh_l0.tensor",

        "9.linear.weight.tensor"
    };

    if (bias) {
        tensors.push_back("9.linear.bias.tensor");
    }

    if (decomposition) {
        tensors.push_back("10.linear.weight.tensor");
    }

    return load_tensors(dir, tensors);
}

ModuleHolder<AnyModule> load_lstm_model(const CRFModelConfig &model_config, const torch::TensorOptions &options) {
    auto model = CRFModel(model_config);
    auto state_dict = load_lstm_model_weights(model_config.model_path, model_config.has_out_features, model_config.bias);
    model->load_state_dict(state_dict);
    model->to(options.dtype().toScalarType());
    model->to(options.device());
    model->eval();

    auto module = AnyModule(model);
    auto holder = ModuleHolder<AnyModule>(module);
    return holder;
}