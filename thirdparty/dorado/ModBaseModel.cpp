#include "ModBaseModel.h"
#include "TxModel.h"
#include "tensor_chunk_utils.h"

#include <optional>

#include <stdexcept>
#include <string>
#include <vector>

using namespace torch::nn;
using namespace torch::indexing;

template <class Model>
ModuleHolder<AnyModule> populate_model(Model&& model,
                                       const std::string &path,
                                       const at::TensorOptions& options) {
    auto state_dict = model->load_weights(path);
    model->load_state_dict(state_dict);
    model->to(options.dtype_opt().value().toScalarType());
    model->to(options.device_opt().value());
    model->eval();

    auto module = AnyModule(std::move(model));
    auto holder = ModuleHolder<AnyModule>(std::move(module));
    return holder;
}

std::vector<std::string> load_modbase_conv_lstm_weights(const ModBaseModelConfig& config) {
    std::vector<std::string> tensors{
        "sig_conv1.weight.tensor",   "sig_conv1.bias.tensor",
        "sig_conv2.weight.tensor",   "sig_conv2.bias.tensor",
        "sig_conv3.weight.tensor",   "sig_conv3.bias.tensor",

        "seq_conv1.weight.tensor",   "seq_conv1.bias.tensor",
        "seq_conv2.weight.tensor",   "seq_conv2.bias.tensor",

        "merge_conv1.weight.tensor", "merge_conv1.bias.tensor",

        "lstm1.weight_ih_l0.tensor", "lstm1.weight_hh_l0.tensor",
        "lstm1.bias_ih_l0.tensor",   "lstm1.bias_hh_l0.tensor",

        "lstm2.weight_ih_l0.tensor", "lstm2.weight_hh_l0.tensor",
        "lstm2.bias_ih_l0.tensor",   "lstm2.bias_hh_l0.tensor",

        "fc.weight.tensor",          "fc.bias.tensor",
    };

    if (config.general.modules.has_value() && config.general.modules->upsample.has_value()) {
        tensors.push_back("linear_up.linear.weight.tensor");
        tensors.push_back("linear_up.linear.bias.tensor");
    }

    return tensors;
}

struct ModsConvImpl : Module {
    ModsConvImpl(int size, int outsize, int k, int stride, int padding)
            : name("conv_act_" + std::to_string(size)) {
        conv = register_module(
                "conv", Conv1d(Conv1dOptions(size, outsize, k).stride(stride).padding(padding)));
        activation = Activation::SWISH;
    }

    ModsConvImpl(const ConvParams& params)
            : name("conv_act_" + std::to_string(params.size)) {
        conv = register_module("conv",
                               Conv1d(Conv1dOptions(params.insize, params.size, params.winlen)
                                              .stride(params.stride)
                                              .padding(params.winlen / 2)));
        activation = params.activation;
    }

    at::Tensor forward(const at::Tensor& x) {
        at::Tensor x_ = conv(x);

        switch (activation) {
        case Activation::SWISH:
            return at::silu(x_);
        case Activation::SWISH_CLAMP:
            throw std::runtime_error("ModsConv is not implemented for SWISH_CLAMP");
        case Activation::TANH:
            return at::tanh(x_);
        }
        throw std::logic_error("ModsConv has unsupported activation");
    }

    const std::string name;
    Conv1d conv{nullptr};
    Activation activation;
};

TORCH_MODULE(ModsConv);

struct ModBaseConvModelImpl : Module {
    ModBaseConvModelImpl(int size, int kmer_len, int num_out) {
        sig_conv1 = register_module("sig_conv1", ModsConv(1, 4, 11, 1, 0));
        sig_conv2 = register_module("sig_conv2", ModsConv(4, 16, 11, 1, 0));
        sig_conv3 = register_module("sig_conv3", ModsConv(16, size, 9, 3, 0));

        seq_conv1 = register_module("seq_conv1", ModsConv(kmer_len * 4, 16, 11, 1, 0));
        seq_conv2 = register_module("seq_conv2", ModsConv(16, 32, 11, 1, 0));
        seq_conv3 = register_module("seq_conv3", ModsConv(32, size, 9, 3, 0));

        merge_conv1 = register_module("merge_conv1", ModsConv(size * 2, size, 5, 1, 0));
        merge_conv2 = register_module("merge_conv2", ModsConv(size, size, 5, 1, 0));
        merge_conv3 = register_module("merge_conv3", ModsConv(size, size, 3, 2, 0));
        merge_conv4 = register_module("merge_conv4", ModsConv(size, size, 3, 2, 0));

        linear = register_module("linear", Linear(size * 3, num_out));
    }

    at::Tensor forward(at::Tensor sigs, at::Tensor seqs) {
        sigs = sig_conv1(sigs);
        sigs = sig_conv2(sigs);
        sigs = sig_conv3(sigs);

        // We are supplied one hot encoded sequences as (batch, signal, kmer_len * base_count) int8.
        // We need (batch, kmer_len * base_count, signal) and a dtype compatible with the float16
        // weights.
        const auto conv_dtype = (seqs.device() == torch::kCPU) ? torch::kFloat32 : torch::kFloat16;
        seqs = seqs.permute({0, 2, 1}).to(conv_dtype);
        seqs = seq_conv1(seqs);
        seqs = seq_conv2(seqs);
        seqs = seq_conv3(seqs);

        auto z = torch::cat({sigs, seqs}, 1);

        z = merge_conv1(z);
        z = merge_conv2(z);
        z = merge_conv3(z);
        z = merge_conv4(z);

        z = z.flatten(1);
        z = linear(z);

        z = z.softmax(1);

        return z;
    }

    void load_state_dict(const std::vector<at::Tensor>& weights) {
        module_load_state_dict(*this, weights);
    }

    std::vector<at::Tensor> load_weights(const std::string& dir) {
        return load_tensors(dir, weight_tensors);
    }

    static const std::vector<std::string> weight_tensors;

    ModsConv sig_conv1{nullptr};
    ModsConv sig_conv2{nullptr};
    ModsConv sig_conv3{nullptr};
    ModsConv seq_conv1{nullptr};
    ModsConv seq_conv2{nullptr};
    ModsConv seq_conv3{nullptr};
    ModsConv merge_conv1{nullptr};
    ModsConv merge_conv2{nullptr};
    ModsConv merge_conv3{nullptr};
    ModsConv merge_conv4{nullptr};
    Linear linear{nullptr};
};

const std::vector<std::string> ModBaseConvModelImpl::weight_tensors{
        "sig_conv1.weight.tensor",   "sig_conv1.bias.tensor",
        "sig_conv2.weight.tensor",   "sig_conv2.bias.tensor",
        "sig_conv3.weight.tensor",   "sig_conv3.bias.tensor",

        "seq_conv1.weight.tensor",   "seq_conv1.bias.tensor",
        "seq_conv2.weight.tensor",   "seq_conv2.bias.tensor",
        "seq_conv3.weight.tensor",   "seq_conv3.bias.tensor",

        "merge_conv1.weight.tensor", "merge_conv1.bias.tensor",
        "merge_conv2.weight.tensor", "merge_conv2.bias.tensor",
        "merge_conv3.weight.tensor", "merge_conv3.bias.tensor",
        "merge_conv4.weight.tensor", "merge_conv4.bias.tensor",

        "fc.weight.tensor",          "fc.bias.tensor",
};

struct ModBaseConvLSTMModelImpl : Module {
    ModBaseConvLSTMModelImpl(const ModBaseModelConfig& config)
            : m_config(config),
              //int size, int kmer_len, int num_out, bool is_conv_lstm_v2, int stride
              m_is_conv_lstm_v2(config.is_chunked_input_model()) {
        const auto& params = m_config.general;
        // conv_lstm_v2 models are padded to ensure the output shape is nicely indexable by the stride
        sig_conv1 = register_module("sig_conv1", ModsConv(1, 4, 5, 1, m_is_conv_lstm_v2 ? 2 : 0));
        sig_conv2 = register_module("sig_conv2", ModsConv(4, 16, 5, 1, m_is_conv_lstm_v2 ? 2 : 0));
        sig_conv3 = register_module("sig_conv3", ModsConv(16, params.size, 9, params.stride,
                                                          m_is_conv_lstm_v2 ? 4 : 0));

        seq_conv1 = register_module(
                "seq_conv1", ModsConv(params.kmer_len * 4, 16, 5, 1, m_is_conv_lstm_v2 ? 2 : 0));
        seq_conv2 = register_module("seq_conv2", ModsConv(16, params.size, 13, params.stride,
                                                          m_is_conv_lstm_v2 ? 6 : 0));

        merge_conv1 = register_module("merge_conv1", ModsConv(params.size * 2, params.size, 5, 1,
                                                              m_is_conv_lstm_v2 ? 2 : 0));

        lstm1 = register_module("lstm1", LSTM(LSTMOptions(params.size, params.size)));
        lstm2 = register_module("lstm2", LSTM(LSTMOptions(params.size, params.size)));

        linear = register_module("linear", Linear(params.size, params.num_out));

        activation = register_module("activation", SiLU());
    }

    at::Tensor forward(at::Tensor sigs, at::Tensor seqs) {
        // INPUT sigs: NCT & seqs: NTC
        sigs = sig_conv1(sigs);
        sigs = sig_conv2(sigs);
        sigs = sig_conv3(sigs);

        // We are supplied one hot encoded sequences as (batch, signal, kmer_len * base_count) int8.
        // We need (batch, kmer_len * base_count, signal) and a dtype compatible with the float16
        // weights.
        const auto conv_dtype = (seqs.device() == torch::kCPU) ? torch::kFloat32 : torch::kFloat16;

        // seqs: NTC -> NCT
        seqs = seqs.permute({0, 2, 1}).to(conv_dtype);
        seqs = seq_conv1(seqs);
        seqs = seq_conv2(seqs);

        // z: NCT
        auto z = torch::cat({sigs, seqs}, 1);
        // z: NCT -> TNC
        z = merge_conv1(z).permute({2, 0, 1});
        auto [z1, h1] = lstm1(z);
        z = activation(z1).flip(0);
        auto [z2, h2] = lstm2(z);
        z = activation(z2).flip(0);

        if (m_is_conv_lstm_v2) {
            // TNC -> NTC
            z = linear(z.permute({1, 0, 2})).softmax(2).flatten(1);
        } else {
            // Take the final time step: TNC -> tNC -> NC
            z = z.index({-1}).permute({0, 1});
            z = linear(z).softmax(1);
        }
        return z;
    }

    void load_state_dict(const std::vector<at::Tensor>& weights) {
        module_load_state_dict(*this, weights);
    }

    std::vector<at::Tensor> load_weights(const std::string& dir) {
        return load_tensors(dir, load_modbase_conv_lstm_weights(m_config));
    }

    const ModBaseModelConfig m_config;
    const bool m_is_conv_lstm_v2{false};
    static const std::vector<std::string> weight_tensors;

    ModsConv sig_conv1{nullptr};
    ModsConv sig_conv2{nullptr};
    ModsConv sig_conv3{nullptr};
    ModsConv seq_conv1{nullptr};
    ModsConv seq_conv2{nullptr};
    ModsConv merge_conv1{nullptr};

    LSTM lstm1{nullptr};
    LSTM lstm2{nullptr};

    Linear linear{nullptr};
    SiLU activation{nullptr};
};

struct ModBaseConvLSTMV3ModelImpl : Module {
    ModBaseConvLSTMV3ModelImpl(const ModBaseModelConfig& config) : m_config(config) {
        const auto& sig_cvs = m_config.general.modules->signal_convs;
        const auto& seq_cvs = m_config.general.modules->sequence_convs;
        const auto& merge_cv = m_config.general.modules->merge_conv;
        const auto& lstms = m_config.general.modules->lstms;
        const auto& ll = m_config.general.modules->linear;
        const auto& lu = m_config.general.modules->upsample;

        if (sig_cvs.size() != 3) {
            throw std::runtime_error("ModBaseConvLSTMV3Model expected 3 signal convolutions");
        }
        if (seq_cvs.size() != 2) {
            throw std::runtime_error("ModBaseConvLSTMV3Model expected 2 sequence convolutions");
        }
        if (lstms.size() != 2) {
            throw std::runtime_error("ModBaseConvLSTMV3Model expected 2 lstms");
        }

        sig_conv1 = register_module("sig_conv1", ModsConv(sig_cvs.at(0)));
        sig_conv2 = register_module("sig_conv2", ModsConv(sig_cvs.at(1)));
        sig_conv3 = register_module("sig_conv3", ModsConv(sig_cvs.at(2)));

        seq_conv1 = register_module("seq_conv1", ModsConv(seq_cvs.at(0)));
        seq_conv2 = register_module("seq_conv2", ModsConv(seq_cvs.at(1)));

        merge_conv1 = register_module("merge_conv1", ModsConv(merge_cv));

        lstm1 = register_module("lstm1", LSTM(LSTMOptions(lstms.at(0).size, lstms.at(0).size)));
        lstm2 = register_module("lstm2", LSTM(LSTMOptions(lstms.at(1).size, lstms.at(1).size)));

        linear = register_module("linear", Linear(ll.in_size, ll.out_size));

        if (lu.has_value()) {
            upsample = register_module("upsample", LinearUpsample(lu.value()));
        }
    }

    at::Tensor forward(at::Tensor sigs, at::Tensor seqs) {
        // INPUT sigs: NCT & seqs: NTC
        sigs = sig_conv1(sigs);
        sigs = sig_conv2(sigs);
        sigs = sig_conv3(sigs);

        // We are supplied one hot encoded sequences as (batch, signal/stride, kmer_len * base_count) int8.
        // We need (batch, kmer_len * base_count, signal/stride) and a dtype compatible with the float16
        // weights.
        const auto conv_dtype = (seqs.device() == torch::kCPU) ? torch::kFloat32 : torch::kFloat16;

        // seqs: NTC -> NCT
        seqs = seqs.permute({0, 2, 1}).to(conv_dtype);
        seqs = seq_conv1(seqs);
        seqs = seq_conv2(seqs);

        // z: NCT -> N(2C)T
        auto z = torch::cat({sigs, seqs}, 1);
        z = merge_conv1(z).permute({2, 0, 1});  // NCT -> TNC
        z = std::get<0>(lstm1(z)).flip(0);  // TNC -> T'NC
        z = std::get<0>(lstm2(z));  // T'NC
        z = linear(z).flip(0).permute({1, 0, 2});  // T'NC -> NTC
        z = upsample->forward(z);  // NTC -> N(sf*T)C
        // NTC -> N(TC)
        return z.softmax(2).flatten(1);
    }

    void load_state_dict(const std::vector<at::Tensor>& weights) {
        module_load_state_dict(*this, weights);
    }

    std::vector<at::Tensor> load_weights(const std::string& dir) {
        return load_tensors(dir, load_modbase_conv_lstm_weights(m_config));
    }

    static const std::vector<std::string> weight_tensors;

    const ModBaseModelConfig m_config;

    ModsConv sig_conv1{nullptr};
    ModsConv sig_conv2{nullptr};
    ModsConv sig_conv3{nullptr};
    ModsConv seq_conv1{nullptr};
    ModsConv seq_conv2{nullptr};
    ModsConv merge_conv1{nullptr};

    LSTM lstm1{nullptr};
    LSTM lstm2{nullptr};

    Linear linear{nullptr};
    LinearUpsample upsample{nullptr};
};

TORCH_MODULE(ModBaseConvModel);
TORCH_MODULE(ModBaseConvLSTMModel);
TORCH_MODULE(ModBaseConvLSTMV3Model);

ModuleHolder<AnyModule> load_modbase_model(const ModBaseModelConfig& config,
                                                const at::TensorOptions& options,
                                                [[maybe_unused]] const int batchsize) {
    at::InferenceMode guard;
    const auto params = config.general;
    switch (params.model_type) {
    case ModelType::CONV_LSTM_V1: {
        auto model = ModBaseConvLSTMModel(config);
        return populate_model(std::move(model), config.model_path, options);
    }
    case ModelType::CONV_LSTM_V2: {
        auto model = ModBaseConvLSTMModel(config);
        return populate_model(std::move(model), config.model_path, options);
    }
    case ModelType::CONV_LSTM_V3: {
        auto model = ModBaseConvLSTMV3Model(config);
        return populate_model(std::move(model), config.model_path, options);
    }
    case ModelType::CONV_V1: {
        auto model = ModBaseConvModel(params.size, params.kmer_len, params.num_out);
        return populate_model(std::move(model), config.model_path, options);
    }
    default:
        throw std::runtime_error("Unknown modbase model type in config file.");
    }
}

std::vector<float> load_kmer_refinement_levels(const ModBaseModelConfig& config) {
    std::vector<float> levels;
    if (!config.refine.do_rough_rescale) {
        return levels;
    }

    std::vector<at::Tensor> tensors = load_tensors(config.model_path, {"refine_kmer_levels.tensor"});
    if (tensors.empty()) {
        throw std::runtime_error("Failed to load modbase refinement tensors.");
    }
    auto& t = tensors.front();
    t.contiguous();
    levels.reserve(t.numel());
    std::copy(t.data_ptr<float>(), t.data_ptr<float>() + t.numel(), std::back_inserter(levels));
    return levels;
}
