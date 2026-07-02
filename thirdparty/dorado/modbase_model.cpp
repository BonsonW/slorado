#include "modbase_model.h"
#include "tensor_chunk_utils.h"

#include <string>
#include <vector>

using namespace torch::indexing;

// A modbase ConvLSTM conv is always SWISH (silu). x stays [N, C, T] (no transpose).
static at::Tensor mods_conv(const conv_layer_t &c, at::Tensor x) {
    return at::silu(at::conv1d(x, c.w, c.b, c.stride, c.padding));
}

// Default (batch_first=false) single-layer LSTM over [T, N, C].
static at::Tensor mb_lstm(const lstm_layer_t &l, at::Tensor x) {
    const int64_t N = x.size(1), C = x.size(2);
    auto h0 = torch::zeros({1, N, C}, x.options());
    auto c0 = torch::zeros({1, N, C}, x.options());
    return std::get<0>(torch::lstm(x, {h0, c0}, {l.w_ih, l.w_hh, l.b_ih, l.b_hh},
                                   /*has_biases*/ true, /*num_layers*/ 1, /*dropout*/ 0.0,
                                   /*train*/ false, /*bidirectional*/ false, /*batch_first*/ false));
}

at::Tensor modbase_model_forward(const modbase_model_t *m, at::Tensor sigs, at::Tensor seqs) {
    // sigs: NCT
    sigs = mods_conv(m->sig_conv[0], sigs);
    sigs = mods_conv(m->sig_conv[1], sigs);
    sigs = mods_conv(m->sig_conv[2], sigs);

    // seqs: one-hot [N, T, kmer_len*4] int8 -> NCT in the conv dtype
    const auto conv_dtype = (seqs.device() == torch::kCPU) ? torch::kFloat32 : torch::kFloat16;
    seqs = seqs.permute({0, 2, 1}).to(conv_dtype);
    seqs = mods_conv(m->seq_conv[0], seqs);
    seqs = mods_conv(m->seq_conv[1], seqs);

    auto z = torch::cat({sigs, seqs}, 1);   // NCT
    z = mods_conv(m->merge_conv, z).permute({2, 0, 1});  // NCT -> TNC
    z = at::silu(mb_lstm(m->lstm1, z)).flip(0);
    z = at::silu(mb_lstm(m->lstm2, z)).flip(0);

    if (m->chunked) {
        z = z.permute({1, 0, 2});  // TNC -> NTC
        // matmul+bias (F::linear semantics for 3D input, matching torch::nn::Linear)
        auto out = z.matmul(m->linear_w.t()) + m->linear_b;
        return out.softmax(2).flatten(1);
    }
    // v1: take the final time step (TNC -> NC), then linear (2D -> addmm)
    z = z.index({-1});
    return at::linear(z, m->linear_w, m->linear_b).softmax(1);
}

modbase_model_t *load_modbase_model_proc(const modbase_model_config_t &config, const at::TensorOptions &options, int /*batchsize*/) {
    modbase_model_t *m = new modbase_model_t();
    m->chunked = is_chunked_input_model(config);
    const auto &p = config.general;
    const int stride = p.stride;
    const bool v2 = m->chunked;

    // Weight files, in load order (ConvLSTM; the supported model has no upsample).
    auto t = load_tensors(config.model_path, {
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
    });
    const auto dtype = options.dtype_opt().value().toScalarType();
    const auto dev = options.device_opt().value();
    auto to_dev = [&](const at::Tensor &x) { return x.to(dtype).to(dev); };

    // Conv strides/paddings match ModBaseConvLSTMModel (v2 pads to keep the stride indexable).
    auto setconv = [&](conv_layer_t &c, int wi, int stride_, int pad) {
        c.w = to_dev(t[wi]);
        c.b = to_dev(t[wi + 1]);
        c.stride = stride_;
        c.padding = pad;
        c.activation = Activation::SWISH;
    };
    setconv(m->sig_conv[0], 0, 1,      v2 ? 2 : 0);
    setconv(m->sig_conv[1], 2, 1,      v2 ? 2 : 0);
    setconv(m->sig_conv[2], 4, stride, v2 ? 4 : 0);
    setconv(m->seq_conv[0], 6, 1,      v2 ? 2 : 0);
    setconv(m->seq_conv[1], 8, stride, v2 ? 6 : 0);
    setconv(m->merge_conv, 10, 1,      v2 ? 2 : 0);

    m->lstm1.w_ih = to_dev(t[12]); m->lstm1.w_hh = to_dev(t[13]);
    m->lstm1.b_ih = to_dev(t[14]); m->lstm1.b_hh = to_dev(t[15]);
    m->lstm2.w_ih = to_dev(t[16]); m->lstm2.w_hh = to_dev(t[17]);
    m->lstm2.b_ih = to_dev(t[18]); m->lstm2.b_hh = to_dev(t[19]);
    m->linear_w = to_dev(t[20]);
    m->linear_b = to_dev(t[21]);

    flatten_lstm_weights(m->lstm1, p.size, p.size, /*batch_first=*/false);
    flatten_lstm_weights(m->lstm2, p.size, p.size, /*batch_first=*/false);

    return m;
}

void free_modbase_model(modbase_model_t *m) {
    delete m;
}
