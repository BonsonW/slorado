#ifndef TORCHBOX_H
#define TORCHBOX_H

#include <torch/torch.h>
#include <cstdint>
#include "slorado.h"

// per read data
struct read_dat {
    torch::Tensor scaled_signal;

    // Number of samples trimmed from the FRONT of the raw signal by the basecall scaler
    // (RNA adapter / DNA pore-open trim). The modbase path must trim the raw signal by the
    // same amount so its signal stays aligned with the moves/sequence (which are derived from
    // the trimmed basecall signal). Set in preprocess_signal, consumed in preprocess_modbase.
    int64_t basecall_trim_start = 0;

    // mod data
    const char *seq;

    std::vector<int8_t> encoded_kmers;
    std::array<std::vector<int64_t>, 4> per_base_hits_seq; // sequence indices for hits for each base (i.e. one per model)
    std::array<std::vector<int64_t>, 4> per_base_hits_sig; // signal indices for hits for each base (i.e. one per model)
    int64_t target_start;

    std::vector<uint8_t> base_mod_probs;
    std::vector<bool> base_mod_simplex_motif_hits;
};

struct runner {
    std::string device;
    torch::Tensor input_tensor;
    torch::TensorOptions tensor_opts;
    torch::nn::ModuleHolder<torch::nn::AnyModule> module{nullptr};
    // Procedural basecall model (ported families). When non-null, model_forward() uses it instead
    // of the torch::nn `module`. Opaque here to avoid pulling model headers into slorado.h.
    void *bc_model = nullptr;
    model_family_t bc_family;   // valid when bc_model != nullptr
#ifdef USE_GPU
    int64_t device_idx;
    openfish_gpubuf_t *gpubuf;
    int decode_tile;   // rows decoded per openfish call (bounds decode scratch; <= gpu batch)
#endif

    // modbase stuff
    at::Tensor input_sigs;
    at::Tensor input_seqs;
};

// Run a basecall runner's model forward: procedural model if bc_model is set, else the torch::nn
// module. (Modbase runners call module->forward directly.)
at::Tensor model_forward(runner_t *runner, const at::Tensor &x);

#endif