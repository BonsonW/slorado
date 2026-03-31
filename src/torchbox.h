#ifndef TORCHBOX_H
#define TORCHBOX_H

#include <torch/torch.h>
#include <cstdint>
#include "slorado.h"

// per read data
struct read_dat {
    torch::Tensor scaled_signal;

    // mod data
    char *seq;

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
#ifdef USE_GPU
    int64_t device_idx;
    openfish_gpubuf_t *gpubuf;
#endif

    // modbase stuff
    at::Tensor input_sigs;
    at::Tensor input_seqs;
};

#endif