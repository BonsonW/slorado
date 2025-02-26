#ifndef TORCHBOX_H
#define TORCHBOX_H

#include <torch/torch.h>
#include "slorado.h"

// result + metadata of a chunk
struct chunk_res {
    size_t input_offset;    // raw signal offset
    size_t idx_in_read;     // order in read
    size_t raw_chunk_size;  // size in raw signal

    std::string seq;
    std::string qstring;
    std::vector<uint8_t> moves;
};

// raw signal of a chunk
struct chunk_sig {
    torch::Tensor tensor;
};

struct chunk_db {
    std::vector<std::vector<chunk_res_t>> chunks_res;
    std::vector<std::vector<chunk_sig_t>> chunks_sig;
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
};

#endif