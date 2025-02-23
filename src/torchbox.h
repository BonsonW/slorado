#ifndef TORCHBOX_H
#define TORCHBOX_H

#include <torch/torch.h>

#include "slorado.h"

struct tensor_db {
    std::vector<std::vector<torch::Tensor>> *tensors;
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