#ifndef ELEPHANT_H
#define ELEPHANT_H

#include <torch/torch.h>
#include <openfish/openfish.h>
#include "slorado.h"
#include "dorado/CRFModel.h"

struct elephant_s {
    std::vector<std::vector<torch::Tensor>> *tensors;
};

struct runner_s {
    std::string device;
    torch::Tensor input_tensor;
    torch::TensorOptions tensor_opts;
    openfish_opt_t decoder_opts;
    torch::nn::ModuleHolder<torch::nn::AnyModule> module{nullptr};
    size_t model_stride;
    size_t chunk_size;
    CRFModelConfig model_config;
#ifdef USE_GPU
    int64_t device_idx;
    openfish_gpubuf_t *gpubuf;
#endif
};

#endif