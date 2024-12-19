#ifndef ELEPHANT_H
#define ELEPHANT_H

#include <torch/torch.h>
#include <openfish/openfish.h>

#include "slorado.h"
#include "dorado/model_config.h"

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

#ifdef HAVE_CUDA
    nvinfer1::Dims input_dims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims output_dims; //!< The dimensions of the output to the network.

    std::unique_ptr<nvinfer1::IRuntime> runtime;   //!< The TensorRT runtime used to deserialize the engine
    std::unique_ptr<nvinfer1::ICudaEngine> engine; //!< The TensorRT engine used to run the network

    std::map<std::string, std::string> io; //!< Input and output mapping of the network
    std::unordered_map<std::string, void *> device_buffers; //!< Mapping from tensor name to buffer
    std::unordered_map<std::string, void *> host_buffers; //!< Mapping from tensor name to buffer
#endif
#endif
};

#endif