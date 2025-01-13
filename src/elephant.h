#ifndef ELEPHANT_H
#define ELEPHANT_H

#include <torch/torch.h>
#include <openfish/openfish.h>

#include "slorado.h"
#include "dorado/model_config.h"

#ifdef HAVE_CUDA
#include "NvInfer.h"
#include "logger.h"
#endif

bool trt_infer(
    std::vector<torch::Tensor> tensors,
    std::vector<Chunk *> chunks,
    const int chunk_size,
    const core_t* core,
    const int runner_idx
);

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
    std::shared_ptr<nvinfer1::ICudaEngine> engine; //!< The TensorRT engine used to run the network

    std::map<std::string, std::string> io; //!< Input and output mapping of the network

    std::unique_ptr<Logger> logger;

    std::unique_ptr<nvinfer1::IExecutionContext> context;
#endif
#endif
};

#endif