#ifndef CRF_MODEL_H
#define CRF_MODEL_H

#include <torch/torch.h>

#include <vector>

#include "model_config.h"

std::vector<torch::Tensor> load_crf_model_weights(const std::string& dir,  bool decomposition, bool bias);

torch::nn::ModuleHolder<torch::nn::AnyModule> load_crf_model(
    const std::string& path,
    const CRFModelConfig& model_config,
    int batch_size,
    int chunk_size,
    const torch::TensorOptions& options
);

#endif