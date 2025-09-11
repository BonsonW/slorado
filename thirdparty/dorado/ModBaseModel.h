#pragma once

#include <torch/torch.h>

#include <vector>

#include "model_config.h"

using namespace torch::nn;

ModuleHolder<AnyModule> load_modbase_model(const ModBaseModelConfig& config,
                                                const at::TensorOptions& options,
                                                const int batchsize);

std::vector<float> load_kmer_refinement_levels(const ModBaseModelConfig& config);
