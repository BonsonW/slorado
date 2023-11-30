#ifndef GLOBALS_H
#define GLOBALS_H

#include <chrono>
#include <cstddef>
#include <torch/torch.h>

extern std::vector<at::Tensor> transposedRNNWeights;
extern std::vector<at::Tensor> GPUWeights;
extern bool setTrans;

#endif