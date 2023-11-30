#include "globals.h"
#include "misc.h"
#include <torch/torch.h>

std::vector<at::Tensor> transposedRNNWeights;
std::vector<at::Tensor> GPUWeights;

bool setTrans = false;