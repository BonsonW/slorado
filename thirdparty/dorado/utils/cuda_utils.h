#pragma once

#include <torch/torch.h>

void matmul_f16(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor &C);
