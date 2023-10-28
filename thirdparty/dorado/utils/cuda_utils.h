#pragma once

#include <torch/torch.h>

#include <array>
#include <optional>
#include <string>
#include <vector>

void matmul_f16(torch::Tensor const &A, torch::Tensor const &B, torch::Tensor &C);

std::vector<std::string> parse_cuda_device_string(std::string device_string);
