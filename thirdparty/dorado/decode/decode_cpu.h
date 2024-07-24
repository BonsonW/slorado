#pragma once

#include "decode.h"
#include "../nn/CRFModel.h"
#include "slorado.h"

#include <torch/torch.h>

std::vector<DecodedChunk> decode_cpu(const torch::Tensor& scores, const int num_chunks, const runner_t *runner);