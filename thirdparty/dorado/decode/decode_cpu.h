#pragma once

#include "decode.h"
#include "../nn/CRFModel.h"
#include "slorado.h"

#include <torch/torch.h>

void decode_cpu(const torch::Tensor& scores, std::vector<DecodedChunk>& chunk_results, const int num_chunks, const core_t* core, const int runner_idx);