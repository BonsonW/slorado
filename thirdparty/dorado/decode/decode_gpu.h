#pragma once

#include "decode.h"
#include "../nn/CRFModel.h"
#include "slorado.h"

#include <torch/torch.h>

torch::Tensor decode_gpu_single(torch::Tensor scores, int num_chunks, const runner_t *runner);

std::vector<DecodedChunk> collect_gpu_decoded_chunks(torch::Tensor moves_sequence_qstring_cpu);

std::vector<DecodedChunk> decode_gpu(const torch::Tensor& scores, const int num_chunks, core_t *core, const int runner_idx);