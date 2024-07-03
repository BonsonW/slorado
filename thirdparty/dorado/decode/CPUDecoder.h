#pragma once

#include "Decoder.h"
#include "../nn/CRFModel.h"

#include <torch/torch.h>

class CPUDecoder final : Decoder {
public:
    std::vector<DecodedChunk> beam_search(const torch::Tensor& scores,
                                          int num_chunks,
                                          const DecoderOptions& options,
                                          std::string &device,
                                          const CRFModelConfig& config) final;
    constexpr static torch::ScalarType dtype = torch::kF32;
};

std::vector<DecodedChunk> beam_search_cpu(const torch::Tensor& scores,
                                          int num_chunks,
                                          const DecoderOptions& options,
                                          std::string &device,
                                          const CRFModelConfig& config);