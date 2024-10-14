#pragma once
#include "../nn/CRFModel.h"
#include <torch/torch.h>

#include <string>
#include <vector>

#define DTYPE_CPU torch::kF32
// #define DTYPE_GPU torch::kF16
#define DTYPE_GPU torch::kF16 // todo: this is temp, for testing full float openfish

struct DecodedChunk {
    std::string sequence;
    std::string qstring;
    std::vector<uint8_t> moves;
};
