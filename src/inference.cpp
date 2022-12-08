#include <c10/core/InferenceMode.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <slow5/slow5.h>
#include <torch/torch.h>

#include "inference.h"
#include "chunk.h"
#include "slorado.h"

void basecall_chunks(torch::Tensor &signal, std::vector<Chunk> &chunks, int chunk_size) {
    torch::InferenceMode guard;
    
    for (auto &chunk : chunks) {
        // Copy the chunk into the input tensor
        auto input_slice = signal.index({ torch::indexing::Slice(chunk.input_offset, chunk.input_offset + chunk_size) });
        size_t slice_size = input_slice.size(0);
    
        // Zero-pad any non-full chunks
        if (slice_size != chunk_size) {
            input_slice = torch::constant_pad_nd(input_slice, c10::IntArrayRef{ 0, int(chunk_size - slice_size) }, 0);
        }
        
        // pass to a model runner
    }
}
