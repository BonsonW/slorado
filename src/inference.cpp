#include <c10/core/InferenceMode.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <slow5/slow5.h>
#include <torch/torch.h>

#include "decode/CPUDecoder.h"
#include "Chunk.h"
#include "nn/ModelRunner.h"
#include "slorado.h"

std::vector<DecodedChunk> basecall_chunks(torch::Tensor &signal, std::vector<Chunk> &chunks, int chunk_size, ModelRunnerBase &model_runner) {
    int chunk_idx = 0;
    for (auto &chunk : chunks) {
        // Copy the chunk into the input tensor
        auto input_slice = signal.index({ torch::indexing::Slice(chunk.input_offset, chunk.input_offset + chunk_size) });
        size_t slice_size = input_slice.size(0);
    
        // Zero-pad any non-full chunks
        if (slice_size != chunk_size) {
            input_slice = torch::constant_pad_nd(input_slice, c10::IntArrayRef{ 0, int(chunk_size - slice_size) }, 0);
        }

        model_runner.accept_chunk(chunk_idx++, input_slice);
        fprintf(stdout, "model runner has accepted chunk: %d\n", chunk_idx);
    }
    
    fprintf(stdout, "base calling chunks...\n");
    return model_runner.call_chunks(chunks.size());
}
