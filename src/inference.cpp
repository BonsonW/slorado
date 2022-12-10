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

void basecall_chunks(torch::Tensor &signal, std::vector<Chunk> &chunks, int chunk_size, int batch_size, ModelRunnerBase &model_runner) {
    int chunk_idx = 0;
    int n_batched_chunks = 0;
    int cur_batch = 0;
    
    while (chunk_idx < chunks.size()) {
        int batch_offset = cur_batch * batch_size;
        
        for (; chunk_idx < chunks.size() && n_batched_chunks < batch_size; ++chunk_idx, ++n_batched_chunks) {
        
            // Copy the chunk into the input tensor
            auto input_slice = signal.index({ torch::indexing::Slice(chunks[chunk_idx].input_offset, chunks[chunk_idx].input_offset + chunk_size) });
            size_t slice_size = input_slice.size(0);
        
            // Zero-pad any non-full chunks
            if (slice_size != chunk_size) {
                input_slice = torch::constant_pad_nd(input_slice, c10::IntArrayRef{ 0, int(chunk_size - slice_size) }, 0);
            }
    
            model_runner.accept_chunk(chunk_idx - batch_offset, input_slice);
        }
        
        fprintf(stdout, "base calling chunks in batch: %d\n", cur_batch);
        std::vector<DecodedChunk> decoded_chunks = model_runner.call_chunks(chunk_idx);
        
        for (int i = 0; i < n_batched_chunks; ++i) {
            chunks[batch_offset + i].seq = decoded_chunks[i].sequence;
            chunks[batch_offset + i].qstring = decoded_chunks[i].qstring;
            chunks[batch_offset + i].moves = decoded_chunks[i].moves;
        }
        
        n_batched_chunks = 0;
        ++cur_batch;
    }
    
}
