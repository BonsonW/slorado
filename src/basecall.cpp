#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <slow5/slow5.h>
#include <torch/torch.h>

#include "decode/CPUDecoder.h"
#include "Chunk.h"
#include "nn/ModelRunner.h"
#include "slorado.h"
#include "misc.h"
#include "error.h"

void basecall_chunks(std::vector<torch::Tensor> &tensors, std::vector<Chunk *> &chunks, int chunk_size, int batch_size, ModelRunnerBase &model_runner, ModelRunnerBase &decoder, timestamps_t *ts) {
    for (int i = 0; i < tensors.size(); ++i) {
        ts->time_accept -= realtime();
        model_runner.accept_chunk(i, tensors[i]);
        ts->time_accept += realtime();
    }

    LOG_TRACE("%s", "basecalling chunks");
    ts->time_basecall -= realtime();
    torch::Tensor scores = model_runner.call_chunks();
    ts->time_basecall += realtime();

    LOG_TRACE("%s", "decoding chunks");
    ts->time_decode -= realtime();
    std::vector<DecodedChunk> decoded_chunks = decoder.decode_chunks(scores, chunks.size());
    ts->time_decode += realtime();

    for (int i = 0; i < chunks.size(); ++i) {
        chunks[i]->seq = decoded_chunks[i].sequence;
        chunks[i]->qstring = decoded_chunks[i].qstring;
        chunks[i]->moves = decoded_chunks[i].moves;
    }
}
