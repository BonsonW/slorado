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

void basecall_chunks(
    std::vector<torch::Tensor> tensors,
    std::vector<Chunk *> chunks,
    int chunk_size,
    int batch_size,
    ModelRunnerBase &model_runner,
    ModelRunnerBase &decoder,
    timestamps_t *ts
) {
    for (size_t i = 0; i < tensors.size(); ++i) {
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

    for (size_t i = 0; i < chunks.size(); ++i) {
        chunks[i]->seq = decoded_chunks[i].sequence;
        chunks[i]->qstring = decoded_chunks[i].qstring;
        chunks[i]->moves = decoded_chunks[i].moves;
    }
}

void basecall_thread(
    core_t* core,
    db_t* db,
    size_t runner_idx,
    size_t start,
    size_t end
) {
    opt_t opt = core->opt;
    timestamps_t *ts = (*core->runner_ts)[runner_idx];

    auto& model_runner = *((*core->runners)[runner_idx]);
    auto& decoder = *((*core->runners)[runner_idx]);
    
    std::vector<Chunk *> chunks;
    std::vector<torch::Tensor> tensors;

    for (size_t read_idx = start; read_idx < end; ++read_idx) {
        for (size_t chunk_idx = 0; chunk_idx < (*db->chunks)[read_idx].size(); ++chunk_idx) {
            chunks.push_back(((*db->chunks)[read_idx])[chunk_idx]);
            tensors.push_back((*db->tensors)[read_idx][chunk_idx]);

            if (chunks.size() == (size_t)opt.gpu_batch_size) {
                basecall_chunks(
                    tensors,
                    chunks,
                    opt.chunk_size,
                    opt.batch_size,
                    model_runner,
                    decoder,
                    ts
                );

                chunks.clear();
                tensors.clear();
            }
        }
    }

    if (chunks.size() > 0) {
        basecall_chunks(
            tensors,
            chunks,
            opt.chunk_size,
            opt.batch_size,
            model_runner,
            decoder,
            ts
        );
    }
}
