/**
 * @file basecall.cpp
 * @brief runs DNA base calling steps
 * @author Bonson Wong (bonson.ym@gmail.com)

MIT License

Copyright (c) 2023 Bonson Wong (bonson.ym@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


******************************************************************************/

#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <slow5/slow5.h>

#include "basecall.h"
#include "error.h"

void basecall_chunks(
    std::vector<torch::Tensor> tensors,
    std::vector<Chunk *> chunks,
    int chunk_size,
    ModelRunnerBase &model_runner,
    timestamps_t *ts
) {
    for (size_t i = 0; i < tensors.size(); ++i) {
        ts->time_accept -= realtime();
        model_runner.accept_chunk(i, tensors[i]);
        ts->time_accept += realtime();
    }

    LOG_DEBUG("%s", "decoding chunks");
    ts->time_decode -= realtime();
    std::vector<DecodedChunk> decoded_chunks = model_runner.call_chunks(chunks.size());
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
                    model_runner,
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
            model_runner,
            ts
        );
    }
}
