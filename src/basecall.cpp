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
#include <stdlib.h>
#include <vector>
#include <slow5/slow5.h>
#include <openfish/openfish.h>

#include <cuda_runtime_api.h>

#include "basecall.h"
#include "error.h"

typedef struct {
    core_t* core;
    db_t* db;
    int32_t runner;
    int32_t start;
    int32_t end;
} model_thread_arg_t;

void accept_chunk(const int num_chunks, at::Tensor slice, const core_t* core, const int runner_idx) {
    runner_t* runner = (*core->runners)[runner_idx];
    runner->input_tensor.index_put_({num_chunks, 0}, slice);
}

void call_chunks(std::vector<DecodedChunk> &chunks, const int num_chunks, const core_t* core, const int runner_idx) {
    torch::InferenceMode guard;
    runner_t* runner = (*core->runners)[runner_idx];
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];

    LOG_DEBUG("%s", "basecalling chunks");
    ts->time_infer -= realtime();
    auto scores = runner->module->forward(runner->input_tensor.to(runner->tensor_opts.device_opt().value()));
#ifdef USE_GPU
    torch::cuda::synchronize(runner->device_idx);
#endif
    ts->time_infer += realtime();

    auto scores_TNC = scores;
    // scores_TNC = scores_TNC.to(torch::kCPU).to(torch::kF32).transpose(0, 1).contiguous();
    scores_TNC = scores_TNC.transpose(0, 1).contiguous();

    const int T = scores_TNC.size(0);
    const int N = scores_TNC.size(1);
    const int C = scores_TNC.size(2);
    const int state_len = runner->model_config.state_len;
    const int target_threads = core->opt.num_thread / core->runners->size();
    
    uint8_t *moves;
    char *sequence;
    char *qstring;

    LOG_DEBUG("%s", "decoding scores");
    decode(T, N, C, target_threads, scores_TNC.data_ptr(), state_len, &runner->decoder_opts, &moves, &sequence, &qstring);
    for (size_t chunk = 0; chunk < chunks.size(); ++chunk) {
        size_t idx = chunk * T;
        chunks[chunk] = {
            std::string(sequence + idx),
            std::string(qstring + idx),
            std::vector<uint8_t>(moves + idx, moves + idx + T),
        };
    }
    ts->time_decode_cleanup += realtime();

    free(moves);
    free(sequence);
    free(qstring);
}

void basecall_chunks(
    std::vector<torch::Tensor> tensors,
    std::vector<Chunk *> chunks,
    const int chunk_size,
    const core_t* core,
    const int runner_idx
) {
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];
    for (size_t i = 0; i < tensors.size(); ++i) {
        ts->time_accept -= realtime();
        accept_chunk(i, tensors[i], core, runner_idx);
        ts->time_accept += realtime();
    }

    std::vector<DecodedChunk> decoded_chunks(chunks.size());
    ts->time_decode -= realtime();
    call_chunks(decoded_chunks, chunks.size(), core, runner_idx);
    ts->time_decode += realtime();

    for (size_t i = 0; i < chunks.size(); ++i) {
        chunks[i]->seq = decoded_chunks[i].sequence;
        chunks[i]->qstring = decoded_chunks[i].qstring;
        chunks[i]->moves = decoded_chunks[i].moves;
    }
}

void* pthread_single_basecall(void* voidargs) {
    model_thread_arg_t* args = (model_thread_arg_t*)voidargs;
    db_t* db = args->db;
    core_t* core = args->core;
    const size_t runner_idx = args->runner;
    const size_t start = args->start;
    const size_t end = args->end;

    runner_t* runner = (*core->runners)[runner_idx];
    cudaSetDevice(runner->device_idx);

    opt_t opt = core->opt;

    std::vector<Chunk *> chunks;
    std::vector<torch::Tensor> tensors;

    for (size_t read_idx = start; read_idx < end; ++read_idx) {
        auto this_chunk = (*db->chunks)[read_idx];
        auto this_tensor = (*db->tensors)[read_idx];

        for (size_t chunk_idx = 0; chunk_idx < this_chunk.size(); ++chunk_idx) {
            chunks.push_back(this_chunk[chunk_idx]);
            tensors.push_back(this_tensor[chunk_idx]);

            if (chunks.size() == (size_t)opt.gpu_batch_size) {
                basecall_chunks(tensors, chunks, opt.chunk_size, core, runner_idx);
                chunks.clear();
                tensors.clear();
            }
        }
    }

    // leftover chunks
    if (chunks.size() > 0) {
        basecall_chunks(tensors, chunks, opt.chunk_size, core, runner_idx);
    }

    pthread_exit(0);
}

void basecall_db(core_t* core, db_t* db) {
    int32_t n_reads = (*db->chunks).size();
    int32_t num_threads = (*core->runners).size();
    int32_t step = (n_reads + num_threads - 1) / num_threads;

    // create threads
    pthread_t tids[num_threads];
    model_thread_arg_t pt_args[num_threads];
    int32_t t, ret;
    int32_t i = 0;
    // set the data structures
    for (t = 0; t < num_threads; t++) {
        pt_args[t].core = core;
        pt_args[t].db = db;
        pt_args[t].start = i;
        pt_args[t].runner = t;
        i += step;
        if (i > n_reads) {
            pt_args[t].end = n_reads;
        } else {
            pt_args[t].end = i;
        }
    }

    // create threads
    for (t = 0; t < num_threads; t++) {
        ret = pthread_create(&tids[t], NULL, pthread_single_basecall,
                                (void*)(&pt_args[t]));
        NEG_CHK(ret);
    }

    double time_sync = 0;

    // pthread joining
    for (t = 0; t < num_threads; t++) {
        int ret = pthread_join(tids[t], NULL);
        NEG_CHK(ret);
        if (t == 0) {
            time_sync -= realtime();
        }
        if (t == num_threads-1) {
            time_sync += realtime();
        }
    }

    core->time_sync += time_sync;
}
