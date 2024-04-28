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

typedef struct {
    core_t* core;
    db_t* db;
    int32_t runner;
    int32_t start;
    int32_t end;
} model_thread_arg_t;

void basecall_chunks(
    std::vector<torch::Tensor> tensors,
    std::vector<Chunk *> chunks,
    int chunk_size,
    ModelRunnerBase &model_runner,
    timestamps_t *ts
);

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

void* pthread_single_basecall(void* voidargs) {

    model_thread_arg_t* args = (model_thread_arg_t*)voidargs;
    db_t* db = args->db;
    core_t* core = args->core;
    size_t runner_idx = args->runner;
    size_t start = args->start;
    size_t end = args->end;

    opt_t opt = core->opt;
    timestamps_t *ts = (*core->runner_ts)[runner_idx];

    auto& model_runner = *((*core->runners)[runner_idx]);

    std::vector<Chunk *> chunks;
    std::vector<torch::Tensor> tensors;

    for (size_t read_idx = start; read_idx < end; ++read_idx) {

        auto this_chunk = (*db->chunks)[read_idx];
        auto this_tensor = (*db->tensors)[read_idx];

        for (size_t chunk_idx = 0; chunk_idx < this_chunk.size(); ++chunk_idx) {
            chunks.push_back(this_chunk[chunk_idx]);
            tensors.push_back(this_tensor[chunk_idx]);

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

    pthread_exit(0);

}

void basecall_db(core_t* core, db_t* db) {

    timestamps_t *ts = &(core->ts);

    int32_t n_reads = (*db->chunks).size();
    int32_t num_threads = (*core->runners).size(); //isnt this possible to be taken from opt?
    int32_t step = (n_reads + num_threads - 1) / num_threads;

    //create threads
    pthread_t tids[num_threads];
    model_thread_arg_t pt_args[num_threads];
    int32_t t, ret;
    int32_t i = 0;
    //set the data structures
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

    //create threads
    for(t = 0; t < num_threads; t++){
        ret = pthread_create(&tids[t], NULL, pthread_single_basecall,
                                (void*)(&pt_args[t]));
        NEG_CHK(ret);
    }

    double time_sync = 0;

    //pthread joining
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

    ts->time_sync += time_sync;

}


// void basecall_cpu_db(core_t* core, db_t* db) {

//     int32_t n_reads = (*db->chunks).size();
//     timestamps_t *ts = &(core->ts);
//     auto& model_runner = *((*core->runners)[0]);

//     for (size_t read_idx = 0; read_idx < n_reads; ++read_idx) {

//         auto this_chunk = (*db->chunks)[read_idx];
//         auto this_tensor = (*db->tensors)[read_idx];

//         for (size_t chunk_idx = 0; chunk_idx < this_chunk.size(); ++chunk_idx) {

//             basecall_chunks(
//                     this_tensor,
//                     this_chunk,
//                     core->opt.chunk_size,
//                     model_runner,
//                     ts
//             );

//         }
//     }

// }