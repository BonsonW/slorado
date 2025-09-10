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

#include "torchbox.h"
#include "basecall.h"
#include "misc.h"
#include "error.h"

#ifdef HAVE_CUDA
#include <c10/cuda/CUDAGuard.h>
#endif

#ifdef HAVE_ROCM
#include <c10/hip/HIPGuard.h>
#endif

typedef struct {
    core_t* core;
    db_t* db;
    int32_t runner;
    int32_t start;
    int32_t end;
} model_thread_arg_t;

static void accept_chunk(const int num_chunks, const chunk_sig_t *chunk_sig, const core_t* core, const int runner_idx) {
    runner_t* runner = (*core->runners)[runner_idx];
    runner->input_tensor.index_put_({num_chunks, 0}, chunk_sig->tensor);
}

static void call_chunks(
    const core_t* core,
    const std::vector<chunk_res_t *> &results,
    const int runner_idx
) {
    torch::InferenceMode guard;
    runner_t* runner = (*core->runners)[runner_idx];
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];

    LOG_DEBUG("%s", "basecalling chunks");
    ts->time_infer -= realtime();
    auto scores = runner->module->forward(runner->input_tensor.to(runner->tensor_opts.device_opt().value()));
#ifdef USE_GPU
    if (runner->device != "cpu") torch::cuda::synchronize(runner->device_idx);
#endif
    ts->time_infer += realtime();

    auto scores_TNC = scores;
    // scores_TNC = scores_TNC.to(torch::kCPU).to(torch::kF32).transpose(0, 1).contiguous();
    scores_TNC = scores_TNC.transpose(0, 1).contiguous();
#ifdef USE_GPU
    if (runner->device != "cpu") torch::cuda::synchronize(runner->device_idx);
#endif

    const int T = scores_TNC.size(0);
    const int N = scores_TNC.size(1);
    const int C = scores_TNC.size(2);
    const int state_len = core->model_config->state_len;
    int nthreads = core->opt.num_thread / core->runners->size();

    uint8_t *moves;
    char *sequence;
    char *qstring;

    LOG_DEBUG("%s", "decoding scores");

    ts->time_decode -= realtime();
    if (runner->device == "cpu") {
        openfish_decode_cpu(T, N, C, nthreads, scores_TNC.data_ptr(), state_len, &core->decoder_opts, &moves, &sequence, &qstring);
    } else {
#ifdef USE_GPU
#ifdef HAVE_CUDA
    c10::cuda::CUDAGuard device_guard(runner->device_idx);
#endif
#ifdef HAVE_ROCM
    c10::hip::HIPGuard device_guard(runner->device_idx);
#endif
        openfish_decode_gpu(T, N, C, scores_TNC.data_ptr(), state_len, &core->decoder_opts, runner->gpubuf, &moves, &sequence, &qstring);
#else
        ERROR("Invalid device: %s. Please compile again for GPU", runner->device.c_str());
        exit(EXIT_FAILURE);
#endif
    }

    LOG_DEBUG("%s", "writing to chunks");

    for (size_t chunk = 0; chunk < results.size(); ++chunk) {
        size_t idx = chunk * T;
        results[chunk]->moves = std::vector<uint8_t>(moves + idx, moves + idx + T);
        size_t num_bases = 0;
        for (auto move: results[chunk]->moves) {
            num_bases += move;
        }
        if (num_bases > (size_t)T) {
            ERROR("num bases %zu greater than number of timesteps %d", num_bases, T);
            exit(EXIT_FAILURE);
        }
        results[chunk]->seq = std::string(sequence + idx, num_bases);
        results[chunk]->qstring = std::string(qstring + idx, num_bases);

        size_t seq_size = strlen(results[chunk]->seq.c_str());
        size_t qstr_size = strlen(results[chunk]->qstring.c_str());

        if (seq_size == 0) {
            ERROR("%s", "empty sequence returned by decoder");
            exit(EXIT_FAILURE);
        }

        if (qstr_size == 0) {
            ERROR("%s", "empty qstring returned by decoder");
            exit(EXIT_FAILURE);
        }
        
        if (seq_size != qstr_size) {
            ERROR("mismatch sequence size of %zu with qstring size of %zu", seq_size, qstr_size);
            ERROR("seq: %s", results[chunk]->seq.c_str());
            ERROR("qstring: %s", results[chunk]->qstring.c_str());
            exit(EXIT_FAILURE);
        }
    }
    ts->time_decode += realtime();

    free(moves);
    free(sequence);
    free(qstring);
}

static void basecall_chunks(
    const core_t* core,
    const int runner_idx,
    const std::vector<chunk_sig_t *> &signals,
    const std::vector<chunk_res_t *> &results
) {
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];
    for (size_t i = 0; i < signals.size(); ++i) {
        ts->time_accept -= realtime();
        accept_chunk(i, signals[i], core, runner_idx);
        ts->time_accept += realtime();
    }

    ts->time_basecall -= realtime();
    call_chunks(core, results, runner_idx);
    ts->time_basecall += realtime();
}

static void* pthread_single_basecall(void* voidargs) {
    model_thread_arg_t* args = (model_thread_arg_t*)voidargs;
    db_t* db = args->db;
    core_t* core = args->core;
    const size_t runner_idx = args->runner;
    const size_t start = args->start;
    const size_t end = args->end;
    opt_t opt = core->opt;

    std::vector<chunk_res_t *> results;
    std::vector<chunk_sig_t *> signals;

    for (size_t read_idx = start; read_idx < end; ++read_idx) {
        auto& chunks_res = (*db->chunk_db->chunks_res)[read_idx];
        auto& chunks_sig = (*db->chunk_db->chunks_sig)[read_idx];

        for (size_t chunk_idx = 0; chunk_idx < chunks_res.size(); ++chunk_idx) {
            results.push_back(&chunks_res[chunk_idx]);
            signals.push_back(&chunks_sig[chunk_idx]);

            if (results.size() == (size_t)opt.gpu_batch_size) {
                basecall_chunks(core, runner_idx, signals, results);
                results.clear();
                signals.clear();
            }
        }
    }

    // leftover chunks
    if (results.size() > 0) {
        basecall_chunks(core, runner_idx, signals, results);
    }

    pthread_exit(0);
}

void basecall_db(core_t* core, db_t* db) {
    int32_t n_reads = (*db->chunk_db->chunks_res).size();
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

void mod_basecall_db(core_t* core, db_t* db) {
    int32_t n_reads = (*db->chunk_db->chunks_res).size();
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
