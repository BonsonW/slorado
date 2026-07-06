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
#include "quant.h"
#include "sensitivity.h"

#ifdef USE_GPU
#include <c10/core/DeviceGuard.h>
#endif

typedef struct {
    core_t* core;
    db_t* db;
    int32_t runner;
    int32_t start;
    int32_t end;
} model_thread_arg_t;

static void accept_chunk(const int num_chunks, const basecall_chunk_t *chunk, runner_t *runner, int chunk_size) {
    ASSERT(chunk->read_dat->scaled_signal.size(0) > 0);
    torch::Tensor input_slice = (chunk->read_dat->scaled_signal).index({torch::indexing::Ellipsis, torch::indexing::Slice(chunk->input_offset, chunk->input_offset + chunk_size)});
    input_slice = input_slice.unsqueeze(0);
    auto slice_size = input_slice.size(1);
    ASSERT(slice_size != 0);

    // repeat-pad non-full chunks
    if (slice_size != chunk_size) {
        int64_t quot = chunk_size / slice_size;
        int64_t rem = chunk_size % slice_size;
        input_slice = torch::concat(
            {
                input_slice.repeat({1, quot}),
                input_slice.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, rem)})
            },
            1
        );
    }

    runner->input_tensor.index_put_({num_chunks, 0}, {input_slice});
}

// Inference half of a batch: run the model and return scores in [T, N, C] layout on the
// runner's device. Split out from decode so the streaming pipeline can overlap decode of
// batch i with inference of batch i+1 (see basecall_infer / basecall_decode).
static at::Tensor infer_chunks(
    const core_t* core,
    const std::vector<basecall_chunk_t *> &chunks,
    const int runner_idx
) {
    (void)chunks;
    runner_t* runner = (*core->runners)[runner_idx];
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];

#ifdef USE_GPU
    c10::DeviceGuard device_guard(runner->tensor_opts.device());
#endif
    torch::InferenceMode guard;

    LOG_DEBUG("%s", "basecalling chunks");
    auto input = runner->input_tensor.to(runner->tensor_opts.device());

    ts->time_infer -= realtime();
    at::Tensor fp16_scores;
    if (core->sensitivity_stats) {
        // Sensitivity mode: run fp16 baseline first, then quantized pass.
        g_quant_active = false;
        fp16_scores = model_forward(runner, input);
        g_quant_active = true;
    }
    auto scores = model_forward(runner, input);
    STAGE_SYNC(runner->device != "cpu", runner->device_idx);
    ts->time_infer += realtime();

    if (core->sensitivity_stats) {
        core->sensitivity_stats->accumulate(fp16_scores, scores);
    }

    auto scores_TNC = scores;
    // scores_TNC = scores_TNC.to(torch::kCPU).to(torch::kF32).transpose(0, 1).contiguous();
    scores_TNC = scores_TNC.transpose(0, 1).contiguous();
    STAGE_SYNC(runner->device != "cpu", runner->device_idx);
    return scores_TNC;
}

// Decode half of a batch: given scores in [T, N, C] layout, run the decoder and write
// moves/seq/qstring back into the chunks. Runs on its own thread in the streaming pipeline.
static void decode_chunks(
    const core_t* core,
    at::Tensor scores_TNC,
    const std::vector<basecall_chunk_t *> &chunks,
    const int runner_idx
) {
    runner_t* runner = (*core->runners)[runner_idx];
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];

#ifdef USE_GPU
    c10::DeviceGuard device_guard(runner->tensor_opts.device());
#endif
    torch::InferenceMode guard;

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
        openfish_decode_gpu(T, N, C, scores_TNC.data_ptr(), state_len, &core->decoder_opts, runner->gpubuf, &moves, &sequence, &qstring);
#else
        ERROR("Invalid device: %s. Please compile again for GPU", runner->device.c_str());
        exit(EXIT_FAILURE);
#endif
    }

    LOG_DEBUG("%s", "writing to chunks");

    for (size_t chunk = 0; chunk < chunks.size(); ++chunk) {
        size_t idx = chunk * T;
        chunks[chunk]->moves = std::vector<uint8_t>(moves + idx, moves + idx + T);
        size_t num_bases = 0;
        for (auto move: chunks[chunk]->moves) {
            num_bases += move;
        }
        if (num_bases > (size_t)T) {
            ERROR("num bases %zu greater than number of timesteps %d", num_bases, T);
            exit(EXIT_FAILURE);
        }
        chunks[chunk]->seq = std::string(sequence + idx, num_bases);
        chunks[chunk]->qstring = std::string(qstring + idx, num_bases);

        size_t seq_size = strlen(chunks[chunk]->seq.c_str());
        size_t qstr_size = strlen(chunks[chunk]->qstring.c_str());

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
            ERROR("seq: %s", chunks[chunk]->seq.c_str());
            ERROR("qstring: %s", chunks[chunk]->qstring.c_str());
            exit(EXIT_FAILURE);
        }
    }
    ts->time_decode += realtime();

    LOG_DEBUG("%s", "done writing to chunks");

    free(moves);
    free(sequence);
    free(qstring);
}

// Batch (non-streaming) path: inference + decode back-to-back.
static void call_chunks(
    const core_t* core,
    const std::vector<basecall_chunk_t *> &chunks,
    const int runner_idx
) {
    auto scores_TNC = infer_chunks(core, chunks, runner_idx);
    decode_chunks(core, scores_TNC, chunks, runner_idx);
}

// Streaming pipeline: accept a packed batch and run inference only, returning scores in
// [T, N, C] layout on the runner's device. The caller hands the scores to basecall_decode
// (on a separate thread) so decode overlaps the next batch's inference.
at::Tensor basecall_infer(
    const core_t* core,
    const int runner_idx,
    const std::vector<basecall_chunk_t *> &chunks
) {
    runner_t* runner = (*core->runners)[runner_idx];
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];
    auto chunk_size = core->chunk_size;

    ts->time_accept -= realtime();
    for (size_t i = 0; i < chunks.size(); ++i) {
        accept_chunk(i, chunks[i], runner, chunk_size);
    }
    ts->time_accept += realtime();

    ts->time_basecall -= realtime();
    auto scores_TNC = infer_chunks(core, chunks, runner_idx);
    ts->time_basecall += realtime();
    return scores_TNC;
}

// Streaming pipeline: decode a batch's scores and write results back into the chunks.
void basecall_decode(
    const core_t* core,
    const int runner_idx,
    at::Tensor scores_TNC,
    const std::vector<basecall_chunk_t *> &chunks
) {
    decode_chunks(core, scores_TNC, chunks, runner_idx);
}

void basecall_chunks(
    const core_t* core,
    const int runner_idx,
    const std::vector<basecall_chunk_t *> &chunks
) {
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];
    runner_t* runner = (*core->runners)[runner_idx];
    auto chunk_size = core->chunk_size;

    LOG_DEBUG("%s", "accepting chunks");
    ts->time_accept -= realtime();
    for (size_t i = 0; i < chunks.size(); ++i) {
        accept_chunk(i, chunks[i], runner, chunk_size);
    }
    ts->time_accept += realtime();
    LOG_DEBUG("%s", "done accepting chunks");

    ts->time_basecall -= realtime();
    call_chunks(core, chunks, runner_idx);
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

    std::vector<basecall_chunk_t *> chunks;

    for (size_t read_idx = start; read_idx < end; ++read_idx) {
        auto& db_chunks = (*db->basecall_chunks)[read_idx];

        for (size_t chunk_idx = 0; chunk_idx < db_chunks.size(); ++chunk_idx) {
            chunks.push_back(&db_chunks[chunk_idx]);

            if (chunks.size() == (size_t)opt.gpu_batch_size) {
                basecall_chunks(core, runner_idx, chunks);
                chunks.clear();
            }
        }
    }

    // leftover chunks
    if (chunks.size() > 0) {
        basecall_chunks(core, runner_idx, chunks);
    }

    pthread_exit(0);
}

void basecall_db(core_t* core, db_t* db) {
    int32_t n_reads = db->n_rec;
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
