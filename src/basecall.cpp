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

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdlib.h>
#include <vector>

#include "torchbox.h"
#include "basecall.h"
#include "misc.h"
#include "error.h"

#ifdef USE_GPU
#include <c10/core/DeviceGuard.h>
#if defined(HAVE_CUDA)
#include <cuda_runtime_api.h>
static inline void *gpu_malloc_managed(size_t bytes) { void *p = nullptr; if (cudaMallocManaged(&p, bytes) != cudaSuccess) return nullptr; return p; }
static inline void gpu_free_managed(void *p) { cudaFree(p); }
#elif defined(HAVE_ROCM)
#include <hip/hip_runtime.h>
static inline void *gpu_malloc_managed(size_t bytes) { void *p = nullptr; if (hipMallocManaged(&p, bytes) != hipSuccess) return nullptr; return p; }
static inline void gpu_free_managed(void *p) { hipFree(p); }
#endif

// Pool of managed/unified-memory buffers for the --cpu-beam int8 score handoff. Managed memory is
// written once by the GPU (the int8 quant cast) and then read by BOTH the GPU scan and the CPU beam
// with no device->host copy. Buffers are reused across batches (a returned buffer serves any request
// that fits) and freed with the pool. from_blob wraps a buffer as a torch tensor whose deleter
// returns it here, so a buffer stays checked out exactly as long as its decode_item is alive.
class ManagedScorePool {
public:
    ~ManagedScorePool() { for (auto &b : free_) gpu_free_managed(b.p); }

    // Wrap a managed buffer (>= bytes) as an [N,T,C] tensor of `dtype` on `dev`.
    at::Tensor get(c10::IntArrayRef sizes, at::ScalarType dtype, c10::Device dev, size_t bytes) {
        buf_t b{nullptr, 0};
        {
            std::lock_guard<std::mutex> lk(mtx_);
            for (size_t i = 0; i < free_.size(); ++i) {
                if (free_[i].cap >= bytes) { b = free_[i]; free_[i] = free_.back(); free_.pop_back(); break; }
            }
        }
        if (!b.p) {
            b.p = gpu_malloc_managed(bytes);
            if (!b.p) { ERROR("cpu-beam: failed to allocate %zu bytes of managed memory", bytes); exit(EXIT_FAILURE); }
            b.cap = bytes;
        }
        auto opts = torch::TensorOptions().dtype(dtype).device(dev);
        return torch::from_blob(b.p, sizes, [this, b](void *) { std::lock_guard<std::mutex> lk(mtx_); free_.push_back(b); }, opts);
    }

private:
    struct buf_t { void *p; size_t cap; };
    std::mutex mtx_;
    std::vector<buf_t> free_;
};

// Factories so the pipeline can own a pool without seeing the (torch + HIP/CUDA) definition.
ManagedScorePool *create_score_pool() { return new ManagedScorePool(); }
void destroy_score_pool(ManagedScorePool *pool) { delete pool; }
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

// Inference half of a batch: run the model and return scores in the model's native [N, T, C]
// layout on the runner's device. Split out from decode so decode can subtile independently of
// the forward batch size (the transpose to [T,N,C] and the decode buffer scale with the decode
// subtile, not the full N) -- letting the forward run at a large batch to fill the GPU.
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
    auto scores = model_forward(runner, input);
    STAGE_SYNC(runner->device != "cpu", runner->device_idx);
    ts->time_infer += realtime();

    STAGE_SYNC(runner->device != "cpu", runner->device_idx);
    return scores;   // [N, T, C]
}

// Copy a decoded openfish output block (moves/sequence/qstring, laid out [nt, T]) back into the
// chunks[n0 .. n0+nt). Shared by the fused decode path and the cpu-beam split.
static void write_chunk_outputs(
    const std::vector<basecall_chunk_t *> &chunks,
    int n0, int nt, int T,
    const uint8_t *moves, const char *sequence, const char *qstring
) {
    for (int j = 0; j < nt; ++j) {
        basecall_chunk_t *ck = chunks[n0 + j];
        size_t idx = (size_t)j * T;
        ck->moves = std::vector<uint8_t>(moves + idx, moves + idx + T);
        size_t num_bases = 0;
        for (auto move: ck->moves) {
            num_bases += move;
        }
        if (num_bases > (size_t)T) {
            ERROR("num bases %zu greater than number of timesteps %d", num_bases, T);
            exit(EXIT_FAILURE);
        }
        ck->seq = std::string(sequence + idx, num_bases);
        ck->qstring = std::string(qstring + idx, num_bases);

        size_t seq_size = strlen(ck->seq.c_str());
        size_t qstr_size = strlen(ck->qstring.c_str());

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
            ERROR("seq: %s", ck->seq.c_str());
            ERROR("qstring: %s", ck->qstring.c_str());
            exit(EXIT_FAILURE);
        }
    }
}

// Decode half of a batch: given scores in the model's native [N, T, C] layout, run the decoder
// and write moves/seq/qstring back into the chunks. Decodes in row-subtiles of runner->decode_tile
// so the [T,N,C] transpose scratch and the openfish decode buffer stay bounded (= decode_tile),
// independent of the (possibly large) forward batch N.
static void decode_chunks(
    const core_t* core,
    at::Tensor scores_NTC,
    const std::vector<basecall_chunk_t *> &chunks,
    const int runner_idx
) {
    runner_t* runner = (*core->runners)[runner_idx];
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];

#ifdef USE_GPU
    c10::DeviceGuard device_guard(runner->tensor_opts.device());
#endif
    torch::InferenceMode guard;

    // scores_NTC may have padding rows beyond the valid chunks (the forward runs on the full
    // batch-sized input tensor); decode only the valid reads.
    const int N = (int)chunks.size();
    const int T = scores_NTC.size(1);
    const int C = scores_NTC.size(2);
    const int state_len = core->model_config->state_len;
    int nthreads = core->opt.num_thread / core->runners->size();

    int tile = runner->decode_tile > 0 ? runner->decode_tile : N;
    if (tile > N || tile <= 0) tile = N;

    LOG_DEBUG("%s", "decoding scores");

    ts->time_decode -= realtime();
    for (int n0 = 0; n0 < N; n0 += tile) {
        const int nt = std::min(tile, N - n0);

        // openfish decodes natively from [N,T,C], so the row-slice [nt,T,C] is passed directly --
        // .contiguous() is a no-op when scores_NTC is contiguous (the common case), so no copy.
        // Subtiling still bounds the openfish decode buffer (gpubuf) to the tile, not the full N.
        auto sub_NTC = scores_NTC.narrow(0, n0, nt).contiguous();

        uint8_t *moves;
        char *sequence;
        char *qstring;
        // Score dtype is carried by the tensor: int8 CRF output (round(tanh*127) / clamp±5·127/5)
        // dequants by 5/127; fp16 output uses 1.0. Runtime dispatch works for any model/mode.
        const bool i8 = sub_NTC.scalar_type() == at::kChar;
        const openfish_score_dtype_t sdt = i8 ? OPENFISH_SCORE_I8 : OPENFISH_SCORE_F16;
        const float sscale = i8 ? SCORES_I8_SCALE : 1.0f;
        if (runner->device == "cpu") {
            openfish_decode_cpu(T, nt, C, nthreads, sub_NTC.data_ptr(), sdt, sscale, state_len, &core->decoder_opts, &moves, &sequence, &qstring);
        } else {
#ifdef USE_GPU
            openfish_decode_gpu(T, nt, C, sub_NTC.data_ptr(), sdt, sscale, state_len, &core->decoder_opts, runner->gpubuf, &moves, &sequence, &qstring);
#else
            ERROR("Invalid device: %s. Please compile again for GPU", runner->device.c_str());
            exit(EXIT_FAILURE);
#endif
        }

        write_chunk_outputs(chunks, n0, nt, T, moves, sequence, qstring);

        free(moves);
        free(sequence);
        free(qstring);
    }
    ts->time_decode += realtime();

    LOG_DEBUG("%s", "done writing to chunks");
}

// Inference + decode back-to-back on a packed batch (decode subtiles internally).
static void call_chunks(
    const core_t* core,
    const std::vector<basecall_chunk_t *> &chunks,
    const int runner_idx
) {
    auto scores_NTC = infer_chunks(core, chunks, runner_idx);
    decode_chunks(core, scores_NTC, chunks, runner_idx);
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

// ---------------------------------------------------------------------------
// cpu-beam decode split (--cpu-beam): the inference half runs on the runner thread and produces the
// score tensor; the scan (GPU) + beam (CPU) decode halves run on a separate decode thread so decode
// of batch k overlaps inference of batch k+1. Only meaningful on a unified-memory GPU where the CPU
// beam reads the GPU-written posteriors (openfish_gpubuf_init_hostvis) with no copy.
// ---------------------------------------------------------------------------

// Runner-thread half: accept chunks, run the model, and return the valid [N, T, C] score rows on the
// runner device. Clamp models (plain LSTM) emit fp16 CRF scores bounded to ±5, so they are quantized
// to int8 (×127/5 → [-127,127], decoder rescales by SCORES_I8_SCALE); FLSTM already emits int8 when
// --quant sets g_scores_i8; TX (no clamp) stays fp16. Runtime dtype dispatch in the scan/beam handles
// all three.
//
// When `pool` is given (the --cpu-beam path) and the scores are int8, the int8 cast is written
// DIRECTLY into a managed/unified buffer: on a unified-memory iGPU that one buffer is then read by
// both the GPU scan and the CPU beam with no device->host copy (the copy that otherwise dominated the
// runner's per-batch cost -- see the cpu-beam profile). fp16 scores can't share a buffer (the GPU scan
// needs fp16, the CPU beam needs fp32), so they take the plain path and the caller makes the host copy.
at::Tensor basecall_infer_scores(
    const core_t* core,
    const int runner_idx,
    const std::vector<basecall_chunk_t *> &chunks,
    ManagedScorePool* pool
) {
    runner_t* runner = (*core->runners)[runner_idx];
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];
    auto chunk_size = core->chunk_size;

    ts->time_accept -= realtime();
    for (size_t i = 0; i < chunks.size(); ++i) {
        accept_chunk(i, chunks[i], runner, chunk_size);
    }
    ts->time_accept += realtime();

    // infer_chunks runs the forward on the full (batch-sized) input tensor; keep only the valid rows.
    auto scores = infer_chunks(core, chunks, runner_idx).narrow(0, 0, (int64_t)chunks.size());

    if (core->model_config->clamp && scores.is_floating_point()) {
        auto q = scores.mul(127.0f / 5.0f).round().clamp_(-127.0f, 127.0f);  // device float
#ifdef USE_GPU
        if (pool) {
            const int64_t N = q.size(0), T = q.size(1), C = q.size(2);
            at::Tensor managed = pool->get({N, T, C}, at::kChar, scores.device(), (size_t)(N * T * C));
            managed.copy_(q);   // single float->int8 cast writing straight into managed memory
            return managed;
        }
#endif
        (void)pool;
        return q.to(torch::kChar);
    }
    return scores.contiguous();
}

#ifdef USE_GPU
// Decode-thread half 1: GPU forward/backward posterior scan into gpubuf (bwd_NTC/post_NTC). No beam,
// no torch ops -- raw openfish device call so it is safe to run off the runner's torch stream.
void basecall_scan_gpu(
    const core_t* core,
    openfish_gpubuf_t *gpubuf,
    const at::Tensor &dev_scores
) {
    const int N = (int)dev_scores.size(0);
    const int T = (int)dev_scores.size(1);
    const int C = (int)dev_scores.size(2);
    const int state_len = core->model_config->state_len;
    const bool i8 = dev_scores.scalar_type() == at::kChar;
    const openfish_score_dtype_t sdt = i8 ? OPENFISH_SCORE_I8 : OPENFISH_SCORE_F16;
    const float sscale = i8 ? SCORES_I8_SCALE : 1.0f;
    openfish_decode_gpu_scan(T, N, C, dev_scores.data_ptr(), sdt, sscale, state_len, &core->decoder_opts, gpubuf);
}

// Decode-thread half 2: CPU beam search + qualities over the GPU-produced posteriors in gpubuf, then
// write moves/seq/qstring back into the chunks. host_scores is the host-side copy of the (int8/fp16)
// score tensor the beam reads sparsely.
void basecall_beam_cpu(
    const core_t* core,
    openfish_gpubuf_t *gpubuf,
    const at::Tensor &host_scores,
    const std::vector<basecall_chunk_t *> &chunks
) {
    const int N = (int)chunks.size();
    const int T = (int)host_scores.size(1);
    const int C = (int)host_scores.size(2);
    const int state_len = core->model_config->state_len;
    const int nthreads = core->opt.num_thread;
    const bool i8 = host_scores.scalar_type() == at::kChar;
    const openfish_score_dtype_t sdt = i8 ? OPENFISH_SCORE_I8 : OPENFISH_SCORE_F16;
    const float sscale = i8 ? SCORES_I8_SCALE : 1.0f;

    uint8_t *moves;
    char *sequence;
    char *qstring;
    openfish_decode_cpu_beam(T, N, C, nthreads, host_scores.data_ptr(), sdt, sscale, state_len,
                             &core->decoder_opts, gpubuf, &moves, &sequence, &qstring);
    write_chunk_outputs(chunks, 0, N, T, moves, sequence, qstring);
    free(moves);
    free(sequence);
    free(qstring);
}
#endif

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
