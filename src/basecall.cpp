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

#if defined(HAVE_METAL)
#include "metal_utils.h"
#endif

typedef struct {
    core_t* core;
    db_t* db;
    int32_t runner;
    int32_t start;
    int32_t end;
} model_thread_arg_t;

#if defined(HAVE_METAL)
// Batch/no-stream decode backend, selected per RUN (so CPU and GPU beam are never mixed in one run):
//   default          -> full GPU decode (openfish_decode_gpu): scan + beam + quality + seq on GPU
//   SLORADO_CPU_BEAM -> GPU forward/backward scan, then CPU beam search (openfish_decode_cpu_beam):
//                       beam + quality + seq on the CPU over the GPU-scanned posteriors.
// The per-stage times (backward/beam/forward+posterior/quality/seq-gen) are gated by
// OPENFISH_DECODE_PROF inside openfish and printed nested under the "- decode:" line (basecaller_main).
static const bool g_cpu_beam = getenv("SLORADO_CPU_BEAM") != nullptr;
#endif

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
    ts->time_todev -= realtime();
    auto input = runner->input_tensor.to(runner->tensor_opts.device());
    ts->time_todev += realtime();

    ts->time_forward -= realtime();
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
    ts->time_forward += realtime();

    if (core->sensitivity_stats) {
        core->sensitivity_stats->accumulate(fp16_scores, scores);
    }

    STAGE_SYNC(runner->device != "cpu", runner->device_idx);
    return scores;   // [N, T, C]
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
        const openfish_score_dtype_t sdt = i8 ? OPENFISH_SCORE_I8
                                          : (sub_NTC.scalar_type() == at::kHalf ? OPENFISH_SCORE_F16
                                                                               : OPENFISH_SCORE_F32);
        const float sscale = i8 ? SCORES_I8_SCALE : 1.0f;
        if (runner->device == "cpu") {
            openfish_decode_cpu(T, nt, C, nthreads, sub_NTC.data_ptr(), sdt, sscale, state_len, &core->decoder_opts, &moves, &sequence, &qstring);
        } else {
#if defined(HAVE_METAL)
            // In-place Metal decode: an MPS tensor's storage().data() bit-casts to its MTLBuffer (cf.
            // ATen getMTLBufferStorage); openfish bridges the same pointer back, so we hand it the
            // scores buffer directly -- no host copy, no scratch MTLBuffer. Needs storage offset 0
            // (clone the rare offset case) and an MPS-stream sync first, since openfish decodes on its
            // own command queue.
            auto sc = sub_NTC.contiguous();
            if (sc.storage_offset() != 0) sc = sc.clone();
            torch::mps::synchronize();
            if (g_cpu_beam) {
                // Quantize the scores to int8 on the GPU before the scan, exactly as the streaming
                // path does in basecall_infer_host: int8 reads identically on the GPU (scan) and the
                // CPU (beam), so the host copy is 1 byte/element with no widening pass -- 4x less DMA
                // than fp32 and no conversion. Only for clamp models (scores bounded to [-5,5]).
                openfish_score_dtype_t use_sdt = sdt;
                float use_scale = sscale;
                if (core->model_config->clamp && sdt != OPENFISH_SCORE_I8) {
                    sc = (sc * (127.0f / 5.0f)).round().clamp_(-127.0f, 127.0f).to(torch::kChar).contiguous();
                    use_sdt = OPENFISH_SCORE_I8;
                    use_scale = SCORES_I8_SCALE;
                }
                // GPU forward/backward scan, then beam+quality+seq on the CPU over the shared posteriors.
                openfish_decode_gpu_scan(T, nt, C, sc.storage().data(), use_sdt, use_scale, state_len, &core->decoder_opts, runner->gpubuf);
                torch::mps::synchronize();
                // The scores must be copied to host: torch's MPS allocator hands out
                // MTLStorageModePrivate buffers, so [buf contents] is NULL and the CPU cannot read
                // the tensor in place -- unified memory notwithstanding. (gpubuf's bwd_NTC/post_NTC
                // ARE openfish-allocated shared buffers, so the posteriors are already zero-copy;
                // that is the part the CUDA/HIP zero-copy note refers to.) fp16 is widened to fp32
                // because the CPU beam's fp16 read path is unverified -- see basecall_finalize_host
                // in the stream path, which does the same for the same reason.
                at::Tensor host_copy = (use_sdt == OPENFISH_SCORE_I8)
                    ? sc.to(torch::kCPU).contiguous()
                    : sc.to(torch::kCPU).to(torch::kFloat32).contiguous();
                const openfish_score_dtype_t cpu_sdt =
                    (use_sdt == OPENFISH_SCORE_I8) ? OPENFISH_SCORE_I8 : OPENFISH_SCORE_F32;
                openfish_decode_cpu_beam(T, nt, C, nthreads, host_copy.data_ptr(), cpu_sdt, use_scale, state_len, &core->decoder_opts, runner->gpubuf, &moves, &sequence, &qstring);
            } else {
                // Full GPU decode: scan + beam + quality + seq all on the GPU.
                openfish_decode_gpu(T, nt, C, sc.storage().data(), sdt, sscale, state_len, &core->decoder_opts, runner->gpubuf, &moves, &sequence, &qstring);
            }
#elif defined(USE_GPU)
            openfish_decode_gpu(T, nt, C, sub_NTC.data_ptr(), sdt, sscale, state_len, &core->decoder_opts, runner->gpubuf, &moves, &sequence, &qstring);
#else
            ERROR("Invalid device: %s. Please compile again for GPU", runner->device.c_str());
            exit(EXIT_FAILURE);
#endif
        }

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

        free(moves);
        free(sequence);
        free(qstring);
    }
    ts->time_decode += realtime();

    LOG_DEBUG("%s", "done writing to chunks");
}

// --- Metal overlap: split inference (GPU) and decode (CPU) into separate pipeline stages ---------

// Accept + infer, then move scores to host so the GPU is freed for the next batch. When the model
// clamps scores to +/-5 we quantize to int8 on the GPU first (halves the copy and the resident
// scores footprint; openfish's CPU decode dequantizes via score_scale). Returns host [nchunks,T,C].
at::Tensor basecall_infer_host(const core_t* core, const int runner_idx, const std::vector<basecall_chunk_t *> &chunks) {
    runner_t* runner = (*core->runners)[runner_idx];
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];
    auto chunk_size = core->chunk_size;

    ts->time_accept -= realtime();
    for (size_t i = 0; i < chunks.size(); ++i) accept_chunk(i, chunks[i], runner, chunk_size);
    ts->time_accept += realtime();

    at::Tensor scores = infer_chunks(core, chunks, runner_idx);          // [batch, T, C] on device
    auto sub = scores.narrow(0, 0, (int64_t)chunks.size());              // valid rows only
    // Quantize on the GPU but DO NOT copy to host here: hand the decode stage the DEVICE tensor and
    // let it do the .to(CPU). That takes the CRF+quant GPU work and the score DMA off the infer
    // thread's critical path, so the GPU can start the next batch's forward while the previous
    // batch's scores are still draining to host / being decoded.
    ts->time_scores_copy -= realtime();
    at::Tensor dev;
    if (core->model_config->clamp) {
        // scores in [-5, 5] -> int8 [-127, 127] (decode rescales by SCORES_I8_SCALE = 5/127).
        dev = (sub * (127.0f / 5.0f)).round().clamp_(-127.0f, 127.0f).to(torch::kChar).contiguous();
    } else {
        // Native (non-clamp, e.g. transformer) scores: the GPU scan reads these as fp16 (openfish
        // Metal F16 == __fp16), so keep the DEVICE copy fp16. basecall_finalize_host converts the
        // HOST copy to fp32 for the CPU beam (openfish CPU F16 == float). Handing fp32 to the GPU
        // scan misreads the bytes as fp16 -> garbage (only bites clamp=false models; LSTM int8).
        dev = sub.to(torch::kHalf).contiguous();
    }
    ts->time_scores_copy += realtime();
    ts->n_batches++;
    return dev;
}

#if defined(HAVE_METAL)
// Split infer (streaming double-buffer): accept + conv+LSTM only -> device LSTM output [N,T,lstm].
// The CRF+quant is deferred to basecall_crf_quant so it can be issued AFTER the previous batch is
// finalized and thus overlap the NEXT batch's conv+LSTM (hiding it off the runner's critical path).
at::Tensor basecall_infer_lstm_host(const core_t* core, const int runner_idx, const std::vector<basecall_chunk_t *> &chunks) {
    runner_t* runner = (*core->runners)[runner_idx];
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];
    auto chunk_size = core->chunk_size;
#ifdef USE_GPU
    c10::DeviceGuard device_guard(runner->tensor_opts.device());
#endif
    torch::InferenceMode guard;

    ts->time_accept -= realtime();
    for (size_t i = 0; i < chunks.size(); ++i) accept_chunk(i, chunks[i], runner, chunk_size);
    ts->time_accept += realtime();

    ts->time_todev -= realtime();
    auto input = runner->input_tensor.to(runner->tensor_opts.device());
    ts->time_todev += realtime();

    ts->time_forward -= realtime();
    at::Tensor lstm_out = model_forward_nocrf(runner, input);           // conv+LSTM (blocks); [batch,T,lstm]
    ts->time_forward += realtime();
    ts->n_batches++;
    return lstm_out.narrow(0, 0, (int64_t)chunks.size());              // valid rows only
}

// CRF linear + clamp + int8 quant, enqueued on the MPS stream but NOT synced. Returns device scores
// (int8 when the model clamps, else fp32). The caller finalizes/copies later (double-buffered).
at::Tensor basecall_crf_quant(const core_t* core, const int runner_idx, at::Tensor lstm_out) {
    runner_t* runner = (*core->runners)[runner_idx];
#ifdef USE_GPU
    c10::DeviceGuard device_guard(runner->tensor_opts.device());
#endif
    torch::InferenceMode guard;
    at::Tensor x = model_crf(runner, lstm_out);                        // linear + clamp -> [N,T,C]
    if (core->model_config->clamp) {
        return (x * (127.0f / 5.0f)).round().clamp_(-127.0f, 127.0f).to(torch::kChar).contiguous();
    }
    // fp16 on-device for the GPU scan (see basecall_infer_host); finalize converts the host copy to fp32.
    return x.to(torch::kHalf).contiguous();
}
#endif

// CPU-decode host scores ([nchunks,T,C], int8 or fp32) and write moves/seq/qstring into the chunks.
// Runs in a separate stage so it overlaps the next batch's GPU inference.
void basecall_decode_host(const core_t* core, const int runner_idx, at::Tensor dev_scores, const std::vector<basecall_chunk_t *> &chunks) {
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];
    // Copy scores device->host here (off the infer thread) so the DMA overlaps the next forward.
    ts->time_host_copy -= realtime();
    at::Tensor host_scores = dev_scores.device().is_cpu() ? dev_scores : dev_scores.to(torch::kCPU).contiguous();
    ts->time_host_copy += realtime();
    const int N = (int)chunks.size();
    const int T = (int)host_scores.size(1);
    const int C = (int)host_scores.size(2);
    const int state_len = core->model_config->state_len;
    int nthreads = core->opt.num_thread;
    if (const char *e = getenv("SLORADO_DECODE_THREADS")) { int v = atoi(e); if (v > 0) nthreads = v; }

    const bool i8 = host_scores.scalar_type() == at::kChar;
    const openfish_score_dtype_t sdt = i8 ? OPENFISH_SCORE_I8 : OPENFISH_SCORE_F16;
    const float sscale = i8 ? SCORES_I8_SCALE : 1.0f;

    uint8_t *moves; char *sequence; char *qstring;
    ts->time_decode -= realtime();
    openfish_decode_cpu(T, N, C, nthreads, host_scores.data_ptr(), sdt, sscale, state_len, &core->decoder_opts, &moves, &sequence, &qstring);
    for (int j = 0; j < N; ++j) {
        basecall_chunk_t *ck = chunks[j];
        size_t idx = (size_t)j * T;
        ck->moves = std::vector<uint8_t>(moves + idx, moves + idx + T);
        size_t num_bases = 0;
        for (auto move : ck->moves) num_bases += move;
        if (num_bases > (size_t)T) { ERROR("num bases %zu greater than number of timesteps %d", num_bases, T); exit(EXIT_FAILURE); }
        ck->seq = std::string(sequence + idx, num_bases);
        ck->qstring = std::string(qstring + idx, num_bases);
        if (ck->seq.empty() || ck->qstring.empty() || ck->seq.size() != ck->qstring.size()) {
            ERROR("decode produced bad seq/qstring (seq %zu qstr %zu)", ck->seq.size(), ck->qstring.size());
            exit(EXIT_FAILURE);
        }
    }
    free(moves); free(sequence); free(qstring);
    ts->time_decode += realtime();
}

#if defined(HAVE_METAL)
// GPU forward/backward scan for a batch: run openfish's scan on the device int8/fp16 scores into
// gpubuf (bwd_NTC/post_NTC, unified-memory shared buffers), then return a host copy of the scores
// for the CPU beam. This is the "scan on GPU" half -- it fills the otherwise-idle GPU (~0.06s/batch)
// and leaves only the light beam search for the CPU.
void basecall_scan_gpu(const core_t* core, const int runner_idx, at::Tensor dev_scores,
                       openfish_gpubuf_t *gpubuf) {
    (void)runner_idx;
    const int N = (int)dev_scores.size(0);
    const int T = (int)dev_scores.size(1);
    const int C = (int)dev_scores.size(2);
    const int state_len = core->model_config->state_len;
    const bool i8 = dev_scores.scalar_type() == at::kChar;
    const openfish_score_dtype_t sdt = i8 ? OPENFISH_SCORE_I8 : OPENFISH_SCORE_F16;
    const float sscale = i8 ? SCORES_I8_SCALE : 1.0f;

    // Runs on the decode thread with NO torch ops (torch MPS isn't safe to touch from two threads --
    // that crashes). The runner already synced the CRF+quant that wrote these scores (double-buffered,
    // so cheaply), so the buffer is stable; openfish's scan reads it on its own command queue.
    auto sc = dev_scores.contiguous();
    if (sc.storage_offset() != 0) sc = sc.clone();
    openfish_decode_gpu_scan(T, N, C, sc.storage().data(), sdt, sscale, state_len, &core->decoder_opts, gpubuf);
}

// Runner thread: finalize a batch's enqueued CRF+quant and copy its int8 scores to host for the CPU
// beam. torch::mps::synchronize() must run on the runner (the only torch-MPS thread). Double-buffered
// by the caller so the CRF+quant already ran during the NEXT batch's conv+LSTM -> the sync is cheap
// and the copy is just the (small, int8) DMA. Returns host [N,T,C].
at::Tensor basecall_finalize_host(const core_t* core, at::Tensor dev_scores) {
    (void)core;
    torch::mps::synchronize();
    at::Tensor host = dev_scores.contiguous().to(torch::kCPU).contiguous();
    // int8 scores read identically on GPU (scan) and CPU (beam). Native fp16 scores, however, must
    // become fp32 for the CPU beam (openfish CPU F16 == float) while the device copy stays fp16 for
    // the GPU scan -- otherwise the beam reads 2-byte halves as 4-byte floats and mis-decodes.
    if (host.scalar_type() != at::kChar) host = host.to(torch::kFloat32).contiguous();
    return host;
}

// CPU beam search over the GPU-scanned posteriors in gpubuf; writes moves/seq/qstring to the chunks.
void basecall_beam_host(const core_t* core, const int runner_idx, at::Tensor host_scores,
                        openfish_gpubuf_t *gpubuf, const std::vector<basecall_chunk_t *> &chunks) {
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];
    const int N = (int)chunks.size();
    const int T = (int)host_scores.size(1);
    const int C = (int)host_scores.size(2);
    const int state_len = core->model_config->state_len;
    int nthreads = core->opt.num_thread;
    if (const char *e = getenv("SLORADO_DECODE_THREADS")) { int v = atoi(e); if (v > 0) nthreads = v; }
    const bool i8 = host_scores.scalar_type() == at::kChar;
    const openfish_score_dtype_t sdt = i8 ? OPENFISH_SCORE_I8 : OPENFISH_SCORE_F16;
    const float sscale = i8 ? SCORES_I8_SCALE : 1.0f;

    uint8_t *moves; char *sequence; char *qstring;
    ts->time_decode -= realtime();
    openfish_decode_cpu_beam(T, N, C, nthreads, host_scores.data_ptr(), sdt, sscale, state_len,
                             &core->decoder_opts, gpubuf, &moves, &sequence, &qstring);
    for (int j = 0; j < N; ++j) {
        basecall_chunk_t *ck = chunks[j];
        size_t idx = (size_t)j * T;
        ck->moves = std::vector<uint8_t>(moves + idx, moves + idx + T);
        size_t num_bases = 0;
        for (auto move : ck->moves) num_bases += move;
        if (num_bases > (size_t)T) { ERROR("num bases %zu greater than number of timesteps %d", num_bases, T); exit(EXIT_FAILURE); }
        ck->seq = std::string(sequence + idx, num_bases);
        ck->qstring = std::string(qstring + idx, num_bases);
        if (ck->seq.empty() || ck->qstring.empty() || ck->seq.size() != ck->qstring.size()) {
            ERROR("beam produced bad seq/qstring (seq %zu qstr %zu)", ck->seq.size(), ck->qstring.size());
            exit(EXIT_FAILURE);
        }
    }
    free(moves); free(sequence); free(qstring);
    ts->time_decode += realtime();
}

// TEMPORARY (benchmark): full GPU decode -- scan + beam + qual + sequence-gen ALL on the GPU (the
// batch path's openfish_decode_gpu), as a drop-in alternative to basecall_scan_gpu +
// basecall_beam_host (GPU scan + CPU beam) in the streaming decode stage. Lets us A/B the beam
// search on GPU vs CPU with the pipeline, GPU scan and score dtypes held identical. Selected via
// SLORADO_GPU_BEAM. Like basecall_scan_gpu, runs on the decode thread and touches NO torch MPS ops:
// dev_scores is already contiguous and was synced by the runner's basecall_finalize_host, so only
// the raw MTLBuffer pointer (storage().data()) is handed to openfish.
void basecall_decode_gpu_full(const core_t* core, const int runner_idx, at::Tensor dev_scores,
                              openfish_gpubuf_t *gpubuf, const std::vector<basecall_chunk_t *> &chunks) {
    runner_stat_t* ts = (*core->runner_stats)[runner_idx];
    const int N = (int)chunks.size();
    const int T = (int)dev_scores.size(1);
    const int C = (int)dev_scores.size(2);
    const int state_len = core->model_config->state_len;
    const bool i8 = dev_scores.scalar_type() == at::kChar;
    const openfish_score_dtype_t sdt = i8 ? OPENFISH_SCORE_I8 : OPENFISH_SCORE_F16;
    const float sscale = i8 ? SCORES_I8_SCALE : 1.0f;

    // Mirror basecall_scan_gpu's buffer handling (contiguous no-op; clone only for the rare non-zero
    // storage offset) so the ONLY behavioural difference vs the GPU-scan+CPU-beam path is the beam.
    auto sc = dev_scores.contiguous();
    if (sc.storage_offset() != 0) sc = sc.clone();

    uint8_t *moves; char *sequence; char *qstring;
    ts->time_decode -= realtime();
    openfish_decode_gpu(T, N, C, sc.storage().data(), sdt, sscale, state_len, &core->decoder_opts,
                        gpubuf, &moves, &sequence, &qstring);
    for (int j = 0; j < N; ++j) {
        basecall_chunk_t *ck = chunks[j];
        size_t idx = (size_t)j * T;
        ck->moves = std::vector<uint8_t>(moves + idx, moves + idx + T);
        size_t num_bases = 0;
        for (auto move : ck->moves) num_bases += move;
        if (num_bases > (size_t)T) { ERROR("num bases %zu greater than number of timesteps %d", num_bases, T); exit(EXIT_FAILURE); }
        ck->seq = std::string(sequence + idx, num_bases);
        ck->qstring = std::string(qstring + idx, num_bases);
        if (ck->seq.empty() || ck->qstring.empty() || ck->seq.size() != ck->qstring.size()) {
            ERROR("gpu decode produced bad seq/qstring (seq %zu qstr %zu)", ck->seq.size(), ck->qstring.size());
            exit(EXIT_FAILURE);
        }
    }
    free(moves); free(sequence); free(qstring);
    ts->time_decode += realtime();
}
#endif

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
