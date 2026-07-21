/* @file pipeline.cpp
**
** streaming (pipelined) basecalling path. See pipeline.h for the rationale.
**
** Stages (each its own thread(s)), connected by bounded blocking queues:
**   loader(1) -> preprocess(P) -> runners(R=#runners) -> stitch(S) -> writer(1)
**
** With --mod, three more stages are spliced in after stitch:
**   ... -> stitch(S) -> mod_preprocess(P) -> mod_runners(R) -> mod_postprocess(S) -> writer(1)
**
** A read (read_state_t, shared_ptr) is decoded by the loader, scaled+chunked by a
** preprocess worker, and each of its chunks is pushed individually to a single global
** chunk queue. Runner threads pop chunks, pack them to gpu_batch_size across read
** boundaries and run inference+decode; when a read's last chunk is done it flows to
** stitching and then output. Bounded queues provide backpressure so the whole file is
** never held in memory at once.
** @@
******************************************************************************/

#include <algorithm>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "pipeline.h"
#include "torchbox.h"
#include "basecall.h"
#include "modcall.h"
#include "writer.h"
#include "error.h"
#include "misc.h"

#include "dorado/tensor_chunk_utils.h"

#ifdef USE_GPU
#ifdef HAVE_CUDA
#include <c10/cuda/CUDAStream.h>
#elif defined(HAVE_ROCM)
#include <c10/hip/HIPStream.h>
#endif
// Synchronize ONLY the runner's own (torch current) stream -- enough to guarantee the forward + int8
// cast are done and input_tensor is safe to reuse -- WITHOUT the device-wide wait of
// torch::cuda::synchronize(), which would also block on the decode thread's scan stream and serialize
// the two. Lets the scan's dedicated stream actually overlap inference.
static inline void sync_runner_stream(int64_t dev) {
#ifdef HAVE_CUDA
    c10::cuda::getCurrentCUDAStream(dev).synchronize();
#elif defined(HAVE_ROCM)
    c10::hip::getCurrentHIPStream(dev).synchronize();
#endif
}
#endif

// defined in torchbox.cpp
void free_read_dat(read_dat_t *read_dat);

// One inference batch handed from the runner stage to the decode stage in --cpu-beam mode. Carries
// the chunk items (their shared_ptr reads stay alive for decode + bookkeeping) plus the scores the two
// decode halves read: scan_scores (GPU-side, for openfish_decode_gpu_scan) and beam_scores (host-side,
// for the CPU beam). For int8 clamp models both alias ONE managed buffer (zero-copy); for fp16 models
// scan_scores is the device fp16 tensor and beam_scores an fp32 host copy (the scan needs fp16, the
// beam needs fp32, so they cannot share).
typedef struct {
    std::vector<chunk_item_t> items;
    at::Tensor scan_scores;
    at::Tensor beam_scores;
    int runner_idx;
} decode_item_t;

// Shared state passed to every stage. Just a bag of pointers: the queues and counters
// are owned as locals by run_pipeline(). total_bytes has a single writer (loader) and
// total_reads a single writer (writer stage), so plain integers are safe (the joins in
// run_pipeline establish the happens-before before they are read).
typedef struct {
    core_t *core;
    bool mod;                        // modbase calling enabled (--mod)
    bool cpu_beam;                   // --cpu-beam: split decode onto GPU scan + CPU beam (decode_stage)
    BoundedQueue<std::shared_ptr<read_state_t>> *read_q;
    BoundedQueue<chunk_item_t> *chunk_q;
    BoundedQueue<decode_item_t> *decode_q;   // runner -> decode_stage (only used when cpu_beam)
    ManagedScorePool *score_pool;            // managed int8 score buffers (cpu_beam; NULL otherwise)
    BoundedQueue<std::shared_ptr<read_state_t>> *stitch_q;
    // modbase stages (only used when mod)
    BoundedQueue<std::shared_ptr<read_state_t>> *mod_pre_q;
    BoundedQueue<mod_chunk_item_t> *mod_chunk_q;
    BoundedQueue<std::shared_ptr<read_state_t>> *mod_post_q;
    BoundedQueue<std::shared_ptr<read_state_t>> *out_q;
    uint64_t total_reads;
    uint64_t total_bytes;

    // --cpu-beam profiling (printed when SLORADO_PIPELINE_STATS is set). Single runner + single
    // decode thread each, so these plain doubles have one writer apiece (safe to read after join).
    double t_infer = 0;       // runner: model forward + int8 quant (kernel launch, async)
    double t_hostcopy = 0;    // runner: scores.to(CPU) -- this is where the GPU inference actually blocks
    double t_push_wait = 0;   // runner: blocked pushing to decode_q (full => decode is the bottleneck)
    double t_pop_wait = 0;    // decode: blocked waiting for a batch  (empty => inference is the bottleneck)
    double t_scan = 0;        // decode: GPU posterior scan (openfish_decode_gpu_scan, incl. sync)
    double t_beam = 0;        // decode: CPU beam search + writeback
    uint64_t n_dec_batches = 0;
} pipeline_ctx_t;

// A read is worth basecalling / mod calling only if it has signal and produced a sequence.
static inline bool read_is_valid(const std::shared_ptr<read_state_t> &rs) {
    return rs->rec->len_raw_signal > 0 && !rs->sequence.empty();
}

// Stage 1: read raw records from the slow5 file, decode them, emit read_state_t.
static void loader_stage(pipeline_ctx_t *ctx) {
    core_t *core = ctx->core;
    uint64_t seq_no = 0;

    while (true) {
        char *mem = NULL;
        size_t bytes = 0;
        if (slow5_get_next_bytes(&mem, &bytes, core->sp) < 0) {
            if (slow5_errno != SLOW5_ERR_EOF) {
                ERROR("Error reading from SLOW5 file %d", slow5_errno);
                exit(EXIT_FAILURE);
            }
            break;  // EOF
        }

        auto rs = std::make_shared<read_state_t>();
        // slow5_decode may realloc *mem; it does not free it.
        if (slow5_decode(&mem, &bytes, &rs->rec, core->sp) < 0) {
            ERROR("%s", "Error decoding a SLOW5 record");
            exit(EXIT_FAILURE);
        }
        free(mem);

        rs->seq_no = seq_no++;
        ctx->total_bytes += bytes;
        ctx->read_q->push(std::move(rs));
    }

    ctx->read_q->close();
}

// Stage 2: scale signal and split into overlapping chunks; emit one item per chunk.
static void preprocess_stage(pipeline_ctx_t *ctx) {
    core_t *core = ctx->core;
    std::shared_ptr<read_state_t> rs;

    while (ctx->read_q->pop(rs)) {
        if (rs->rec->len_raw_signal > 0) {
            rs->read_dat = new read_dat_t;
            preprocess_signal(core, rs->rec, rs->read_dat, rs->chunks);
        }

        if (rs->chunks.empty()) {
            // zero-length signal: nothing to basecall, straight to output (no fastq emitted)
            ctx->stitch_q->push(std::move(rs));
            continue;
        }

        rs->chunks_remaining.store((int)rs->chunks.size(), std::memory_order_relaxed);
        int n = (int)rs->chunks.size();
        for (int c = 0; c < n; ++c) {
            chunk_item_t item;
            item.read = rs;
            item.chunk_idx = c;
            ctx->chunk_q->push(item);
        }
    }
}

// Stage 3: pack chunks to gpu_batch_size across reads and run inference+decode. Decode subtiles
// internally (see basecall.cpp) so its scratch/decode-buffer stay bounded at large batch. (Decode
// is inline, not a separate stage: at the batch sizes we run the GPU is saturated by inference, so
// a decode thread can't overlap it -- it just adds an extra resident scores tensor.)
static void runner_stage(pipeline_ctx_t *ctx, int runner_idx) {
    core_t *core = ctx->core;
    const size_t gpu_batch = (size_t)core->opt.gpu_batch_size;

    std::vector<chunk_item_t> buf;
    buf.reserve(gpu_batch);

    // --cpu-beam: run inference only, then hand the scores to the decode_stage (GPU scan + CPU beam)
    // so decode of this batch overlaps inference of the next. Otherwise decode inline (fused GPU).
    auto flush = [&]() {
        if (buf.empty()) return;
        std::vector<basecall_chunk_t *> ptrs;
        ptrs.reserve(buf.size());
        for (size_t i = 0; i < buf.size(); ++i) ptrs.push_back(&buf[i].read->chunks[buf[i].chunk_idx]);

#ifdef USE_GPU
        if (ctx->cpu_beam) {
            double a = realtime();
            // int8 clamp models -> scores land in managed memory (zero-copy); fp16 -> device tensor.
            at::Tensor scores = basecall_infer_scores(core, runner_idx, ptrs, ctx->score_pool);
            const bool i8 = scores.scalar_type() == at::kChar;
            double b = realtime();

            at::Tensor scan_scores, beam_scores;
            if (i8) {
                // Managed buffer: both the GPU scan and CPU beam read it directly. Sync the runner's
                // own stream so the quant's write (and the whole forward) is complete before the decode
                // thread reads the buffer -- stream-scoped, so it does NOT wait on the decode scan.
                sync_runner_stream((*core->runners)[runner_idx]->device_idx);
                scan_scores = scores;
                beam_scores = scores;
            } else {
                // fp16: GPU scan reads the device fp16 tensor; the CPU beam needs an fp32 host copy
                // (openfish's OPENFISH_SCORE_F16 == fp16 on the GPU, fp32 on the CPU). .to(CPU) syncs.
                scan_scores = scores;
                beam_scores = scores.to(torch::kCPU, torch::kFloat32);
            }
            double c = realtime();
            decode_item_t di;
            di.items = std::move(buf);
            di.scan_scores = std::move(scan_scores);
            di.beam_scores = std::move(beam_scores);
            di.runner_idx = runner_idx;
            ctx->decode_q->push(std::move(di));
            double d = realtime();
            ctx->t_infer += b - a;
            ctx->t_hostcopy += c - b;   // managed sync (int8) or fp32 host copy (fp16)
            ctx->t_push_wait += d - c;
            buf.clear();   // moved-from; ensure empty for the next batch
            return;
        }
#endif

        basecall_chunks(core, runner_idx, ptrs);

        for (size_t i = 0; i < buf.size(); ++i) {
            // last chunk of this read done -> hand off to stitching
            if (buf[i].read->chunks_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                ctx->stitch_q->push(buf[i].read);
            }
        }
        buf.clear();
    };

    chunk_item_t item;
    while (ctx->chunk_q->pop(item)) {
        buf.push_back(std::move(item));
        if (buf.size() == gpu_batch) flush();
    }
    flush();
}

// Stage 3b (--cpu-beam): decode a batch of scores produced by the runner stage. Runs the GPU
// forward/backward posterior scan then the CPU beam search on its own thread, so this batch's decode
// overlaps the runner stage's inference of the next batch. One thread suffices because the CPU beam
// is internally multi-threaded. Decodes in row-subtiles of decode_tile (like the fused decode_chunks)
// so the host-visible gpubuf's scan tensors stay bounded regardless of the (possibly large) batch;
// the CPU beam reads the GPU-written posteriors from managed memory with no copy.
static void decode_stage(pipeline_ctx_t *ctx) {
#ifdef USE_GPU
    core_t *core = ctx->core;
    const int tile = std::min(core->opt.gpu_batch_size, DEFAULT_GPU_BATCH_SIZE);
    openfish_gpubuf_t *gpubuf = openfish_gpubuf_init_hostvis(
        core->chunk_size / core->model_stride, tile, core->model_config->state_len);

    decode_item_t di;
    while (true) {
        double w0 = realtime();
        bool got = ctx->decode_q->pop(di);
        ctx->t_pop_wait += realtime() - w0;
        if (!got) break;

        const int N = (int)di.items.size();
        for (int n0 = 0; n0 < N; n0 += tile) {
            const int nt = std::min(tile, N - n0);
            std::vector<basecall_chunk_t *> ptrs;
            ptrs.reserve(nt);
            for (int j = 0; j < nt; ++j) ptrs.push_back(&di.items[n0 + j].read->chunks[di.items[n0 + j].chunk_idx]);

            // narrow() on dim 0 of the contiguous [N,T,C] tensors gives a contiguous row-block whose
            // data_ptr() points at row n0 -- exactly what the openfish scan/beam expect for nt rows.
            // (For int8, scan_scores and beam_scores alias the same managed buffer.)
            double s0 = realtime();
            basecall_scan_gpu(core, gpubuf, di.scan_scores.narrow(0, n0, nt));
            double s1 = realtime();
            basecall_beam_cpu(core, gpubuf, di.beam_scores.narrow(0, n0, nt), ptrs);
            ctx->t_scan += s1 - s0;
            ctx->t_beam += realtime() - s1;
        }
        ctx->n_dec_batches++;
        di.scan_scores = at::Tensor{};   // release scores now that all tiles are decoded
        di.beam_scores = at::Tensor{};   // (returns the managed buffer to the pool for int8)

        for (int i = 0; i < N; ++i) {
            if (di.items[i].read->chunks_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                ctx->stitch_q->push(di.items[i].read);
            }
        }
    }

    openfish_gpubuf_free(gpubuf);
#else
    (void)ctx;
#endif
}

// Stage 4: stitch a read's chunks back into a single sequence (+ RNA reversal). Routes valid reads
// to the modbase stages when --mod, else straight to output.
static void stitch_stage(pipeline_ctx_t *ctx) {
    core_t *core = ctx->core;
    const bool rna = is_rna(core->model_config->sample_type);
    std::shared_ptr<read_state_t> rs;

    while (ctx->stitch_q->pop(rs)) {
        if (!rs->chunks.empty()) {
            stitch_chunks_vec(rs->chunks, rs->sequence, rs->qstring, rs->moves,
                              rs->rec->len_raw_signal, (int)core->model_stride);
            if (rna) {
                std::reverse(rs->sequence.begin(), rs->sequence.end());
                std::reverse(rs->qstring.begin(), rs->qstring.end());
                std::reverse(rs->moves.begin(), rs->moves.end());
            }
        }
        if (ctx->mod && read_is_valid(rs)) {
            ctx->mod_pre_q->push(std::move(rs));
        } else {
            ctx->out_q->push(std::move(rs));
        }
    }
}

// Stage 4b (mod): build the seq->signal mapping and modbase chunks; emit one item per mod chunk.
static void mod_preprocess_stage(pipeline_ctx_t *ctx) {
    core_t *core = ctx->core;
    std::shared_ptr<read_state_t> rs;

    while (ctx->mod_pre_q->pop(rs)) {
        preprocess_modbase(core, rs->rec, rs->read_dat, rs->sequence.c_str(), rs->moves, rs->mod_chunks);

        if (rs->mod_chunks.empty()) {
            // no motif hits: still postprocess to emit (empty) MM/ML tags, matching the batch path
            ctx->mod_post_q->push(std::move(rs));
            continue;
        }

        rs->mod_chunks_remaining.store((int)rs->mod_chunks.size(), std::memory_order_relaxed);
        int n = (int)rs->mod_chunks.size();
        for (int c = 0; c < n; ++c) {
            mod_chunk_item_t item;
            item.read = rs;
            item.chunk_idx = c;
            ctx->mod_chunk_q->push(item);
        }
    }
}

// Stage 4c (mod): pack mod chunks to mod_gpu_batch_size across reads and run the modbase model.
static void mod_runner_stage(pipeline_ctx_t *ctx, int runner_idx) {
    core_t *core = ctx->core;
    const size_t gpu_batch = (size_t)core->opt.mod_gpu_batch_size;

    std::vector<mod_chunk_item_t> buf;
    buf.reserve(gpu_batch);

    auto flush = [&]() {
        if (buf.empty()) return;
        std::vector<mod_chunk_t *> ptrs;
        ptrs.reserve(buf.size());
        for (size_t i = 0; i < buf.size(); ++i) ptrs.push_back(&buf[i].read->mod_chunks[buf[i].chunk_idx]);

        mod_basecall_chunks(core, runner_idx, ptrs);

        for (size_t i = 0; i < buf.size(); ++i) {
            // last mod chunk of this read done -> hand off to mod postprocess
            if (buf[i].read->mod_chunks_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                ctx->mod_post_q->push(buf[i].read);
            }
        }
        buf.clear();
    };

    mod_chunk_item_t item;
    while (ctx->mod_chunk_q->pop(item)) {
        buf.push_back(std::move(item));
        if (buf.size() == gpu_batch) flush();
    }
    flush();
}

// Stage 4d (mod): turn base_mod_probs into MM/ML tags.
static void mod_postprocess_stage(pipeline_ctx_t *ctx) {
    core_t *core = ctx->core;
    std::shared_ptr<read_state_t> rs;

    while (ctx->mod_post_q->pop(rs)) {
        postprocess_modbase(core, rs->read_dat, rs->mod_string, rs->mod_prob);
        ctx->out_q->push(std::move(rs));
    }
}

// Stage 5: write output (FASTQ, or SAM with MM/ML under --mod) and free per-read resources.
// Single thread keeps fprintf serialized; output order is not guaranteed to match the input file.
static void writer_stage(pipeline_ctx_t *ctx) {
    core_t *core = ctx->core;
    const bool sam = (core->opt.flag & SLORADO_SAM) != 0;
    std::shared_ptr<read_state_t> rs;
    uint64_t n = 0;

    while (ctx->out_q->pop(rs)) {
        if (read_is_valid(rs)) {
            if (sam) {
                write_to_file_sam(core->opt.out, rs->sequence.c_str(), rs->qstring.c_str(),
                                  rs->rec->read_id, rs->mod_string.c_str(), rs->mod_prob);
            } else {
                write_to_file_fastq(core->opt.out, rs->sequence.c_str(), rs->qstring.c_str(),
                                    rs->rec->read_id);
            }
        }
        ++n;

        if (rs->read_dat) {
            free_read_dat(rs->read_dat);
            rs->read_dat = NULL;
        }
        slow5_rec_free(rs->rec);
        rs->rec = NULL;
    }

    ctx->total_reads = n;
}

void run_pipeline(core_t *core) {
    const int n_runners = (int)core->runners->size();
    if (n_runners < 1) {
        ERROR("%s", "no runners available for streaming pipeline");
        exit(EXIT_FAILURE);
    }
    const bool mod = core->opt.mod != NULL;
    const bool cpu_beam = (core->opt.flag & SLORADO_CPU_BEAM) != 0;
    const int n_mod_runners = mod ? (int)core->mod_runners->size() : 0;

    // Split the worker-thread budget between preprocess (heavier: signal scaling +
    // tensor ops) and stitch, preprocess-weighted. Runner threads are separate (1/GPU).
    const int workers = std::max(2, core->opt.num_thread);
    const int n_pre = std::max(1, (workers * 2) / 3);
    const int n_stitch = std::max(1, workers - n_pre);

    const size_t gpu_batch = (size_t)core->opt.gpu_batch_size;
    const size_t chunk_cap = std::max<size_t>(gpu_batch * 4 * n_runners, gpu_batch * 4);
    const size_t read_cap = std::max<size_t>(64, (size_t)n_pre * 4);
    const size_t stitch_cap = 256;
    const size_t out_cap = 256;

    // Queues and counters are owned here; the context just points at them.
    BoundedQueue<std::shared_ptr<read_state_t>> read_q(read_cap);
    BoundedQueue<chunk_item_t> chunk_q(chunk_cap);
    // Small cap (per runner) bounds the resident score tensors handed to the decode stage.
    BoundedQueue<decode_item_t> decode_q(std::max<size_t>(2, (size_t)n_runners * 2));
    BoundedQueue<std::shared_ptr<read_state_t>> stitch_q(stitch_cap);
    BoundedQueue<std::shared_ptr<read_state_t>> mod_pre_q(stitch_cap);
    BoundedQueue<mod_chunk_item_t> mod_chunk_q(chunk_cap);
    BoundedQueue<std::shared_ptr<read_state_t>> mod_post_q(out_cap);
    BoundedQueue<std::shared_ptr<read_state_t>> out_q(out_cap);

    ManagedScorePool *score_pool = nullptr;
#ifdef USE_GPU
    if (cpu_beam) score_pool = create_score_pool();
#endif

    pipeline_ctx_t ctx;
    ctx.core = core;
    ctx.mod = mod;
    ctx.cpu_beam = cpu_beam;
    ctx.read_q = &read_q;
    ctx.chunk_q = &chunk_q;
    ctx.decode_q = &decode_q;
    ctx.score_pool = score_pool;
    ctx.stitch_q = &stitch_q;
    ctx.mod_pre_q = &mod_pre_q;
    ctx.mod_chunk_q = &mod_chunk_q;
    ctx.mod_post_q = &mod_post_q;
    ctx.out_q = &out_q;
    ctx.total_reads = 0;
    ctx.total_bytes = 0;

    if (mod) {
        fprintf(stderr, "[%s] streaming pipeline: %d preprocess, %d runner, %d stitch, "
                "%d mod-preprocess, %d mod-runner, %d mod-postprocess threads\n",
                __func__, n_pre, n_runners, n_stitch, n_pre, n_mod_runners, n_stitch);
    } else {
        fprintf(stderr, "[%s] streaming pipeline: %d preprocess, %d runner, %s%d stitch threads\n",
                __func__, n_pre, n_runners, cpu_beam ? "1 decode (GPU scan + CPU beam), " : "", n_stitch);
    }

    // Start downstream stages first so they are ready to consume.
    std::thread writer(writer_stage, &ctx);

    std::vector<std::thread> mod_postproc, mod_runners, mod_preproc;
    if (mod) {
        for (int i = 0; i < n_stitch; ++i) mod_postproc.emplace_back(mod_postprocess_stage, &ctx);
        for (int i = 0; i < n_mod_runners; ++i) mod_runners.emplace_back(mod_runner_stage, &ctx, i);
        for (int i = 0; i < n_pre; ++i) mod_preproc.emplace_back(mod_preprocess_stage, &ctx);
    }

    std::vector<std::thread> stitch;
    for (int i = 0; i < n_stitch; ++i) stitch.emplace_back(stitch_stage, &ctx);

    // --cpu-beam: single decode thread between the runners and stitching (CPU beam is internally
    // multi-threaded). Started before the runners so it is ready to consume decode_q.
    std::thread decode;
    if (cpu_beam) decode = std::thread(decode_stage, &ctx);

    std::vector<std::thread> runners;
    for (int i = 0; i < n_runners; ++i) runners.emplace_back(runner_stage, &ctx, i);

    std::vector<std::thread> preproc;
    for (int i = 0; i < n_pre; ++i) preproc.emplace_back(preprocess_stage, &ctx);

    std::thread loader(loader_stage, &ctx);

    // Drain in dependency order: closing each queue only after its producers are done
    // preserves full pipeline overlap during steady state.
    loader.join();                          // loader closed read_q at EOF
    for (auto &t : preproc) t.join();
    chunk_q.close();
    for (auto &t : runners) t.join();
    if (cpu_beam) {
        decode_q.close();                   // runners done -> no more decode items
        decode.join();
    }
    stitch_q.close();
    for (auto &t : stitch) t.join();
    if (mod) {
        mod_pre_q.close();
        for (auto &t : mod_preproc) t.join();
        mod_chunk_q.close();
        for (auto &t : mod_runners) t.join();
        mod_post_q.close();
        for (auto &t : mod_postproc) t.join();
    }
    out_q.close();
    writer.join();

    core->total_reads = (int64_t)ctx.total_reads;
    core->sum_bytes = (int64_t)ctx.total_bytes;

#ifdef USE_GPU
    if (score_pool) destroy_score_pool(score_pool);
#endif

    if (cpu_beam && getenv("SLORADO_PIPELINE_STATS") != NULL) {
        const uint64_t nb = ctx.n_dec_batches ? ctx.n_dec_batches : 1;
        fprintf(stderr,
            "\n[cpu-beam profile] %lu decode batches (n_runners=%d, decode threads=1)\n"
            "  runner : infer(launch)=%.2fs  hostcopy+sync=%.2fs  push_wait=%.2fs\n"
            "  decode : pop_wait=%.2fs  gpu_scan=%.2fs  cpu_beam=%.2fs\n"
            "  per-batch avg (ms): infer=%.1f hostcopy=%.1f push_wait=%.1f | pop_wait=%.1f scan=%.1f beam=%.1f\n"
            "  bottleneck hint: runner push_wait high => decode-bound (CPU beam too slow);"
            " decode pop_wait high => inference-bound (overlap helps)\n",
            (unsigned long)ctx.n_dec_batches, n_runners,
            ctx.t_infer, ctx.t_hostcopy, ctx.t_push_wait,
            ctx.t_pop_wait, ctx.t_scan, ctx.t_beam,
            1e3 * ctx.t_infer / nb, 1e3 * ctx.t_hostcopy / nb, 1e3 * ctx.t_push_wait / nb,
            1e3 * ctx.t_pop_wait / nb, 1e3 * ctx.t_scan / nb, 1e3 * ctx.t_beam / nb);
    }
}
