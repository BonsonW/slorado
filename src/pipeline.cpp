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

// defined in torchbox.cpp
void free_read_dat(read_dat_t *read_dat);

// Shared state passed to every stage. Just a bag of pointers: the queues and counters
// are owned as locals by run_pipeline(). total_bytes has a single writer (loader) and
// total_reads a single writer (writer stage), so plain integers are safe (the joins in
// run_pipeline establish the happens-before before they are read).
#if defined(HAVE_METAL)
// Metal overlap: a packed batch's host scores + its chunk items, queued from the inference stage to
// the CPU decode stage so decode runs concurrently with the next batch's GPU inference.
struct decode_item_t {
    std::vector<chunk_item_t> items;
    at::Tensor dev_scores;           // device [nchunks, T, C] int8/fp32 (decode stage GPU-scans)
    at::Tensor host_scores;          // host copy (runner-side, after sync) for the CPU beam
    int runner_idx;
};
#endif

typedef struct {
    core_t *core;
    bool mod;                        // modbase calling enabled (--mod)
    BoundedQueue<std::shared_ptr<read_state_t>> *read_q;
    BoundedQueue<chunk_item_t> *chunk_q;
    BoundedQueue<std::shared_ptr<read_state_t>> *stitch_q;
#if defined(HAVE_METAL)
    BoundedQueue<decode_item_t> *decode_q;   // inference stage -> GPU-scan+CPU-beam stage (Metal overlap)
#endif
    // modbase stages (only used when mod)
    BoundedQueue<std::shared_ptr<read_state_t>> *mod_pre_q;
    BoundedQueue<mod_chunk_item_t> *mod_chunk_q;
    BoundedQueue<std::shared_ptr<read_state_t>> *mod_post_q;
    BoundedQueue<std::shared_ptr<read_state_t>> *out_q;
    uint64_t total_reads;
    uint64_t total_bytes;
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
#if defined(HAVE_METAL)
    runner_stat_t *ts = (*core->runner_stats)[runner_idx];
    ts->time_wall -= realtime();
#endif

    std::vector<chunk_item_t> buf;
    buf.reserve(gpu_batch);

#if defined(HAVE_METAL)
    // Double-buffered: hold one batch's inference result (device scores, CRF+quant enqueued but not
    // yet synced) as `pending`. When the NEXT batch's conv+LSTM has run (during which pending's
    // CRF+quant executed on the MPS stream, overlapped), finalize pending cheaply and push it. This
    // keeps the (~1.5s) CRF+quant+copy off the runner's critical path -- it hides behind inference.
    decode_item_t pending;
    bool have_pending = false;
    // Only the plain-LSTM family can split conv+LSTM from the CRF; others (tx/flstm) use the combined
    // forward with no double-buffer (the CRF+quant stays inline).
    const bool split = model_supports_split((*core->runners)[runner_idx]);

    auto infer_and_pipeline = [&]() {
        if (buf.empty()) return;
        std::vector<basecall_chunk_t *> ptrs;
        ptrs.reserve(buf.size());
        for (size_t i = 0; i < buf.size(); ++i) ptrs.push_back(&buf[i].read->chunks[buf[i].chunk_idx]);

        if (!split) {
            // combined forward -> device scores; finalize immediately (no overlap available).
            decode_item_t di;
            at::Tensor dev = basecall_infer_host(core, runner_idx, ptrs);
            di.host_scores = basecall_finalize_host(core, dev);
            di.dev_scores = std::move(dev);
            di.items = std::move(buf);
            di.runner_idx = runner_idx;
            ctx->decode_q->push(std::move(di));
            buf = std::vector<chunk_item_t>();
            buf.reserve(gpu_batch);
            return;
        }

        // 1. conv+LSTM ONLY (blocks on metal_lstm's queue). During this block the PREVIOUS batch's
        //    CRF+quant (enqueued in step 3 last time) executes on the MPS stream -- overlapped.
        at::Tensor lstm_out = basecall_infer_lstm_host(core, runner_idx, ptrs);

        // 2. Finalize+push the previous batch. The sync is cheap: its CRF+quant already ran during
        //    step 1 above, and the current batch's CRF+quant is NOT enqueued yet (step 3), so the
        //    sync doesn't drag it onto the critical path.
        if (have_pending) {
            ts->time_scores_copy -= realtime();
            pending.host_scores = basecall_finalize_host(core, pending.dev_scores);
            ts->time_scores_copy += realtime();
            ts->time_push -= realtime();
            ctx->decode_q->push(std::move(pending));
            ts->time_push += realtime();
        }

        // 3. Now enqueue THIS batch's CRF+quant (async) -> hides behind the next batch's conv+LSTM.
        pending = decode_item_t();
        pending.dev_scores = basecall_crf_quant(core, runner_idx, lstm_out);
        pending.items = std::move(buf);
        pending.runner_idx = runner_idx;
        have_pending = true;
        buf = std::vector<chunk_item_t>();
        buf.reserve(gpu_batch);
    };

    chunk_item_t item;
    ts->time_pop -= realtime();
    while (ctx->chunk_q->pop(item)) {
        ts->time_pop += realtime();
        buf.push_back(std::move(item));
        if (buf.size() == gpu_batch) infer_and_pipeline();
        ts->time_pop -= realtime();
    }
    ts->time_pop += realtime();
    infer_and_pipeline();                // infer the final partial batch (becomes pending)
    if (have_pending) {                  // drain the last pending batch
        pending.host_scores = basecall_finalize_host(core, pending.dev_scores);
        ctx->decode_q->push(std::move(pending));
    }
    ts->time_wall += realtime();
#else
    auto flush = [&]() {
        if (buf.empty()) return;
        std::vector<basecall_chunk_t *> ptrs;
        ptrs.reserve(buf.size());
        for (size_t i = 0; i < buf.size(); ++i) ptrs.push_back(&buf[i].read->chunks[buf[i].chunk_idx]);
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
#endif
#if !defined(HAVE_METAL)
    while (ctx->chunk_q->pop(item)) {
        buf.push_back(std::move(item));
        if (buf.size() == gpu_batch) flush();
    }
    flush();
#endif
}

#if defined(HAVE_METAL)
// Decode-stage instrumentation (single decode thread): busy vs blocked-on-queue.
static double g_dec_wall = 0, g_dec_pop = 0, g_dec_scan = 0, g_dec_decode = 0, g_dec_push = 0;
static uint64_t g_dec_batches = 0;

// Stage 3b (Metal): GPU forward/backward scan + CPU beam search per batch. Runs on its own thread so
// the GPU scan overlaps the runner stage's next-batch inference (separate Metal command queues) and
// the CPU beam overlaps it too. Owns a single gpubuf (scan writes it, beam reads it, serially).
static void decode_stage(pipeline_ctx_t *ctx) {
    core_t *core = ctx->core;
    const int scan_T = (int)(core->chunk_size / core->model_stride);
    openfish_gpubuf_t *gpubuf = openfish_gpubuf_init(scan_T, (int)core->opt.gpu_batch_size,
                                                     core->model_config->state_len);
    decode_item_t di;
    g_dec_wall -= realtime();
    g_dec_pop -= realtime();
    while (ctx->decode_q->pop(di)) {
        g_dec_pop += realtime();
        std::vector<basecall_chunk_t *> ptrs;
        ptrs.reserve(di.items.size());
        for (size_t i = 0; i < di.items.size(); ++i) ptrs.push_back(&di.items[i].read->chunks[di.items[i].chunk_idx]);

        // GPU scan (fills gpubuf bwd/post; runner already synced the scores) then CPU beam.
        g_dec_scan -= realtime();
        basecall_scan_gpu(core, di.runner_idx, di.dev_scores, gpubuf);
        g_dec_scan += realtime();
        di.dev_scores = at::Tensor();   // release the device scores now the scan has consumed them
        g_dec_decode -= realtime();
        basecall_beam_host(core, di.runner_idx, di.host_scores, gpubuf, ptrs);
        g_dec_decode += realtime();

        g_dec_push -= realtime();
        for (size_t i = 0; i < di.items.size(); ++i) {
            if (di.items[i].read->chunks_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                ctx->stitch_q->push(di.items[i].read);
            }
        }
        g_dec_push += realtime();
        g_dec_batches++;
        g_dec_pop -= realtime();
    }
    g_dec_pop += realtime();
    g_dec_wall += realtime();
    openfish_gpubuf_free(gpubuf);
}
#endif

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
    BoundedQueue<std::shared_ptr<read_state_t>> stitch_q(stitch_cap);
    BoundedQueue<std::shared_ptr<read_state_t>> mod_pre_q(stitch_cap);
    BoundedQueue<mod_chunk_item_t> mod_chunk_q(chunk_cap);
    BoundedQueue<std::shared_ptr<read_state_t>> mod_post_q(out_cap);
    BoundedQueue<std::shared_ptr<read_state_t>> out_q(out_cap);
#if defined(HAVE_METAL)
    // Each item holds a batch's device+host int8 scores; bound resident memory with a small cap. The
    // GPU scan + CPU beam happen in the decode stage (which owns its gpubuf).
    BoundedQueue<decode_item_t> decode_q(2);
#endif

    pipeline_ctx_t ctx;
    ctx.core = core;
    ctx.mod = mod;
    ctx.read_q = &read_q;
    ctx.chunk_q = &chunk_q;
    ctx.stitch_q = &stitch_q;
#if defined(HAVE_METAL)
    ctx.decode_q = &decode_q;
#endif
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
        fprintf(stderr, "[%s] streaming pipeline: %d preprocess, %d runner, %d stitch threads\n",
                __func__, n_pre, n_runners, n_stitch);
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

#if defined(HAVE_METAL)
    // CPU decode stage(s) sit between the inference (runner) stage and stitching. One thread is
    // enough: openfish_decode_cpu parallelises each batch internally across num_thread threads.
    std::vector<std::thread> decoders;
    decoders.emplace_back(decode_stage, &ctx);
#endif

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
    for (auto &t : runners) t.join();       // inference stage done
#if defined(HAVE_METAL)
    decode_q.close();                       // no more batches to decode
    for (auto &t : decoders) t.join();      // CPU decode drained -> all reads pushed to stitch
#endif
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

#if defined(HAVE_METAL)
    if (getenv("SLORADO_PIPELINE_STATS")) {
        fprintf(stderr, "\n[pipeline_stats] per-stage busy/blocked breakdown (metal streaming)\n");
        for (int i = 0; i < n_runners; ++i) {
            runner_stat_t *ts = (*core->runner_stats)[i];
            double w = ts->time_wall > 0 ? ts->time_wall : 1.0;
            double busy = ts->time_accept + ts->time_todev + ts->time_forward + ts->time_scores_copy;
            fprintf(stderr, "[pipeline_stats] runner[%d]: wall=%.2fs batches=%llu busy=%.1f%% "
                    "(accept=%.2f todev=%.2f forward=%.2f scores_copy=%.2f) pop_wait=%.2f(%.1f%%) push_wait=%.2f(%.1f%%)\n",
                    i, ts->time_wall, (unsigned long long)ts->n_batches, 100.0 * busy / w,
                    ts->time_accept, ts->time_todev, ts->time_forward, ts->time_scores_copy,
                    ts->time_pop, 100.0 * ts->time_pop / w, ts->time_push, 100.0 * ts->time_push / w);
        }
        double dw = g_dec_wall > 0 ? g_dec_wall : 1.0;
        fprintf(stderr, "[pipeline_stats] decode: wall=%.2fs batches=%llu gpu_scan=%.2f(%.1f%%) "
                "cpu_beam=%.2f(%.1f%%) pop_wait=%.2f(%.1f%%) push_wait=%.2f(%.1f%%)\n",
                g_dec_wall, (unsigned long long)g_dec_batches, g_dec_scan, 100.0 * g_dec_scan / dw,
                g_dec_decode, 100.0 * g_dec_decode / dw,
                g_dec_pop, 100.0 * g_dec_pop / dw, g_dec_push, 100.0 * g_dec_push / dw);
    }
#endif
}
