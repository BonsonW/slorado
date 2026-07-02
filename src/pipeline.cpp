/* @file pipeline.cpp
**
** streaming (pipelined) basecalling path. See pipeline.h for the rationale.
**
** Stages (each its own thread(s)), connected by bounded blocking queues:
**   loader(1) -> preprocess(P) -> runners(R=#runners) -> stitch(S) -> writer(1)
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
typedef struct {
    core_t *core;
    BoundedQueue<std::shared_ptr<read_state_t>> *read_q;
    BoundedQueue<chunk_item_t> *chunk_q;
    BoundedQueue<std::shared_ptr<read_state_t>> *stitch_q;
    BoundedQueue<std::shared_ptr<read_state_t>> *out_q;
    uint64_t total_reads;
    uint64_t total_bytes;
} pipeline_ctx;

// Stage 1: read raw records from the slow5 file, decode them, emit read_state_t.
static void loader_stage(pipeline_ctx *ctx) {
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
static void preprocess_stage(pipeline_ctx *ctx) {
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

// Stage 3: pack chunks to gpu_batch_size across reads and run inference+decode.
static void runner_stage(pipeline_ctx *ctx, int runner_idx) {
    core_t *core = ctx->core;
    const size_t gpu_batch = (size_t)core->opt.gpu_batch_size;

    std::vector<chunk_item_t> buf;
    buf.reserve(gpu_batch);

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
    while (ctx->chunk_q->pop(item)) {
        buf.push_back(std::move(item));
        if (buf.size() == gpu_batch) flush();
    }
    flush();
}

// Stage 4: stitch a read's chunks back into a single sequence (+ RNA reversal).
static void stitch_stage(pipeline_ctx *ctx) {
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
        ctx->out_q->push(std::move(rs));
    }
}

// Stage 5: write output (FASTQ) and free per-read resources. Single thread keeps
// fprintf serialized; output order is not guaranteed to match the input file.
static void writer_stage(pipeline_ctx *ctx) {
    core_t *core = ctx->core;
    std::shared_ptr<read_state_t> rs;
    uint64_t n = 0;

    while (ctx->out_q->pop(rs)) {
        if (rs->rec->len_raw_signal > 0 && !rs->sequence.empty()) {
            write_to_file_fastq(core->opt.out, rs->sequence.c_str(), rs->qstring.c_str(),
                                rs->rec->read_id);
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
    BoundedQueue<std::shared_ptr<read_state_t>> out_q(out_cap);

    pipeline_ctx ctx;
    ctx.core = core;
    ctx.read_q = &read_q;
    ctx.chunk_q = &chunk_q;
    ctx.stitch_q = &stitch_q;
    ctx.out_q = &out_q;
    ctx.total_reads = 0;
    ctx.total_bytes = 0;

    fprintf(stderr, "[%s] streaming pipeline: %d preprocess, %d runner, %d stitch threads\n",
            __func__, n_pre, n_runners, n_stitch);

    // Start downstream stages first so they are ready to consume.
    std::thread writer(writer_stage, &ctx);

    std::vector<std::thread> stitch;
    for (int i = 0; i < n_stitch; ++i) stitch.emplace_back(stitch_stage, &ctx);

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
    stitch_q.close();
    for (auto &t : stitch) t.join();
    out_q.close();
    writer.join();

    core->total_reads = (int64_t)ctx.total_reads;
    core->sum_bytes = (int64_t)ctx.total_bytes;
}
