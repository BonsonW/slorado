/* @file pipeline.cpp
**
** async (pipelined) basecalling path.
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
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "pipeline.h"
#include "torchbox.h"
#include "basecall.h"
#include "writer.h"
#include "error.h"
#include "misc.h"

#include "dorado/tensor_chunk_utils.h"

// State for a single read as it flows through the stages. Held by std::shared_ptr so ownership is
// shared between the in-flight chunk items and the stage currently working on the read; freed once
// the writer is done. (make_shared value-initialises this, zeroing the scalar/atomic members.)
typedef struct {
    slow5_rec_t *rec;                // read_id, len_raw_signal; freed by writer
    read_dat_t *read_dat;            // scaled_signal etc.; freed by writer

    std::vector<basecall_chunk_t> chunks;
    std::atomic<int> chunks_remaining;

    std::string sequence;
    std::string qstring;
    std::vector<uint8_t> moves;

    // modbase
    std::vector<mod_chunk_t> mod_chunks;
    std::atomic<int> mod_chunks_remaining;   // multi-writer: mod runners decrement
    std::string mod_string;                  // MM tag
    std::vector<uint8_t> mod_prob;           // ML tag
} read_state_t;

// undecoded record as read off disk. preprocess_stage is responsible for freeing it.
typedef struct {
    char *mem;
    size_t bytes;
} raw_rec_t;

// One chunk of a read queued for the runner stage. Holds a shared_ptr so the
// read (and its chunk vector) stays alive across the GPU forward pass.
typedef struct {
    std::shared_ptr<read_state_t> read;
    int chunk_idx;
} chunk_item_t;

// One modbase chunk of a read queued for the mod runner stage.
typedef struct {
    std::shared_ptr<read_state_t> read;
    int chunk_idx;
} mod_chunk_item_t;

// Thread-safe bounded blocking queue (MPMC). push() blocks while full; pop()
// blocks while empty and returns false once the queue is drained AND closed.
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mtx_);
        not_full_.wait(lock, [this] { return q_.size() < capacity_ || closed_; });
        if (closed_) return;  // dropped on shutdown; upstream should not push after close
        q_.push_back(std::move(item));
        lock.unlock();
        not_empty_.notify_one();
    }

    // Returns false when the queue is empty and closed (no more items will arrive).
    bool pop(T &out) {
        std::unique_lock<std::mutex> lock(mtx_);
        not_empty_.wait(lock, [this] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;  // closed and drained
        out = std::move(q_.front());
        q_.pop_front();
        lock.unlock();
        not_full_.notify_one();
        return true;
    }

    // Signal that no more items will be pushed; wakes all waiters.
    void close() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    std::mutex mtx_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    std::deque<T> q_;
    size_t capacity_;
    bool closed_ = false;
};

// Shared state passed to every stage.
typedef struct {
    core_t *core;
    bool mod;

    BoundedQueue<raw_rec_t> *read_q;
    BoundedQueue<chunk_item_t> *chunk_q;
    BoundedQueue<std::shared_ptr<read_state_t>> *stitch_q;

    // modbase stages (only used when mod)
    BoundedQueue<std::shared_ptr<read_state_t>> *mod_pre_q;
    BoundedQueue<mod_chunk_item_t> *mod_chunk_q;
    BoundedQueue<std::shared_ptr<read_state_t>> *mod_post_q;
    BoundedQueue<std::shared_ptr<read_state_t>> *out_q;

    uint64_t total_reads;
    uint64_t total_bytes;
} pipeline_ctx_t;

// The batch path processes (and emits) exactly the reads that carry signal.
static inline bool read_has_signal(const std::shared_ptr<read_state_t> &rs) {
    return rs->rec->len_raw_signal > 0;
}

// read raw recs, single thread
static void loader_stage(pipeline_ctx_t *ctx) {
    core_t *core = ctx->core;

    while (true) {
        raw_rec_t raw = {NULL, 0};
        if (slow5_get_next_bytes(&raw.mem, &raw.bytes, core->sp) < 0) {
            if (slow5_errno != SLOW5_ERR_EOF) {
                ERROR("Error reading from SLOW5 file %d", slow5_errno);
                exit(EXIT_FAILURE);
            }
            break;
        }
        ctx->total_bytes += raw.bytes;
        ctx->read_q->push(raw);
    }

    ctx->read_q->close();
}

// parse the record, then scale the signal and split it into overlapping chunks
static void preprocess_stage(pipeline_ctx_t *ctx) {
    core_t *core = ctx->core;
    raw_rec_t raw;

    while (ctx->read_q->pop(raw)) {
        auto rs = std::make_shared<read_state_t>();
        // slow5_decode may realloc raw.mem (and overwrite raw.bytes); it does not free it.
        if (slow5_decode(&raw.mem, &raw.bytes, &rs->rec, core->sp) < 0) {
            ERROR("%s", "Error decoding a SLOW5 record");
            exit(EXIT_FAILURE);
        }
        free(raw.mem);

        if (read_has_signal(rs)) {
            rs->read_dat = new read_dat_t;
            preprocess_signal(core, rs->rec, rs->read_dat, rs->chunks);
        }

        if (rs->chunks.empty()) {
            // zero-length signal: nothing to basecall, straight to output (nothing emitted)
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

// pack chunks to gpu_batch_size across reads and run inference+decode
// one thread per GPU
static void runner_stage(pipeline_ctx_t *ctx, int runner_idx) {
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

// stitch reads
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
            }
        }

        // queue mod or out
        if (ctx->mod && read_has_signal(rs)) {
            ctx->mod_pre_q->push(std::move(rs));
        } else {
            ctx->out_q->push(std::move(rs));
        }
    }
}

// mod preproccess: build the seq->signal mapping and modbase chunks; emit one item per mod chunk.
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

// mod runner: pack mod chunks to mod_gpu_batch_size across reads and run the modbase model
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

// mod postprocess: turn base_mod_probs into MM/ML tags
static void mod_postprocess_stage(pipeline_ctx_t *ctx) {
    core_t *core = ctx->core;
    std::shared_ptr<read_state_t> rs;

    while (ctx->mod_post_q->pop(rs)) {
        postprocess_modbase(core, rs->read_dat, rs->mod_string, rs->mod_prob);
        ctx->out_q->push(std::move(rs));
    }
}

// write output, single thread
static void writer_stage(pipeline_ctx_t *ctx) {
    core_t *core = ctx->core;
    const bool sam = (core->opt.flag & SLORADO_SAM) != 0;
    std::shared_ptr<read_state_t> rs;
    uint64_t n = 0;

    while (ctx->out_q->pop(rs)) {
        if (read_has_signal(rs)) {
            if (sam) {
                write_to_file_sam(core->opt.out, rs->sequence.c_str(), rs->qstring.c_str(), rs->rec->read_id, rs->mod_string.c_str(), rs->mod_prob);
            } else {
                write_to_file_fastq(core->opt.out, rs->sequence.c_str(), rs->qstring.c_str(), rs->rec->read_id);
            }
        }
        ++n;

        // free
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
        ERROR("%s", "no runners available for the async pipeline");
        exit(EXIT_FAILURE);
    }
    const bool mod = core->opt.mod != NULL;
    const int n_mod_runners = mod ? (int)core->mod_runners->size() : 0;

    // one thread for reading, one thread for writing
    // split the worker-thread budget between preprocess and stitch (heavier for preprocess)
    // single runner thread are per GPU
    const int workers = std::max(2, core->opt.num_thread);
    const int n_pre = std::max(1, (workers * 2) / 3);
    const int n_stitch = std::max(1, workers - n_pre);

    const size_t gpu_batch = (size_t)core->opt.gpu_batch_size;
    const size_t mod_gpu_batch = (size_t)core->opt.mod_gpu_batch_size;
    const size_t chunk_cap = std::max<size_t>(gpu_batch * 4 * n_runners, gpu_batch * 4);
    const size_t mod_chunk_cap = std::max<size_t>(mod_gpu_batch * 4 * std::max(n_mod_runners, 1), mod_gpu_batch * 4);
    const size_t read_cap = std::max<size_t>(256, (size_t)n_pre * 8);
    const size_t stitch_cap = 256;
    const size_t out_cap = 256;

    BoundedQueue<raw_rec_t> read_q(read_cap);
    BoundedQueue<chunk_item_t> chunk_q(chunk_cap);
    BoundedQueue<std::shared_ptr<read_state_t>> stitch_q(stitch_cap);
    BoundedQueue<std::shared_ptr<read_state_t>> mod_pre_q(stitch_cap);
    BoundedQueue<mod_chunk_item_t> mod_chunk_q(mod_chunk_cap);
    BoundedQueue<std::shared_ptr<read_state_t>> mod_post_q(out_cap);
    BoundedQueue<std::shared_ptr<read_state_t>> out_q(out_cap);

    pipeline_ctx_t ctx;
    ctx.core = core;
    ctx.mod = mod;
    ctx.read_q = &read_q;
    ctx.chunk_q = &chunk_q;
    ctx.stitch_q = &stitch_q;
    ctx.mod_pre_q = &mod_pre_q;
    ctx.mod_chunk_q = &mod_chunk_q;
    ctx.mod_post_q = &mod_post_q;
    ctx.out_q = &out_q;
    ctx.total_reads = 0;
    ctx.total_bytes = 0;

    if (mod) {
        fprintf(stderr, "[%s] async pipeline: %d preprocess, %d runner, %d stitch, "
                "%d mod-preprocess, %d mod-runner, %d mod-postprocess threads\n",
                __func__, n_pre, n_runners, n_stitch, n_pre, n_mod_runners, n_stitch);
    } else {
        fprintf(stderr, "[%s] async pipeline: %d preprocess, %d runner, %d stitch threads\n",
                __func__, n_pre, n_runners, n_stitch);
    }

    std::thread writer(writer_stage, &ctx);

    std::vector<std::thread> mod_postproc, mod_runners, mod_preproc;
    if (mod) {
        for (int i = 0; i < n_stitch; ++i) mod_postproc.emplace_back(mod_postprocess_stage, &ctx);
        for (int i = 0; i < n_mod_runners; ++i) mod_runners.emplace_back(mod_runner_stage, &ctx, i);
        for (int i = 0; i < n_pre; ++i) mod_preproc.emplace_back(mod_preprocess_stage, &ctx);
    }

    std::vector<std::thread> stitch;
    for (int i = 0; i < n_stitch; ++i) stitch.emplace_back(stitch_stage, &ctx);

    std::vector<std::thread> runners;
    for (int i = 0; i < n_runners; ++i) runners.emplace_back(runner_stage, &ctx, i);

    std::vector<std::thread> preproc;
    for (int i = 0; i < n_pre; ++i) preproc.emplace_back(preprocess_stage, &ctx);

    std::thread loader(loader_stage, &ctx);

    // closing each queue only after its producers are done preserves full pipeline overlap during steady state
    loader.join();
    for (auto &t : preproc) t.join();
    chunk_q.close();
    for (auto &t : runners) t.join();
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
}
