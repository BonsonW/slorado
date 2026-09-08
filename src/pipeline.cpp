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
** A read (read_state_t) is decoded by the loader, scaled+chunked by a
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
#include <mutex>
#include <pthread.h>
#include <string>
#include <vector>

#include "pipeline.h"
#include "torchbox.h"
#include "basecall.h"
#include "writer.h"
#include "error.h"
#include "misc.h"

#include "dorado/tensor_chunk_utils.h"

// State for a single read as it flows through the stages. Owned by exactly one stage at a time and
// passed down the queues as a plain pointer; the writer deletes it. During the chunk phase the N
// in-flight chunk items all point at it, but none of them owns it -- chunks_remaining below is the
// refcount that decides when the read moves on, so no extra ownership tracking is needed.
typedef struct {
    slow5_rec_t *rec = NULL;         // read_id, len_raw_signal; freed by writer
    read_dat_t *read_dat = NULL;     // scaled_signal etc.; freed by writer

    std::vector<basecall_chunk_t> chunks;
    std::atomic<int> chunks_remaining{0};

    std::string sequence;
    std::string qstring;
    std::vector<uint8_t> moves;

    // modbase
    std::vector<mod_chunk_t> mod_chunks;
    std::atomic<int> mod_chunks_remaining{0}; // multi-writer: mod runners decrement
    std::string mod_string;                  // MM tag
    std::vector<uint8_t> mod_prob;           // ML tag
} read_state_t;

// undecoded record as read off disk. preprocess_stage is responsible for freeing it.
typedef struct {
    char *mem;
    size_t bytes;
} raw_rec_t;

// One chunk of a read queued for the runner stage. Borrows the read; the read outlives every one
// of its chunk items by construction (it only advances once chunks_remaining hits zero).
typedef struct {
    read_state_t *read;
    int chunk_idx;
} chunk_item_t;

// One modbase chunk of a read queued for the mod runner stage.
typedef struct {
    read_state_t *read;
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
    BoundedQueue<read_state_t *> *stitch_q;

    // modbase stages (only used when mod)
    BoundedQueue<read_state_t *> *mod_pre_q;
    BoundedQueue<mod_chunk_item_t> *mod_chunk_q;
    BoundedQueue<read_state_t *> *mod_post_q;
    BoundedQueue<read_state_t *> *out_q;

    uint64_t total_reads;
    uint64_t total_bytes;
} pipeline_ctx_t;

// pthread entry points take a single void*, so the two runner stages -- the only ones needing more
// than the context -- get their device index through this.
typedef struct {
    pipeline_ctx_t *ctx;
    int runner_idx;
} runner_arg_t;

// pthread_create returns an errno value directly rather than setting errno, so NEG_CHK cannot see it.
static void spawn(pthread_t *tid, void *(*fn)(void *), void *arg) {
    int ret = pthread_create(tid, NULL, fn, arg);
    if (ret != 0) {
        ERROR("Could not create pipeline thread: %s", strerror(ret));
        exit(EXIT_FAILURE);
    }
}

static void join_all(std::vector<pthread_t> &tids) {
    for (size_t i = 0; i < tids.size(); ++i) {
        pthread_join(tids[i], NULL);
    }
}

// The batch path processes (and emits) exactly the reads that carry signal.
static inline bool read_has_signal(const read_state_t *rs) {
    return rs->rec->len_raw_signal > 0;
}

// read raw recs, single thread
static void *loader_stage(void *arg) {
    pipeline_ctx_t *ctx = (pipeline_ctx_t *)arg;
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

    return NULL;
}

// parse the record, then scale the signal and split it into overlapping chunks
static void *preprocess_stage(void *arg) {
    pipeline_ctx_t *ctx = (pipeline_ctx_t *)arg;
    core_t *core = ctx->core;
    raw_rec_t raw;

    while (ctx->read_q->pop(raw)) {
        read_state_t *rs = new read_state_t();
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
            ctx->stitch_q->push(rs);
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

    return NULL;
}

// pack chunks to gpu_batch_size across reads and run inference+decode
// one thread per GPU
static void *runner_stage(void *arg) {
    pipeline_ctx_t *ctx = ((runner_arg_t *)arg)->ctx;
    const int runner_idx = ((runner_arg_t *)arg)->runner_idx;
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

    return NULL;
}

// stitch reads
static void *stitch_stage(void *arg) {
    pipeline_ctx_t *ctx = (pipeline_ctx_t *)arg;
    core_t *core = ctx->core;
    const bool rna = is_rna(core->model_config->sample_type);
    read_state_t *rs = NULL;

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
            ctx->mod_pre_q->push(rs);
        } else {
            ctx->out_q->push(rs);
        }
    }

    return NULL;
}

// mod preproccess: build the seq->signal mapping and modbase chunks; emit one item per mod chunk.
static void *mod_preprocess_stage(void *arg) {
    pipeline_ctx_t *ctx = (pipeline_ctx_t *)arg;
    core_t *core = ctx->core;
    read_state_t *rs = NULL;

    while (ctx->mod_pre_q->pop(rs)) {
        preprocess_modbase(core, rs->rec, rs->read_dat, rs->sequence.c_str(), rs->moves, rs->mod_chunks);

        if (rs->mod_chunks.empty()) {
            // no motif hits: still postprocess to emit (empty) MM/ML tags, matching the batch path
            ctx->mod_post_q->push(rs);
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

    return NULL;
}

// mod runner: pack mod chunks to mod_gpu_batch_size across reads and run the modbase model
static void *mod_runner_stage(void *arg) {
    pipeline_ctx_t *ctx = ((runner_arg_t *)arg)->ctx;
    const int runner_idx = ((runner_arg_t *)arg)->runner_idx;
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

    return NULL;
}

// mod postprocess: turn base_mod_probs into MM/ML tags
static void *mod_postprocess_stage(void *arg) {
    pipeline_ctx_t *ctx = (pipeline_ctx_t *)arg;
    core_t *core = ctx->core;
    read_state_t *rs = NULL;

    while (ctx->mod_post_q->pop(rs)) {
        postprocess_modbase(core, rs->read_dat, rs->mod_string, rs->mod_prob);
        ctx->out_q->push(rs);
    }

    return NULL;
}

// write output, single thread
static void *writer_stage(void *arg) {
    pipeline_ctx_t *ctx = (pipeline_ctx_t *)arg;
    core_t *core = ctx->core;
    const bool sam = (core->opt.flag & SLORADO_SAM) != 0;
    read_state_t *rs = NULL;
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
        delete rs;
    }

    ctx->total_reads = n;

    return NULL;
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
    BoundedQueue<read_state_t *> stitch_q(stitch_cap);
    BoundedQueue<read_state_t *> mod_pre_q(stitch_cap);
    BoundedQueue<mod_chunk_item_t> mod_chunk_q(mod_chunk_cap);
    BoundedQueue<read_state_t *> mod_post_q(out_cap);
    BoundedQueue<read_state_t *> out_q(out_cap);

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

    // The runner args must outlive their threads; they are joined below, before this returns.
    std::vector<runner_arg_t> runner_args(n_runners);
    std::vector<runner_arg_t> mod_runner_args(n_mod_runners);
    for (int i = 0; i < n_runners; ++i) { runner_args[i].ctx = &ctx; runner_args[i].runner_idx = i; }
    for (int i = 0; i < n_mod_runners; ++i) { mod_runner_args[i].ctx = &ctx; mod_runner_args[i].runner_idx = i; }

    pthread_t loader, writer;
    std::vector<pthread_t> preproc(n_pre), runners(n_runners), stitch(n_stitch);
    std::vector<pthread_t> mod_preproc, mod_runners, mod_postproc;

    spawn(&writer, writer_stage, &ctx);

    if (mod) {
        mod_postproc.resize(n_stitch);
        mod_runners.resize(n_mod_runners);
        mod_preproc.resize(n_pre);
        for (int i = 0; i < n_stitch; ++i) spawn(&mod_postproc[i], mod_postprocess_stage, &ctx);
        for (int i = 0; i < n_mod_runners; ++i) spawn(&mod_runners[i], mod_runner_stage, &mod_runner_args[i]);
        for (int i = 0; i < n_pre; ++i) spawn(&mod_preproc[i], mod_preprocess_stage, &ctx);
    }

    for (int i = 0; i < n_stitch; ++i) spawn(&stitch[i], stitch_stage, &ctx);
    for (int i = 0; i < n_runners; ++i) spawn(&runners[i], runner_stage, &runner_args[i]);
    for (int i = 0; i < n_pre; ++i) spawn(&preproc[i], preprocess_stage, &ctx);
    spawn(&loader, loader_stage, &ctx);

    // closing each queue only after its producers are done preserves full pipeline overlap during steady state
    pthread_join(loader, NULL);
    join_all(preproc);
    chunk_q.close();
    join_all(runners);
    stitch_q.close();
    join_all(stitch);
    if (mod) {
        mod_pre_q.close();
        join_all(mod_preproc);
        mod_chunk_q.close();
        join_all(mod_runners);
        mod_post_q.close();
        join_all(mod_postproc);
    }
    out_q.close();
    pthread_join(writer, NULL);

    core->total_reads = (int64_t)ctx.total_reads;
    core->sum_bytes = (int64_t)ctx.total_bytes;
}
