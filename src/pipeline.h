/* @file pipeline.h
**
** streaming (pipelined) basecalling path: overlaps disk I/O, CPU preprocessing,
** GPU inference and output so the GPU is fed continuously instead of draining
** between read batches. Selected with the --stream CLI flag; the batch path in
** process_db() is left untouched.
** @@
******************************************************************************/

#ifndef PIPELINE_H
#define PIPELINE_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "slorado.h"

// State for a single read as it flows through the pipeline stages. Held by
// std::shared_ptr so ownership is shared between the in-flight chunk items and
// the stage currently working on the read; freed once the writer is done.
// (make_shared value-initialises this, zeroing the scalar/atomic members.)
typedef struct {
    slow5_rec_t *rec;                // read_id, len_raw_signal; freed by writer
    read_dat_t *read_dat;            // scaled_signal etc.; freed by writer

    std::vector<basecall_chunk_t> chunks;
    std::atomic<int> chunks_remaining;

    std::string sequence;
    std::string qstring;
    std::vector<uint8_t> moves;

    uint64_t seq_no;                 // load order (for optional ordered output)
} read_state_t;

// One chunk of a read queued for the runner stage. Holds a shared_ptr so the
// read (and its chunk vector) stays alive across the GPU forward pass.
typedef struct {
    std::shared_ptr<read_state_t> read;
    int chunk_idx;
} chunk_item_t;

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

// Run the whole streaming pipeline to completion over core->sp, writing to core->opt.out.
void run_pipeline(core_t *core);

#endif
