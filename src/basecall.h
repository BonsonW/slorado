/* @file basecall.h
**
** methods for base calling step
** @@
******************************************************************************/

#ifndef BASECALL
#define BASECALL

#include <ATen/core/TensorBody.h>

#include "slorado.h"

void basecall_db(core_t* core, db_t* db);

// Run inference + decode on a packed batch of chunk pointers using the given runner.
// Exposed for the streaming pipeline (basecall.cpp defines it).
void basecall_chunks(const core_t* core, const int runner_idx, const std::vector<basecall_chunk_t *> &chunks);

// --- cpu-beam decode split (--cpu-beam); see basecall.cpp for the mechanism. ---
// Pool of managed/unified score buffers; when passed, int8 scores are written into managed memory so
// the GPU scan and CPU beam share one buffer with no device->host copy. Defined in basecall.cpp.
class ManagedScorePool;

// Runner-thread half: accept + infer (+ optional int8 quant), returns valid [N,T,C] scores (managed
// int8 when `pool` is given and the model is a clamp model, else device fp16 / device int8).
at::Tensor basecall_infer_scores(const core_t* core, const int runner_idx, const std::vector<basecall_chunk_t *> &chunks, ManagedScorePool* pool = nullptr);

#ifdef USE_GPU
// Decode-thread halves: GPU posterior scan into gpubuf, then CPU beam + writeback into the chunks.
void basecall_scan_gpu(const core_t* core, openfish_gpubuf_t *gpubuf, const at::Tensor &dev_scores);
void basecall_beam_cpu(const core_t* core, openfish_gpubuf_t *gpubuf, const at::Tensor &host_scores, const std::vector<basecall_chunk_t *> &chunks);

// Own a managed score-buffer pool without seeing its definition (it lives in basecall.cpp).
ManagedScorePool* create_score_pool();
void destroy_score_pool(ManagedScorePool* pool);
#endif

#endif