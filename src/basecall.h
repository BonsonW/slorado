/* @file basecall.h
**
** methods for base calling step
** @@
******************************************************************************/

#ifndef BASECALL
#define BASECALL

#include "slorado.h"

namespace at { class Tensor; }

void basecall_db(core_t* core, db_t* db);

// Run inference + decode on a packed batch of chunk pointers using the given runner.
// Exposed for the streaming pipeline (basecall.cpp defines it).
void basecall_chunks(const core_t* core, const int runner_idx, const std::vector<basecall_chunk_t *> &chunks);

// Split inference/decode for the Metal overlap pipeline: infer produces host-side scores (int8 when
// the model clamps, else fp32) so the GPU is freed, then decode runs on the CPU (overlapping the
// next batch's GPU inference). basecall_infer_host returns [nchunks, T, C] host scores.
at::Tensor basecall_infer_host(const core_t* core, const int runner_idx, const std::vector<basecall_chunk_t *> &chunks);
void basecall_decode_host(const core_t* core, const int runner_idx, at::Tensor host_scores, const std::vector<basecall_chunk_t *> &chunks);

#endif