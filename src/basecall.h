/* @file basecall.h
**
** methods for base calling step
** @@
******************************************************************************/

#ifndef BASECALL
#define BASECALL

#include "slorado.h"
#include "torchbox.h"   // at::Tensor

void basecall_db(core_t* core, db_t* db);

// Run inference + decode on a packed batch of chunk pointers using the given runner.
// Exposed for the streaming pipeline (basecall.cpp defines it).
void basecall_chunks(const core_t* core, const int runner_idx, const std::vector<basecall_chunk_t *> &chunks);

// Streaming pipeline: inference and decode split into separate stages so decode of one batch
// overlaps inference of the next. basecall_infer accepts + runs the model, returning scores in
// [T, N, C] layout on the runner's device; basecall_decode consumes those scores.
at::Tensor basecall_infer(const core_t* core, const int runner_idx, const std::vector<basecall_chunk_t *> &chunks);
void basecall_decode(const core_t* core, const int runner_idx, at::Tensor scores_TNC, const std::vector<basecall_chunk_t *> &chunks);

#endif