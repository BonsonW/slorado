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

#if defined(HAVE_METAL)
// scan-on-GPU / beam-on-CPU split: infer produces device int8 scores, basecall_scan_gpu runs the
// forward/backward scan on the GPU into gpubuf and returns a host copy of the scores, then
// basecall_beam_host runs the CPU beam search over gpubuf's posteriors.
void basecall_scan_gpu(const core_t* core, const int runner_idx, at::Tensor dev_scores, openfish_gpubuf_t *gpubuf);
at::Tensor basecall_finalize_host(const core_t* core, at::Tensor dev_scores);
// Split infer for double-buffering: conv+LSTM only, then CRF+quant deferred.
at::Tensor basecall_infer_lstm_host(const core_t* core, const int runner_idx, const std::vector<basecall_chunk_t *> &chunks);
at::Tensor basecall_crf_quant(const core_t* core, const int runner_idx, at::Tensor lstm_out);
void basecall_beam_host(const core_t* core, const int runner_idx, at::Tensor host_scores, openfish_gpubuf_t *gpubuf, const std::vector<basecall_chunk_t *> &chunks);
// TEMPORARY (benchmark, SLORADO_GPU_BEAM): full GPU decode (scan+beam+qual+gen on GPU) as a drop-in
// alternative to basecall_scan_gpu + basecall_beam_host, to A/B the beam search on GPU vs CPU.
void basecall_decode_gpu_full(const core_t* core, const int runner_idx, at::Tensor dev_scores, openfish_gpubuf_t *gpubuf, const std::vector<basecall_chunk_t *> &chunks);
#endif

#endif