/* @file signal_prep.h
**
** methods for preparing a signal for the base calling step
** @@
******************************************************************************/

#ifndef SIGNAL_PREP_H
#define SIGNAL_PREP_H

#include "torchbox.h"

template <typename T>
T div_round_up(const T a, const T b) {
    return (a + b - 1) / b;
}
template <typename T>
T pad_to(const T a, const T b) {
    return div_round_up(a, b) * b;
}

void scale_signal(core_t *core, torch::Tensor &signal, float scaling, float offset, SignalNormalisationParams &scaling_params);

// Given a read with unstitched chunks, stitch the chunks (accounting for overlap) and assign basecalled read and qstring to Read
// The single-chunk stitch is bounded by the basecalled (front-trimmed) signal length taken from
// read_dat->basecall_trim_start (matching dorado get_raw_data_samples()) for both DNA and RNA.
void stitch_chunks(db_t *basecall_db, size_t i, std::string &sequence, std::string &qstring, std::vector<uint8_t> &moves, size_t len_raw_signal, int model_stride);
void stitch_chunks_vec(std::vector<basecall_chunk_t> &chunks, std::string &sequence, std::string &qstring, std::vector<uint8_t> &moves, size_t len_raw_signal, int model_stride);

// dorado utils::mux_change_trim_read: reads with end_reason mux_change / unblock_mux_change have a
// degraded low-qscore tail; trim it from the stitched sequence, qstring and moves (signal order,
// i.e. before any RNA reversal). No-op for other end reasons or if the file has no end_reason.
void mux_change_trim_read(core_t *core, slow5_rec_t *rec, read_dat_t *read_dat, std::string &sequence, std::string &qstring, std::vector<uint8_t> &moves, int model_stride);

// Load serialised tensor from disk.
std::vector<torch::Tensor> load_tensors(const std::string& dir, const std::vector<std::string>& tensors);

// Computes the q-th quantiles of each row of the input tensor `t`
// using a partial sort as opposed a full sort per torch::quantiles
// Only `interpolation='lower'` is currently implemented.
torch::Tensor quantile(const torch::Tensor t, const torch::Tensor q);

// Computes the q-th quantiles of each row of the input tensor `t`
// using a counting sort which is extremely fast for low range integers.
// Only `interpolation='lower'` is currently implemented.
torch::Tensor quantile_counting(const torch::Tensor t, const torch::Tensor q);

// temporary
inline void module_load_state_dict(torch::nn::Module& module, const std::vector<torch::Tensor>& weights) {
    assert(weights.size() == module.parameters().size());
    for (size_t idx = 0; idx < weights.size(); idx++) {
        module.parameters()[idx].data() = weights[idx].data();
    }
}

#endif