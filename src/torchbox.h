#ifndef TORCHBOX_H
#define TORCHBOX_H

#include <torch/torch.h>
#include <cstdint>
#include "slorado.h"

// per read data
struct read_dat {
    torch::Tensor scaled_signal;

    // Number of samples trimmed from the FRONT of the raw signal by the basecall scaler
    // (RNA adapter / DNA pore-open trim). The modbase path must trim the raw signal by the
    // same amount so its signal stays aligned with the moves/sequence (which are derived from
    // the trimmed basecall signal). Set in preprocess_signal, consumed in preprocess_modbase.
    int64_t basecall_trim_start = 0;

    // mod data
    const char *seq;

    std::vector<int8_t> encoded_kmers;
    std::array<std::vector<int64_t>, 4> per_base_hits_seq; // sequence indices for hits for each base (i.e. one per model)
    std::array<std::vector<int64_t>, 4> per_base_hits_sig; // signal indices for hits for each base (i.e. one per model)
    int64_t target_start;

    std::vector<uint8_t> base_mod_probs;
    std::vector<bool> base_mod_simplex_motif_hits;
};

struct runner {
    std::string device;
    torch::Tensor input_tensor;
    torch::TensorOptions tensor_opts;
    torch::nn::ModuleHolder<torch::nn::AnyModule> module{nullptr};
#ifdef USE_GPU
    int64_t device_idx;
    openfish_gpubuf_t *gpubuf;
#endif

    // modbase stuff
    at::Tensor input_sigs;
    at::Tensor input_seqs;
};

void preprocess_signal(core_t *core, slow5_rec_t *rec, read_dat_t *read_dat, std::vector<basecall_chunk_t> &chunks);
void preprocess_modbase(core_t *core, slow5_rec_t *rec, read_dat_t *read_dat, const char *seq, std::vector<uint8_t> &moves, std::vector<mod_chunk_t> &mod_chunks);
void postprocess_modbase(core_t *core, read_dat_t *read_dat, std::string &mod_string_out, std::vector<uint8_t> &mod_prob_out);

void preprocess_signal_db(core_t *core, db_t *db, int32_t i);
void preprocess_modbase_db(core_t *core, db_t *db, int32_t i);
void postprocess_modbase_db(core_t *core, db_t *db, int32_t i);

void free_read_dat(read_dat_t *read_dat);

#endif