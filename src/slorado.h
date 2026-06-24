/**
 * @file slorado.c
 * @brief common functions for slorado
 * @author Hasindu Gamaarachchi (hasindu@unsw.edu.au)
 * @author Bonson Wong (bonson.ym@gmail.com)

MIT License

Copyright (c) 2019 Hasindu Gamaarachchi (hasindu@unsw.edu.au)
Copyright (c) 2023 Bonson Wong (bonson.ym@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


******************************************************************************/

#ifndef SLORADO_H
#define SLORADO_H

#include <stdlib.h>
#include <stdint.h>
#include <slow5/slow5.h>
#include <openfish/openfish.h>
#include <unordered_map>
#include <vector>
#include <string>

#include "dorado/model_config.h"
#include "quant.h"

// Forward declarations — full definitions in calib.h (requires torch headers).
struct calib_stats_t;
struct calib_layer_t;
struct sensitivity_stats_t;

#define SLORADO_VERSION "0.5.0-beta"

/*******************************************************
 * flags related to the user specified options (opt_t) *
 *******************************************************/

#define SLORADO_PRF         0x001 // cpu-profile mode
#define SLORADO_ACC         0x002 // accelerator enable
#define SLORADO_SAM         0x004 // emit sam enable
#define SLORADO_FLASH       0x008 // flash attention enable

#define WORK_STEAL 1 // simple work stealing enabled or not (no work stealing mean no load balancing)
#define STEAL_THRESH 1 // stealing threshold

#define NUM_BASES (4)

#define DEFAULT_CHUNK_SIZE (10000)
#define DEFAULT_OVERLAP (500)
#define DEFAULT_BATCH_SIZE (4096)
#define DEFAULT_GPU_BATCH_SIZE (512)
#define BATCH_SIZE_SAMPLE_READS (256)

/* user specified options */
typedef struct {
    uint64_t flag;              // flags
    int32_t batch_size;         // max reads loaded at once: K
    int32_t gpu_batch_size;     // max chunks loaded at once: C
    int64_t batch_size_bytes;   // max bytes loaded at once: B

    int32_t num_thread;         // number of threads used: t
    int32_t debug_break;

    const char *out_path;       // path to output file: o
    FILE *out;

    const char *device;         // specified device: x
    size_t chunk_size;          // size of chunks: c
    int32_t overlap;            // overlap: p

    const char *mod;         // specified modbase: x
    const char *calibrate_out;   // path for calibration JSON output (NULL = disabled)
    const char *quant_config_path; // path for per-layer quant config JSON (NULL = disabled)
    const char *sensitivity_out;   // path for sensitivity KL output JSON (NULL = disabled)
} opt_t;

typedef struct read_dat read_dat_t;

// result + metadata of a chunk
struct basecall_chunk {
    size_t input_offset;    // raw signal offset
    size_t idx_in_read;     // order in read
    size_t raw_chunk_size;  // size in raw signal

    std::string seq;
    std::string qstring;
    std::vector<uint8_t> moves;

    read_dat_t *read_dat;
};

// result + metadata of a modbase chunk
struct mod_chunk {
    read_dat_t *read_dat;

    int model_id;
    int base_id;

    size_t signal_offset;   // starting offset of the preprocessed signal
    size_t hit_offset;      // starting offset of the context hits

    int64_t num_states;   // number of states predicted by the modbase model `num_mods + 1`
    
    std::vector<float> scores;  // model predictions for this chunk arranged in `[canonical, mod1, .., modN, canonical, mod1, ..]`
};
typedef struct mod_chunk mod_chunk_t;

typedef struct basecall_chunk basecall_chunk_t;

/* a batch of read data (dynamic data based on the reads) */
typedef struct {
    int32_t n_rec;
    int32_t capacity_rec;

    char **mem_records;
    size_t *mem_bytes;

    slow5_rec_t **slow5_rec;

    double *means;

    // intermediate data
    std::vector<std::vector<basecall_chunk_t>> *basecall_chunks;
    std::vector<std::vector<mod_chunk_t>> *mod_chunks;
    std::vector<read_dat_t *> *read_dats;

    // basecall results
    std::vector<std::string> *sequence;
    std::vector<std::string> *qstring;
    std::vector<std::vector<uint8_t>> *moves;

    // modcall results
    std::vector<std::string> *mod_string;
    std::vector<std::vector<uint8_t>> *mod_prob;

    // stats
    int64_t sum_bytes;
    int64_t total_reads; // total number mapped entries in the bam file (after filtering based on flags, mapq etc)
} db_t;

typedef struct {
    double time_conv_stack;
    double time_rnns;
    double time_crf_1;
    double time_crf_2;
    double time_clamp;
    // FLSTM timings (accumulated across all layers)
    double time_flstm_precompute;   // batched ih = x @ W_ih_fused + bias_ih (before loop)
    double time_flstm_recurrence;   // full per-step loop: addmm(W_hh_fused) + gate update

    calib_stats_t *calib_stats = nullptr;
    const std::unordered_map<std::string, std::string> *quant_config = nullptr;
    std::unordered_map<std::string, layer_quant_t> quant_methods;
} lstm_stats_t;

typedef struct {
    double time_conv_stack;
    double time_tx_encoder;
    double time_tx_decoder;
    double time_crf;

    double time_self_attn;
    double time_norm1;
    double time_ff;
    double time_norm2;

    double time_mm;
    double time_rotary_emb;
    double time_sdp_attn;
    double time_out_proj;

    calib_stats_t *calib_stats = nullptr;
    const std::unordered_map<std::string, std::string> *quant_config = nullptr;
    std::unordered_map<std::string, layer_quant_t> quant_methods;
    bool use_flash = false;
    int nthreads = 1;
} tx_stats_t;

/* time stamps */
typedef struct {
    double time_accept;
    double time_basecall;
    double time_infer;
    double time_decode;
    double time_modcall;

    void *model_stats;

    uint64_t total_dp;
} runner_stat_t;

typedef struct runner runner_t;

/* core data structure (mostly static data throughout the program lifetime) */
typedef struct {
    // slow5
    slow5_file_t *sp;

    // options
    opt_t opt;
    openfish_opt_t decoder_opts;
    CRFModelConfig *model_config;
    ModBaseModelConfig *modbase_config = NULL;
    ModBaseInfo *modbase_info = NULL;
    size_t model_stride;
    size_t chunk_size;

    // create model runner
    // only one per GPU is used for now
    std::vector<runner_t *> *runners;
    std::vector<runner_t *> *mod_runners;

    // realtime0
    double realtime0;

    // timings
    double time_init_runners;
    double time_load_db;
    double time_process_db;
    double time_free_db;
    double time_parse;
    double time_preproc;
    double time_runners;
    double time_sync;
    double time_postproc;
    double time_preproc_mod;
    double time_postproc_mod;
    double time_output;

    double time_tens_from_rec;
    double time_init_base_mod_probs;
    double time_seq_to_sig_map;
    double time_seq_to_ints;
    double time_populate_hits_sig;
    double time_populate_signal;
    double time_get_minimal_encoding_skips;
    double time_populate_encoded_kmer;

    // stats for each runner
    std::vector<runner_stat_t *> *runner_stats;

    // stats, set by output_db
    int64_t sum_bytes;
    int64_t total_reads; // total number mapped entries in the bam file (after filtering based on flags, mapq etc)

    // calibration stats (NULL unless --calibrate is set)
    calib_stats_t *calib_stats = nullptr;

    // quantization config (NULL unless --quant-config is set)
    std::unordered_map<std::string, std::string> *quant_config = nullptr;

    // sensitivity stats (NULL unless --sensitivity is set)
    sensitivity_stats_t *sensitivity_stats = nullptr;
} core_t;

/* argument wrapper for the multithreaded framework used for data processing */
typedef struct {
    core_t* core;
    db_t* db;
    int32_t starti;
    int32_t endi;
    void (*func)(core_t*, db_t*, int);
    int32_t thread_index;
#ifdef WORK_STEAL
    void *all_pthread_args;
#endif
#ifdef HAVE_CUDA
    int32_t *ultra_long_reads; // reads that are assigned to the CPU due to the unsuitability to process on the GPU
    double ret1; // return value
#endif
} pthread_arg_t;

/* return status by the load_db - used for termination when all the data is processed */
typedef struct {
    int32_t num_reads;
    int64_t num_bytes;
} ret_status_t;

/******************************************
 * function prototype for major functions *
 ******************************************/

/* initialise user specified options */
void init_opt(opt_t* opt);

/* initialise the core data structure */
core_t* init_core(char *slow5file, opt_t opt, char *model, double realtime0);

/* initialise a data batch */
db_t* init_db(core_t* core);

/* load a data batch from disk */
ret_status_t load_db(core_t* dg, db_t* db);

void work_per_single_read(core_t* core, db_t* db, int32_t i);
/* process all reads in the given batch db */
void work_db(core_t* core, db_t* db, void (*func)(core_t*, db_t*, int));

/* process a data batch */
void process_db(core_t* core, db_t* db);

/* align a single read specified by index i*/
void process_single(core_t* core, db_t* db, int32_t i);

/* write the output for a processed data batch */
void output_db(core_t* core, db_t* db);

/* partially free a data batch - only the read dependent allocations are freed */
void free_db_tmp(db_t* db);

/* completely free a data batch */
void free_db(db_t* db);

/* free the core data structure */
void free_core(core_t* core,opt_t opt);

#endif