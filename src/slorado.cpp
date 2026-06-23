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

#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "slorado.h"
#include "misc.h"
#include "error.h"
#include "calib.h"
#include "quant.h"
#include "sensitivity.h"

#include "basecall.h"
#include "writer.h"

#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <unordered_set>

void init_runners(core_t* core, opt_t *opt, char *model);
void free_runners(core_t *core);
void preprocess_signal(core_t* core, db_t* db, int32_t i);
void stitch_chunks(db_t *basecall_db, size_t i, std::string &sequence, std::string &qstring, std::vector<uint8_t> &moves, size_t len_raw_signal, int model_stride);
void free_read_dat(read_dat_t *read_dat);
void preprocess_modbase(core_t *core, db_t *db, int32_t i);
void postprocess_modbase(core_t *core, db_t *db, int32_t i);

static size_t estimate_bytes_per_read(const char *slow5file) {
    slow5_file_t *sp = slow5_open(slow5file, "r");
    if (!sp) return 0;

    size_t total_bytes = 0;
    int n = 0;
    char *mem = NULL;
    size_t bytes = 0;

    while (n < BATCH_SIZE_SAMPLE_READS) {
        if (slow5_get_next_bytes(&mem, &bytes, sp) < 0) break;
        total_bytes += bytes;
        free(mem);
        mem = NULL;
        n++;
    }

    slow5_close(sp);
    return n > 0 ? total_bytes / n : 0;
}

/* initialise the core data structure */
core_t* init_core(char *slow5file, opt_t opt, char *model, double realtime0) {
    core_t* core = (core_t*)calloc(1, sizeof(core_t));
    MALLOC_CHK(core);
    core->opt = opt;

    if (opt.batch_size == 0) {
        size_t avg_bytes = estimate_bytes_per_read(slow5file);
        core->opt.batch_size = avg_bytes > 0
            ? (int32_t)(opt.batch_size_bytes / avg_bytes)
            : DEFAULT_BATCH_SIZE;
        if (core->opt.batch_size < 1) core->opt.batch_size = 1;
    }

    core->realtime0 = realtime0;

    core->sp = slow5_open(slow5file, "r");
    if (core->sp == NULL) {
        VERBOSE("Error opening SLOW5 file %s\n", slow5file);
        exit(EXIT_FAILURE);
    }

    // modbase stuff
    if (opt.mod != NULL) {
        INFO("%s", "modification calling detected, output will be in SAM format");
        core->opt.flag |= SLORADO_SAM;
        
        LOG_TRACE("%s", "loading modbase configs...");
        auto model_str = std::string(model);
        if (model_str.back() == '/') {
            model_str.pop_back(); // remove trailing slash if exists
        }
        auto modbase_config_path = model_str + "_" + opt.mod;
        ModBaseModelConfig modbase_config = load_modbase_model_config(modbase_config_path.c_str());
        auto configs = std::vector<ModBaseModelConfig>({modbase_config});
        ModBaseInfo modbase_info = get_modbase_info(configs);
        core->modbase_config = new ModBaseModelConfig(modbase_config);
        core->modbase_info = new ModBaseInfo(modbase_info);
        LOG_TRACE("%s", "modbase config loaded");
    }

    CRFModelConfig model_config;
    if (is_tx_model_config(model)) {
        model_config = load_tx_model_config(model);
    } else {
        model_config = load_lstm_model_config(model);
    }
    model_config.model_path = std::string(model);
    model_config.sample_type = get_sample_type_from_model_name(model_config.model_path);

    core->model_stride = static_cast<size_t>(model_config.stride);

    size_t resolved_chunk_size = opt.chunk_size > 0 ? opt.chunk_size
                                 : model_config.chunk_size > 0 ? (size_t)model_config.chunk_size
                                 : DEFAULT_CHUNK_SIZE;
    int32_t resolved_overlap = opt.overlap > 0 ? opt.overlap
                               : model_config.overlap > 0 ? model_config.overlap
                               : DEFAULT_OVERLAP;

    core->chunk_size = resolved_chunk_size - (resolved_chunk_size % core->model_stride);
    core->opt.overlap = resolved_overlap - (resolved_overlap % (int32_t)core->model_stride);
    // Overlap must be strictly less than chunk_size (model default can exceed a user-specified -c).
    if ((size_t)core->opt.overlap >= core->chunk_size) {
        core->opt.overlap = (int32_t)(core->chunk_size - (int32_t)core->model_stride);
    }

    core->decoder_opts = openfish_decoder_default_opts();
    core->decoder_opts.q_shift = model_config.qbias;
    core->decoder_opts.q_scale = model_config.qscale;

    core->model_config = new CRFModelConfig(model_config);
    LOG_TRACE("%s", "model config loaded");

    if (opt.calibrate_out != NULL) {
        core->calib_stats = new calib_stats_t();
        fprintf(stderr, "[init_core] calibration mode enabled, stats will be written to %s\n", opt.calibrate_out);
    }

    if (opt.quant_config_path != NULL) {
        core->quant_config = new std::unordered_map<std::string, std::string>(
            load_quant_config(std::string(opt.quant_config_path)));
    }

    if (opt.sensitivity_out != NULL) {
        if (core->quant_config == nullptr) {
            fprintf(stderr, "[init_core] warning: --sensitivity requires --quant-config; no quant layers will differ from fp16\n");
        }
        core->sensitivity_stats = new sensitivity_stats_t();
        fprintf(stderr, "[init_core] sensitivity mode enabled, KL stats will be written to %s\n", opt.sensitivity_out);
    }

    core->time_init_runners -= realtime();
    init_runners(core, &opt, model);
    core->opt.gpu_batch_size = opt.gpu_batch_size; // sync auto-detected value back to core
    core->time_init_runners += realtime();
    LOG_DEBUG("%s", "successfully initialized runners");

    core->sum_bytes=0;
    core->total_reads=0; // total number mapped entries in the bam file (after filtering based on flags, mapq etc)

    return core;
}


/* free the core data structure */
void free_core(core_t* core, opt_t opt) {
    if (core->calib_stats != nullptr && opt.calibrate_out != nullptr) {
        core->calib_stats->save_json(std::string(opt.calibrate_out));
        delete core->calib_stats;
        core->calib_stats = nullptr;
    }

    if (core->sensitivity_stats != nullptr && opt.sensitivity_out != nullptr) {
        core->sensitivity_stats->save_tsv(std::string(opt.sensitivity_out), opt.quant_config_path);
        delete core->sensitivity_stats;
        core->sensitivity_stats = nullptr;
    }

    if (core->quant_config != nullptr) {
        delete core->quant_config;
        core->quant_config = nullptr;
    }

    free_runners(core);

    slow5_close(core->sp);
    delete core->runners;
    delete core->mod_runners;
    delete core->runner_stats;
    delete core->model_config;

    if (core->modbase_config != NULL) {
        delete core->modbase_config;
        delete core->modbase_info;
    }
    free(core);
}

/* initialise a data batch */
db_t* init_db(core_t* core) {
    db_t* db = (db_t*)(malloc(sizeof(db_t)));
    MALLOC_CHK(db);

    db->capacity_rec = core->opt.batch_size;
    db->n_rec = 0;

    db->mem_records = (char **)(calloc(db->capacity_rec, sizeof(char *)));
    MALLOC_CHK(db->mem_records);
    db->mem_bytes = (size_t *)(calloc(db->capacity_rec, sizeof(size_t)));
    MALLOC_CHK(db->mem_bytes);

    db->slow5_rec = (slow5_rec_t**)calloc(db->capacity_rec,sizeof(slow5_rec_t*));
    MALLOC_CHK(db->slow5_rec);

    db->means = (double*)calloc(db->capacity_rec,sizeof(double));
    MALLOC_CHK(db->means);

    db->sequence = new std::vector<std::string>(db->capacity_rec);
    db->qstring = new std::vector<std::string>(db->capacity_rec);
    db->read_dats = new std::vector<read_dat_t *>(db->capacity_rec, NULL);
    db->basecall_chunks = new std::vector<std::vector<basecall_chunk_t>>(db->capacity_rec, std::vector<basecall_chunk_t>());
    db->mod_chunks = new std::vector<std::vector<mod_chunk_t>>(db->capacity_rec, std::vector<mod_chunk_t>());
    db->moves = new std::vector<std::vector<uint8_t>>(db->capacity_rec, std::vector<uint8_t>());

    db->mod_string = new std::vector<std::string>(db->capacity_rec);
    db->mod_prob = new std::vector<std::vector<uint8_t>>(db->capacity_rec, std::vector<uint8_t>());

    db->total_reads = 0;
    db->sum_bytes = 0;

    return db;
}

/* load a data batch from disk */
ret_status_t load_db(core_t* core, db_t* db) {
    double load_start = realtime();

    db->n_rec = 0;
    db->sum_bytes = 0;
    db->total_reads = 0;

    ret_status_t status = {0, 0};
    int32_t i = 0;
    while (db->n_rec < db->capacity_rec && db->sum_bytes<core->opt.batch_size_bytes) {
        i=db->n_rec;

        if (slow5_get_next_bytes(&db->mem_records[i], &db->mem_bytes[i], core->sp) < 0) {
            if (slow5_errno != SLOW5_ERR_EOF) {
                ERROR("Error reading from SLOW5 file %d", slow5_errno);
                exit(EXIT_FAILURE);
            } else {
                break;
            }
        } else {
            db->n_rec++;
            db->total_reads++; // candidate read
            db->sum_bytes += db->mem_bytes[i];
        }
    }

    status.num_reads=db->n_rec;
    status.num_bytes=db->sum_bytes;

    double load_end = realtime();
    core->time_load_db += (load_end-load_start);

    return status;
}

void parse_single(core_t* core,db_t* db, int32_t i) {
    assert(db->mem_bytes[i] > 0);
    assert(db->mem_records[i] != NULL);

    int ret = slow5_decode(&db->mem_records[i], &db->mem_bytes[i], &db->slow5_rec[i], core->sp);
    if (ret < 0) {
        ERROR("Error parsing the record %d", i);
        exit(EXIT_FAILURE);
    }
}

void postprocess_signal(core_t* core, db_t* db, int32_t i) {
    slow5_rec_t* rec = db->slow5_rec[i];
    uint64_t len_raw_signal = rec->len_raw_signal;

    if (len_raw_signal > 0) {
        auto& sequence = (*db->sequence)[i];
        sequence.clear();
        auto& qstring = (*db->qstring)[i];
        qstring.clear();
        auto& moves = (*db->moves)[i];
        moves.clear();

        stitch_chunks(db, i, sequence, qstring, moves, len_raw_signal, core->model_stride);
        
        if (is_rna(core->model_config->sample_type)) {
            std::reverse(sequence.begin(), sequence.end());
            std::reverse(qstring.begin(), qstring.end());
            std::reverse(moves.begin(), moves.end()); // might not need this, no idea
        }

    }
}

void process_db(core_t* core, db_t* db) {
    double proc_start = realtime();
    double a, b;

    a = realtime();
    work_db(core, db, parse_single);
    b = realtime();
    core->time_parse += (b - a);
    LOG_DEBUG("%s", "parsed reads");

    a = realtime();
    work_db(core, db, preprocess_signal);
    b = realtime();
    core->time_preproc += (b-a);
    LOG_DEBUG("%s", "preprocessed reads");

    a = realtime();
    basecall_db(core, db);
    b = realtime();
    core->time_runners += (b-a);
    LOG_DEBUG("%s", "basecalled reads");

    a = realtime();
    work_db(core, db, postprocess_signal);
    b = realtime();
    core->time_postproc += (b-a);
    LOG_DEBUG("%s", "postprocessed reads");

    if (core->opt.mod != NULL) {
        a = realtime();
        work_db(core, db, preprocess_modbase);
        b = realtime();
        core->time_preproc_mod += (b-a);
        LOG_DEBUG("%s", "mod preprocessed reads");

        a = realtime();
        mod_basecall_db(core, db);
        b = realtime();
        core->time_runners += (b-a);
        LOG_DEBUG("%s", "mod basecalled reads");

        a = realtime();
        work_db(core, db, postprocess_modbase);
        b = realtime();
        core->time_postproc_mod += (b-a);
        LOG_DEBUG("%s", "mod postprocessed reads");
    }

    double proc_end = realtime();
    core->time_process_db += (proc_end-proc_start);
}

/* write the output for a processed data batch */
void output_db(core_t* core, db_t* db) {
    double output_start = realtime();

    int32_t i = 0;
    for (i = 0; i < db->n_rec; i++) {
        if (db->slow5_rec[i]->len_raw_signal > 0) {
            if ((core->opt.flag & SLORADO_SAM) != 0) {
                write_to_file_sam(core->opt.out, (*db->sequence)[i].c_str(), (*db->qstring)[i].c_str(), db->slow5_rec[i]->read_id, (*db->mod_string)[i].c_str(), (*db->mod_prob)[i]);
            } else {
                write_to_file_fastq(core->opt.out, (*db->sequence)[i].c_str(), (*db->qstring)[i].c_str(), db->slow5_rec[i]->read_id);
            }
        }
    }

    core->sum_bytes += db->sum_bytes;
    core->total_reads += db->total_reads;

    double output_end = realtime();
    core->time_output += (output_end-output_start);
}

/* partially free a data batch - only the read dependent allocations are freed */
void free_db_tmp(db_t* db) {
    LOG_DEBUG("%s", "freeing db_tmp");
    int32_t i = 0;
    for (i = 0; i < db->n_rec; ++i) {
        free(db->mem_records[i]);
        db->mem_records[i] = NULL;
    }
}

/* completely free a data batch */
void free_db(db_t* db) {
    LOG_DEBUG("%s", "freeing db");
    int32_t i = 0;
    for (i = 0; i < db->capacity_rec; ++i) {
        free_read_dat((*db->read_dats)[i]);
        slow5_rec_free(db->slow5_rec[i]);
    }
    free(db->slow5_rec);
    free(db->mem_records);
    free(db->mem_bytes);
    free(db->means);
    delete db->sequence;
    delete db->qstring;
    delete db->moves;
    delete db->mod_string;
    delete db->mod_prob;
    delete db->basecall_chunks;
    delete db->mod_chunks;
    delete db->read_dats;
    free(db);
}

/* initialise user specified options */
void init_opt(opt_t* opt) {
    memset(opt, 0, sizeof(opt_t));
    opt->gpu_batch_size = 0; // 0 = auto
    opt->batch_size_bytes = 512*1000*1000;
    opt->num_thread = 8;

    opt->debug_break = -1;

#ifdef USE_GPU
    opt->device = "cuda:all";
#else
    opt->device = "cpu";
#endif

    opt->out = stdout;

    opt->mod = NULL;

    // opt->flag |= SLORADO_SAM;
}
