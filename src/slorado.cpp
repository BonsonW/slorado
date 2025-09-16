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

#include "basecall.h"
#include "writer.h"

#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#include <algorithm>

void init_runners(core_t* core, opt_t *opt, char *model);
void free_runners(core_t *core);
void init_chunk_db(db_t *db);
void free_chunk_db(db_t *db);
void preprocess_signal(core_t* core, db_t* db, int32_t i);
void stitch_chunks(chunk_db_t *chunk_db, size_t i, std::string &sequence, std::string &qstring);

/* initialise the core data structure */
core_t* init_core(char *slow5file, opt_t opt, char *model, double realtime0) {
    core_t* core = (core_t*)calloc(1, sizeof(core_t));
    MALLOC_CHK(core);

    core->realtime0 = realtime0;

    core->sp = slow5_open(slow5file, "r");
    if (core->sp == NULL) {
        VERBOSE("Error opening SLOW5 file %s\n", slow5file);
        exit(EXIT_FAILURE);
    }

    // modbase stuff
    auto modbase_type = std::string("5mCG_5hmCG@v2");
    auto modbase_config_path = std::string(model) + "_" + modbase_type;
    ModBaseModelConfig modbase_config = load_modbase_model_config(modbase_config_path.c_str());
    auto configs = std::vector({modbase_config});
    ModBaseInfo modbase_info = get_modbase_info(configs);

    CRFModelConfig model_config;
    if (is_tx_model_config(model)) {
        model_config = load_tx_model_config(model);
    } else {
        model_config = load_lstm_model_config(model);
    }
    model_config.model_path = std::string(model);
    model_config.sample_type = get_sample_type_from_model_name(model_config.model_path);

    core->model_stride = static_cast<size_t>(model_config.stride);
    core->chunk_size = opt.chunk_size - (opt.chunk_size % core->model_stride);

    core->decoder_opts = DECODER_INIT;
    core->decoder_opts.q_shift = model_config.qbias;
    core->decoder_opts.q_scale = model_config.qscale;

    core->model_config = new CRFModelConfig(model_config);
    core->modbase_config = new ModBaseModelConfig(modbase_config);
    core->modbase_info = new ModBaseInfo(modbase_info);
    LOG_TRACE("%s", "model config loaded");

    core->time_init_runners -= realtime();
    init_runners(core, &opt, model);
    core->time_init_runners += realtime();
    LOG_DEBUG("%s", "successfully initialized runners");

    core->sum_bytes=0;
    core->total_reads=0; // total number mapped entries in the bam file (after filtering based on flags, mapq etc)

    core->opt = opt;

    return core;
}


/* free the core data structure */
void free_core(core_t* core, opt_t opt) {
    free_runners(core);

    slow5_close(core->sp);
    delete core->runners;
    delete core->modbase_runners;
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

    init_chunk_db(db);
    db->sequence = new std::vector<char *>(db->capacity_rec, NULL);
    db->qstring = new std::vector<char *>(db->capacity_rec, NULL);

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
        std::string sequence;
        std::string qstring;
        stitch_chunks(db->chunk_db, i, sequence, qstring);
        
        if (is_rna(core->model_config->sample_type)) {
            std::reverse(sequence.begin(), sequence.end());
            std::reverse(qstring.begin(), qstring.end());
        }

        (*db->sequence)[i] = strdup(sequence.c_str());
        assert((*db->sequence)[i] != NULL);

        (*db->qstring)[i] = strdup(qstring.c_str());
        assert((*db->qstring)[i] != NULL);
    }
}

void process_db(core_t* core, db_t* db) {
    double proc_start = realtime();

    double a = realtime();
    work_db(core, db, parse_single);
    double b = realtime();
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

    double proc_end = realtime();
    core->time_process_db += (proc_end-proc_start);
}

/* write the output for a processed data batch */
void output_db(core_t* core, db_t* db) {
    double output_start = realtime();

    int32_t i = 0;
    for (i = 0; i < db->n_rec; i++) {
        if(db->slow5_rec[i]->len_raw_signal>0){
            write_to_file(core->opt.out, (*db->sequence)[i], (*db->qstring)[i], db->slow5_rec[i]->read_id, (core->opt.flag & SLORADO_EFQ) != 0);
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
        free((*db->sequence)[i]);
        free((*db->qstring)[i]);
    }
}

/* completely free a data batch */
void free_db(db_t* db) {
    LOG_DEBUG("%s", "freeing db");
    int32_t i = 0;
    for (i = 0; i < db->capacity_rec; ++i) {
        slow5_rec_free(db->slow5_rec[i]);
    }
    free(db->slow5_rec);
    free(db->mem_records);
    free(db->mem_bytes);
    free(db->means);
    delete db->sequence;
    delete db->qstring;
    free_chunk_db(db);
    free(db);
}

/* initialise user specified options */
void init_opt(opt_t* opt) {
    memset(opt, 0, sizeof(opt_t));
    opt->batch_size = 2000;
    opt->gpu_batch_size = 500;
    opt->batch_size_bytes = 500*1000*1000;
    opt->num_thread = 8;

    opt->debug_break = -1;

#ifdef USE_GPU
    opt->device = "cuda:all";
#else
    opt->device = "cpu";
#endif

    opt->chunk_size = 10000;
    opt->overlap = 150;

    opt->out = stdout;

    opt->flag |= SLORADO_EFQ;
}

