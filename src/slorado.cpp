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

#include <slow5/slow5.h>
#include <openfish/openfish.h>

#include <sys/wait.h>
#include <unistd.h>
#include <vector>

void init_runners(core_t* core, opt_t *opt, char *model);
void free_runners(core_t *core);
void init_elephant(db_t *db);
void free_elephant(db_t *db);
void preprocess_signal(core_t* core, db_t* db, int32_t i);
void stitch_chunks(std::vector<Chunk *> &chunks, std::string &sequence, std::string &qstring);

/* initialise the core data structure */
core_t* init_core(char *slow5file, opt_t opt, char *model, double realtime0) {
    core_t* core = (core_t*)malloc(sizeof(core_t));
    MALLOC_CHK(core);

    core->realtime0 = realtime0;

    core->time_init_runners = 0;
    core->time_sync = 0;
    core->time_load_db = 0;
    core->time_process_db = 0;
    core->time_preproc = 0;
    core->time_basecall = 0;
    core->time_postproc = 0;
    core->time_output = 0;
    core->time_parse = 0;

    core->sp = slow5_open(slow5file, "r");
    if (core->sp == NULL) {
        VERBOSE("Error opening SLOW5 file %s\n", slow5file);
        exit(EXIT_FAILURE);
    }

    core->time_init_runners -= realtime();
    init_runners(core, &opt, model);
    LOG_DEBUG("%s", "successfully initialized runners");

    core->time_init_runners += realtime();

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
    delete core->runner_stats;
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


    db->chunks = new std::vector<std::vector<Chunk *>>(db->capacity_rec, std::vector<Chunk *>());

    init_elephant(db);
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
        std::vector<Chunk *> chunks = (*db->chunks)[i];

        std::string sequence;
        std::string qstring;
        stitch_chunks(chunks, sequence, qstring);

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
    core->time_basecall += (b-a);
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
        for (Chunk *chunk: (*db->chunks)[i]) delete chunk;
        (*db->chunks)[i].clear();
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
    delete db->chunks;
    delete db->sequence;
    delete db->qstring;
    free_elephant(db);
    free(db);
}

/* initialise user specified options */
void init_opt(opt_t* opt) {
    memset(opt, 0, sizeof(opt_t));
    opt->batch_size = 2000;
    opt->gpu_batch_size = 800;
    opt->batch_size_bytes = 200*1000*1000;
    opt->num_thread = 8;

    opt->debug_break = -1;

    opt->device = "cuda:0";
    opt->chunk_size = 8000;
    opt->overlap = 150;
    opt->num_runners = 1;

    opt->out = stdout;

    opt->flag |= SLORADO_EFQ;
}

