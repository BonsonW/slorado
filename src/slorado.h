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
#include <vector>
#include <memory>
#include "dorado/nn/ModelRunner.h"
#include "dorado/Chunk.h"

#define SLORADO_VERSION "0.1.0"

/*******************************************************
 * flags related to the user specified options (opt_t) *
 *******************************************************/

#define SLORADO_PRF 0x001 //cpu-profile mode
#define SLORADO_ACC 0x002 //accelerator enable
#define SLORADO_EFQ 0x004 //emit fastq enable

#define WORK_STEAL 1 //simple work stealing enabled or not (no work stealing mean no load balancing)
#define STEAL_THRESH 1 //stealing threshold

/* user specified options */
typedef struct {
    uint64_t flag;              //flags
    int32_t batch_size;         //max reads loaded at once: K
    int32_t gpu_batch_size;     //max chunks loaded at once: C
    int64_t batch_size_bytes;   //max bytes loaded at once: B

    int32_t num_thread;         //number of threads used: t
    int32_t debug_break;

    const char *out_path;       //path to output file: o
    FILE *out;

    const char *device;         //specified device: x
    int32_t chunk_size;         //size of chunks: c
    int32_t overlap;            //overlap: p
    int32_t num_runners;       //number of runners: r
} opt_t;


/* a batch of read data (dynamic data based on the reads) */
typedef struct {
    int32_t n_rec;
    int32_t capacity_rec;

    char **mem_records;
    size_t *mem_bytes;

    slow5_rec_t **slow5_rec;

    double *means;

    std::vector<std::vector<Chunk *>> *chunks;
    std::vector<std::vector<torch::Tensor>> *tensors;

    std::vector<char *> *sequence;
    std::vector<char *> *qstring;

    //stats
    int64_t sum_bytes;
    int64_t total_reads; //total number mapped entries in the bam file (after filtering based on flags, mapq etc)
} db_t;

/* time stamps */
typedef struct {
    double_t time_init_runners;
    double_t time_read;
    double_t time_tens;
    double_t time_trim;
    double_t time_scale;
    double_t time_chunk;
    double_t time_copy;
    double_t time_pad;
    double_t time_assign;
    double_t time_accept;
    double_t time_basecall;
    double_t time_decode;
    double_t time_stitch;
    double_t time_sync;
    double_t time_write;
    double_t time_total;
    double_t time_beam_search_emplace;

} timestamps_t;

/* core data structure (mostly static data throughout the program lifetime) */
typedef struct {
    //slow5
    slow5_file_t *sp;

    // options
    opt_t opt;

    // create model runner
    // only one is used for now
    std::vector<Runner> *runners;

    //realtime0
    double realtime0;

    double load_db_time;
    double process_db_time;
    double parse_time;
    double preproc_time;
    double basecall_time;
    double postproc_time;
    double output_time;
    timestamps_t ts;
    std::vector<timestamps_t *> *runner_ts;

    //stats //set by output_db
    int64_t sum_bytes;
    int64_t total_reads; //total number mapped entries in the bam file (after filtering based on flags, mapq etc)
} core_t;

/* argument wrapper for the multithreaded framework used for data processing */
typedef struct {
    core_t* core;
    db_t* db;
    int32_t starti;
    int32_t endi;
    void (*func)(core_t*,db_t*,int);
    int32_t thread_index;
#ifdef WORK_STEAL
    void *all_pthread_args;
#endif
#ifdef HAVE_CUDA
    int32_t *ultra_long_reads; //reads that are assigned to the CPU due to the unsuitability to process on the GPU
    double ret1;    //return value
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

void work_per_single_read(core_t* core,db_t* db, int32_t i);
/* process all reads in the given batch db */
void work_db(core_t* core, db_t* db, void (*func)(core_t*,db_t*,int));

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

void init_timestamps(timestamps_t* time_stamps);

#endif
