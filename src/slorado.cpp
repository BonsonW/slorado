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

#include "dorado/signal_prep.h"
#include "basecall.h"
#include "writer.h"
#include "dorado/utils/stitch.h"

#include <slow5/slow5.h>

#include <sys/wait.h>
#include <unistd.h>
#include <vector>

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

    core->runners = new std::vector<runner_t*>();
    core->runner_stats = new std::vector<runner_stat_t*>();

    core->time_init_runners -= realtime();

    if (strcmp(opt.device, "cpu") == 0) {
        std::string device = opt.device;
        for (int i = 0; i < opt.num_runners; ++i) {
            core->runner_stats->push_back((runner_stat_t*)malloc(sizeof(runner_stat_t)));
            init_runner_stat((*core->runner_stats).back());

            core->runners->push_back(new runner_t());
            init_runner((*core->runners).back(), model, device, opt.chunk_size, opt.gpu_batch_size, torch::kF32);
        }
    } else {
#ifdef USE_GPU
        std::vector<std::string> devices;
        std::string device_args = std::string(opt.device);
        devices = parse_cuda_device_string(device_args);

        for (auto device: devices) {
            for (int i = 0; i < opt.num_runners; ++i) {
                core->runner_stats->push_back((runner_stat_t*)malloc(sizeof(runner_stat_t)));
                init_runner_stat((*core->runner_stats).back());
                core->runners->push_back(new runner_t());
                init_runner((*core->runners).back(), model, device, opt.chunk_size, opt.gpu_batch_size, torch::kF16);
            }
        }
#else
        fprintf(stderr, "Error. Please compile again for GPU\n");
        exit(EXIT_FAILURE);
#endif
    }

    LOG_DEBUG("%s", "successfully initialized runners");

    auto adjusted_chunk_size = core->runners->front()->chunk_size;
    if (opt.chunk_size != adjusted_chunk_size) {
        LOG_DEBUG("Adjusting chunk size to %zu", adjusted_chunk_size);
        opt.chunk_size = adjusted_chunk_size;
    }

    core->time_init_runners += realtime();

    core->sum_bytes=0;
    core->total_reads=0; // total number mapped entries in the bam file (after filtering based on flags, mapq etc)

    core->opt = opt;

#ifdef HAVE_ACC
    if (core->opt.flag & SLORADO_ACC) {
        VERBOSE("%s","Initialising accelator");
    }
#endif

    return core;
}

/* free the core data structure */
void free_core(core_t* core, opt_t opt) {
#ifdef HAVE_ACC
    if (core->opt.flag & SLORADO_ACC) {
        VERBOSE("%s","Freeing accelator");
    }
#endif

    for (size_t i = 0; i < core->runner_stats->size(); ++i) {
        free((*core->runner_stats)[i]);
    }

    for (size_t i = 0; i < core->runners->size(); ++i) {
        delete (*core->runners)[i];
    }

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
    db->tensors = new std::vector<std::vector<torch::Tensor>>(db->capacity_rec, std::vector<torch::Tensor>());
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

#define TO_PICOAMPS(RAW_VAL,DIGITISATION,OFFSET,RANGE) (((RAW_VAL)+(OFFSET))*((RANGE)/(DIGITISATION)))

void mean_single(core_t* core,db_t* db, int32_t i) {
    slow5_rec_t* rec = db->slow5_rec[i];
    uint64_t len_raw_signal = rec->len_raw_signal;

    if (len_raw_signal > 0) {
        double sum = 0;
        for (uint64_t i = 0; i < len_raw_signal; i++){
            double pA = TO_PICOAMPS(rec->raw_signal[i], rec->digitisation, rec->offset, rec->range);
            sum += pA;
        }
        double mean = sum / len_raw_signal;
        db->means[i] = mean;
    }
}

void preprocess_signal(core_t* core, db_t* db, int32_t i) {
    slow5_rec_t* rec = db->slow5_rec[i];
    uint64_t len_raw_signal = rec->len_raw_signal;
    opt_t opt = core->opt;

    if (len_raw_signal > 0) {
        torch::Tensor signal = tensor_from_record(rec).to(torch::kCPU);

        scale_signal(signal, rec->range / rec->digitisation, rec->offset);

        std::vector<Chunk *> chunks = chunks_from_tensor(signal, opt.chunk_size, opt.overlap);

        (*db->chunks)[i] = chunks;
        LOG_TRACE("%s","assigned chunks");

        std::vector<torch::Tensor> tensors = tensor_as_chunks(signal, chunks, opt.chunk_size);

        (*db->tensors)[i] = tensors;
        LOG_TRACE("%s","assigned tensors");
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
    delete db->tensors;
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

#ifdef HAVE_ACC
    opt->flag |= SLORADO_ACC;
#endif
}

/* initialise runner_stat */
void init_runner_stat(runner_stat_t* time_stamps) {
    memset(time_stamps, 0, sizeof(runner_stat_t));
}

/* initialise runners */
void init_runner(
    runner_t* runner,
    const std::string &model_path,
    const std::string &device,
    int chunk_size,
    int batch_size,
    torch::ScalarType dtype
) {
    LOG_TRACE("initializing model runner for device %s", device.c_str());

    const auto model_config = load_crf_model_config(model_path);
    LOG_TRACE("%s", "model config loaded");

    runner->model_stride = static_cast<size_t>(model_config.stride);

    runner->decoder_opts = DECODER_INIT;
    runner->decoder_opts.q_shift = model_config.qbias;
    runner->decoder_opts.q_scale = model_config.qscale;

    runner->device = device;
    runner->model_config = model_config;

    runner->tensor_opts = torch::TensorOptions().dtype(dtype).device(device);
    runner->module = load_crf_model(model_path, model_config, batch_size, chunk_size, runner->tensor_opts);
    LOG_TRACE("%s", "model loaded");

    chunk_size -= chunk_size % runner->model_stride;
    runner->input_tensor = torch::zeros({batch_size, 1, chunk_size}, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    runner->chunk_size = runner->input_tensor.size(2);

    int64_t device_idx = device[device.size()-1] - '0'; // quick and dirty device index extraction
    runner->device_idx = device_idx;

    LOG_DEBUG("fully initialized model runner for device %s", device.c_str());
}