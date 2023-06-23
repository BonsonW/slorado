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

#include "dorado/decode/GPUDecoder.h"
#include "dorado/decode/CPUDecoder.h"

#ifdef USE_GPU
#include "dorado/nn/CudaCRFModel.h"
#endif

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

    core->sp = slow5_open(slow5file,"r");
    if (core->sp == NULL) {
        VERBOSE("Error opening SLOW5 file %s\n",slow5file);
        exit(EXIT_FAILURE);
    }

    init_timestamps(&core->ts);

    core->opt = opt;

    core->runners = new std::vector<Runner>();
    core->runner_ts = new std::vector<timestamps_t *>();

    core->ts.time_init_runners -= realtime();

#ifdef USE_GPU
    if (strcmp(opt.device, "cpu") == 0) {
        for (int i = 0; i < opt.num_runners; ++i) {
            core->runners->push_back(std::make_shared<ModelRunner<CPUDecoder>>(model, opt.device, opt.chunk_size, opt.gpu_batch_size));
            core->runner_ts->push_back((timestamps_t *)malloc(sizeof(timestamps_t)));
            init_timestamps((*core->runner_ts).back());
        }
    } else {
        std::vector<std::string> devices;
        std::string device_name = "";
        std::string device_args = std::string(opt.device);
        std::string delimiter = ":";
        size_t pos = device_args.find(delimiter);
        device_name = device_args.substr(0, pos + delimiter.length());
        device_args.erase(0, pos + delimiter.length());

        delimiter = ",";
        while ((pos = device_args.find(delimiter)) != std::string::npos) {
            devices.push_back(device_name + device_args.substr(0, pos));
            device_args.erase(0, pos + delimiter.length());
        }
        devices.push_back(device_name + device_args.substr(0, pos));

        for (auto device: devices) {
#ifdef USE_CUDA_LSTM
            auto caller = create_cuda_caller(model, opt.chunk_size, opt.gpu_batch_size, device);
#endif
            for (int i = 0; i < opt.num_runners; ++i) {
#ifdef USE_CUDA_LSTM
                core->runners->push_back(std::make_shared<CudaModelRunner>(caller, opt.chunk_size, opt.gpu_batch_size));
#else
                core->runners->push_back(std::make_shared<ModelRunner<GPUDecoder>>(model, device, opt.chunk_size, opt.gpu_batch_size));
#endif
                core->runner_ts->push_back((timestamps_t *)malloc(sizeof(timestamps_t)));
                init_timestamps((*core->runner_ts).back());
            }
        }
    }
#else
    if (strcmp(opt.device, "cpu") == 0) {
        for (int i = 0; i < opt.num_runners; ++i) {
            core->runners->push_back(std::make_shared<ModelRunner<CPUDecoder>>(model, opt.device, opt.chunk_size, opt.gpu_batch_size));
            core->runner_ts->push_back((timestamps_t *)malloc(sizeof(timestamps_t)));
            init_timestamps((*core->runner_ts).back());
        }
    } else {
        fprintf(stderr, "Error. Please compile again for GPU\n");
        exit(EXIT_FAILURE);
    }
#endif

    LOG_DEBUG("%s", "successfully initialized runners");

    core->ts.time_init_runners += realtime();

    //realtime0
    core->realtime0=realtime0;

    core->load_db_time=0;
    core->process_db_time=0;
    core->preproc_time=0;
    core->basecall_time=0;
    core->postproc_time=0;
    core->output_time=0;

    core->sum_bytes=0;
    core->total_reads=0; //total number mapped entries in the bam file (after filtering based on flags, mapq etc)

#ifdef HAVE_ACC
    if (core->opt.flag & SLORADO_ACC) {
        VERBOSE("%s","Initialising accelator");
    }
#endif

    return core;
}

/* free the core data structure */
void free_core(core_t* core,opt_t opt) {
#ifdef HAVE_ACC
    if (core->opt.flag & SLORADO_ACC) {
        VERBOSE("%s","Freeing accelator");
    }
#endif

    slow5_close(core->sp);
    free(core->runners);
    free(core->runner_ts);
    free(core);
}

/* initialise a data batch */
db_t* init_db(core_t* core) {
    db_t* db = (db_t*)(malloc(sizeof(db_t)));
    MALLOC_CHK(db);

    db->capacity_rec = core->opt.batch_size;
    db->n_rec = 0;

    db->mem_records = (char**)(calloc(db->capacity_rec,sizeof(char*)));
    MALLOC_CHK(db->mem_records);
    db->mem_bytes = (size_t*)(calloc(db->capacity_rec,sizeof(size_t)));
    MALLOC_CHK(db->mem_bytes);

    db->slow5_rec = (slow5_rec_t**)calloc(db->capacity_rec,sizeof(slow5_rec_t*));
    MALLOC_CHK(db->slow5_rec);

    db->means = (double*)calloc(db->capacity_rec,sizeof(double));
    MALLOC_CHK(db->means);

    db->chunks = new std::vector<std::vector<Chunk *>>(db->capacity_rec, std::vector<Chunk *>());
    db->tensors = new std::vector<std::vector<torch::Tensor>>(db->capacity_rec, std::vector<torch::Tensor>());
    db->sequence = new std::vector<char *>(db->capacity_rec, NULL);
    db->qstring = new std::vector<char *>(db->capacity_rec, NULL);

    db->total_reads=0;
    db->sum_bytes=0;

    return db;
}

/* load a data batch from disk */
ret_status_t load_db(core_t* core, db_t* db) {
    double load_start = realtime();

    db->n_rec = 0;
    db->sum_bytes = 0;
    db->total_reads = 0;

    ret_status_t status={0,0};
    int32_t i = 0;
    while (db->n_rec < db->capacity_rec && db->sum_bytes<core->opt.batch_size_bytes) {
        i=db->n_rec;

        if (slow5_get_next_bytes(&db->mem_records[i],&db->mem_bytes[i],core->sp) < 0){
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
    core->load_db_time += (load_end-load_start);

    return status;
}

void parse_single(core_t* core,db_t* db, int32_t i){
    assert(db->mem_bytes[i] > 0);
    assert(db->mem_records[i] != NULL);

    int ret=slow5_decode(&db->mem_records[i], &db->mem_bytes[i], &db->slow5_rec[i], core->sp);
    if(ret<0){
        ERROR("Error parsing the record %d",i);
        exit(EXIT_FAILURE);
    }
}

#define TO_PICOAMPS(RAW_VAL,DIGITISATION,OFFSET,RANGE) (((RAW_VAL)+(OFFSET))*((RANGE)/(DIGITISATION)))

void mean_single(core_t* core,db_t* db, int32_t i){
    slow5_rec_t* rec = db->slow5_rec[i];
    uint64_t len_raw_signal = rec->len_raw_signal;

    if(len_raw_signal>0){
        double sum = 0;
        for(uint64_t i=0;i<len_raw_signal;i++){
            double pA = TO_PICOAMPS(rec->raw_signal[i],rec->digitisation,rec->offset,rec->range);
            sum += pA;
        }
        double mean = sum/len_raw_signal;
        db->means[i]=mean;
    }
}

void preprocess_signal(core_t* core,db_t* db, int32_t i){
    slow5_rec_t* rec = db->slow5_rec[i];
    uint64_t len_raw_signal = rec->len_raw_signal;
    opt_t opt = core->opt;

    if (len_raw_signal > 0) {
        torch::Tensor signal = tensor_from_record(rec).to(torch::kCPU);

        scale_signal(signal, rec->range / rec->digitisation, rec->offset);

        std::vector<Chunk *> chunks = chunks_from_tensor(signal, opt.chunk_size, opt.overlap);

        (*db->chunks)[i] = chunks;
        LOG_DEBUG("%s","assigned chunks");

        std::vector<torch::Tensor> tensors = tensor_as_chunks(signal, chunks, opt.chunk_size);

        (*db->tensors)[i] = tensors;
        LOG_DEBUG("%s","assigned tensors");
    }
}

void basecall_db(core_t* core, db_t* db) {
    timestamps_t *ts = &(core->ts);

    size_t num_threads = (*core->runners).size();
    size_t n_reads = (*db->chunks).size();

    std::vector<std::unique_ptr<std::thread>> threads;
    threads.reserve(num_threads);

    size_t reads_per_thread = (n_reads + num_threads - 1) / num_threads;

    size_t start = 0;
    size_t end = reads_per_thread;

    bool last = false;
    for (size_t runner = 0; runner < (*core->runners).size(); ++runner) {
        threads.emplace_back(
            new std::thread(
                basecall_thread,
                core,
                db,
                runner,
                start,
                end
            )
        );
        start = end;
        end = std::min(end + reads_per_thread, n_reads);

        if (last) break;
        if (end == n_reads) last = true;
    }

    auto time_sync = 0;

    for (size_t i = 0; i < threads.size(); ++i) {
        threads[i]->join();
        if (i == 0) {
            time_sync -= realtime();
        }
        if (i == threads.size()-1) {
            time_sync += realtime();
        }
    }
    ts->time_sync += time_sync;
}


void postprocess_signal(core_t* core,db_t* db, int32_t i){
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

void process_db(core_t* core,db_t* db){
    double proc_start = realtime();

    double a = realtime();
    work_db(core,db,parse_single);
    double b = realtime();
    core->parse_time += (b-a);
    LOG_DEBUG("%s","Parsed reads");  

    a = realtime();
    work_db(core,db,preprocess_signal);
    b = realtime();
    core->preproc_time += (b-a);
    LOG_DEBUG("%s","Preprocessed reads");
    
    a = realtime();
    basecall_db(core,db);
    b = realtime();
    core->basecall_time += (b-a);
    LOG_DEBUG("%s","Basecalled reads");
    
    a = realtime();
    work_db(core,db,postprocess_signal);
    b = realtime();
    core->postproc_time += (b-a);
    LOG_DEBUG("%s","Postprocessed reads");
    
    double proc_end = realtime();
    core->process_db_time += (proc_end-proc_start);
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

    //core->read_index = core->read_index + db->n_rec;
    double output_end = realtime();
    core->output_time += (output_end-output_start);
}

/* partially free a data batch - only the read dependent allocations are freed */
void free_db_tmp(db_t* db) {
    int32_t i = 0;
    for (i = 0; i < db->n_rec; ++i) {
        free(db->mem_records[i]);
        free((*db->sequence)[i]);
        free((*db->qstring)[i]);
    }
}

/* completely free a data batch */
void free_db(db_t* db) {
    int32_t i = 0;
    for (i = 0; i < db->capacity_rec; ++i) {
        slow5_rec_free(db->slow5_rec[i]);
        for (Chunk *chunk: (*db->chunks)[i]) free(chunk);
    }
    free(db->slow5_rec);
    free(db->mem_records);
    free(db->mem_bytes);
    free(db->means);
    free(db->chunks);
    free(db->sequence);
    free(db->qstring);
    free(db->tensors);
    free(db);
}

/* initialise user specified options */
void init_opt(opt_t* opt) {
    memset(opt, 0, sizeof(opt_t));
    opt->batch_size = 2000;
    opt->gpu_batch_size = 800;
    opt->batch_size_bytes = 20*1000*1000;
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

/* initialise timestamps */
void init_timestamps(timestamps_t* time_stamps) {
    memset(time_stamps, 0, sizeof(timestamps_t));

    time_stamps->time_init_runners = 0;
    time_stamps->time_read = 0;
    time_stamps->time_tens = 0;
    time_stamps->time_trim = 0;
    time_stamps->time_scale = 0;
    time_stamps->time_chunk = 0;
    time_stamps->time_copy = 0;
    time_stamps->time_pad = 0;
    time_stamps->time_assign = 0;
    time_stamps->time_accept = 0;
    time_stamps->time_basecall = 0;
    time_stamps->time_decode = 0;
    time_stamps->time_sync = 0;
    time_stamps->time_stitch = 0;
    time_stamps->time_write = 0;
    time_stamps->time_total = 0;
    time_stamps->time_beam_search_emplace = 0;
    // time_stamps->time_forward = 0;
    // time_forward = 0;
    // forward_l62 = 0;
    // forward_l159 = 0;
    // forward_l469 = 0;
    // forward_l536 = 0;
    // forward_l577 = 0;
    // forward_l642 = 0;
}

