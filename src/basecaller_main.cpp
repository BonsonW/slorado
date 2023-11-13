/**
 * @file basecaller_main.cpp
 * @brief entry point to basecaller_main
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

#include "globals.h"
#include "slorado.h"
#include "dorado/signal_prep.h"
#include "misc.h"

#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <getopt.h>
#include <memory>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void generateSplitBar(const long* values, const std::string* names, int size);


static struct option long_options[] = {
    {"threads", required_argument, 0, 't'},         //0 number of threads [8]
    {"batchsize", required_argument, 0, 'K'},       //1 batchsize - number of reads loaded at once [1000]
    {"max-bytes", required_argument, 0, 'B'},       //2 batchsize - number of bytes loaded at once
    {"verbose", required_argument, 0, 'v'},         //3 verbosity level [1]
    {"help", no_argument, 0, 'h'},                  //4
    {"version", no_argument, 0, 'V'},               //5
    {"output", required_argument, 0, 'o'},          //6 output to a file [stdout]
    {"debug-break", required_argument, 0, 0},       //7 break after processing the first batch (used for debugging)
    {"profile-cpu", required_argument, 0, 0},       //8 perform section by section (used for profiling - for CPU only)
    {"accel",required_argument, 0, 0},              //9 accelerator
    {"chunk-size", required_argument, 0, 'c'},      //10 chunk size [8000]
    {"overlap", required_argument, 0, 'p'},         //11 overlap [150]
    {"device", required_argument, 0, 'x'},          //12 device [cpu]
    {"num-runners", required_argument, 0, 'r'},     //13 number of runners [1]
    {"emit-fastq", required_argument, 0, 0},        //14 toggles emit fastq
    {"gpu_batchsize", required_argument, 0, 'C'},   //15 gpu batchsize - number of chunks loaded at once [512]
    {0, 0, 0, 0}};


static inline void print_help_msg(FILE *fp_help, opt_t opt){
    fprintf(fp_help, "usage: slorado basecaller [model] [data]\n");
    fprintf(fp_help, "positional arguments:\n");
    fprintf(fp_help, "  model FILE                  the basecaller model to run.\n");
    fprintf(fp_help, "  data FILE                   the data directory.\n");
    fprintf(fp_help, "\nbasic options:\n");
    fprintf(fp_help, "  -t INT                      number of processing threads [%d]\n", opt.num_thread);
    fprintf(fp_help, "  -K INT                      batch size (max number of reads loaded at once) [%d]\n", opt.batch_size);
    fprintf(fp_help, "  -C INT                      gpu batch size (max number of chunks loaded at once) [%d]\n", opt.gpu_batch_size);
    fprintf(fp_help, "  -B FLOAT[K/M/G]             max number of bytes loaded at once [%.1fM]\n", opt.batch_size_bytes/(float)(1000*1000));
    fprintf(fp_help, "  -o FILE                     output to file [%s]\n", opt.out_path);
    fprintf(fp_help, "  -c INT                      chunk size [%d]\n", opt.chunk_size);
    fprintf(fp_help, "  -p INT                      overlap [%d]\n", opt.overlap);
    fprintf(fp_help, "  -x DEVICE                   specify device [%s]\n", opt.device);
    // fprintf(fp_help, "  -r INT                      number of runners [%d]\n", opt.num_runners);
    fprintf(fp_help, "  -h                          shows help message and exits\n");
    fprintf(fp_help, "  --verbose INT               verbosity level [%d]\n",(int)get_log_level());
    fprintf(fp_help, "  --version                   print version\n");
    fprintf(fp_help, "\nadvanced options:\n");
    fprintf(fp_help, "  --debug-break INT           break after processing the specified no. of batches\n");
    // fprintf(fp_help, "  --emit-fastq=yes|no         emits fastq output format\n");
    fprintf(fp_help, "  --profile-cpu=yes|no        process section by section (used for profiling on CPU)\n");
#ifdef HAVE_ACC
    // fprintf(fp_help,"   --accel=yes|no             Running on accelerator [%s]\n",(opt.flag&SLORADO_ACC?"yes":"no"));
#endif
}

int basecaller_main(int argc, char* argv[]) {
    double realtime0 = realtime();

    const char* optstring = "t:B:K:C:v:o:x:r:p:c:hV";

    int longindex = 0;
    int32_t c = -1;

    char *data = NULL;
    char *model = NULL;

    FILE *fp_help = stderr;

    opt_t opt;
    init_opt(&opt); //initialise options to defaults

    //parse the user args
    while ((c = getopt_long(argc, argv, optstring, long_options, &longindex)) >= 0) {
        if (c == 'B') {
            opt.batch_size_bytes = mm_parse_num(optarg);
            if(opt.batch_size_bytes<=0){
                ERROR("%s","Maximum number of bytes should be larger than 0.");
                exit(EXIT_FAILURE);
            }
        } else if (c == 'K') {
            opt.batch_size = atoi(optarg);
            if (opt.batch_size < 1) {
                ERROR("Batch size should larger than 0. You entered %d",opt.batch_size);
                exit(EXIT_FAILURE);
            }
        } else if (c == 'C') {
            opt.gpu_batch_size = atoi(optarg);
            if (opt.gpu_batch_size < 1) {
                ERROR("Batch size should larger than 0. You entered %d",opt.gpu_batch_size);
                exit(EXIT_FAILURE);
            }
        } else if (c == 't') {
            opt.num_thread = atoi(optarg);
            if (opt.num_thread < 1) {
                ERROR("Number of threads should larger than 0. You entered %d", opt.num_thread);
                exit(EXIT_FAILURE);
            }
        } else if (c == 'v') {
            int v = atoi(optarg);
            set_log_level((enum log_level_opt)v);
        } else if (c == 'x') {
            opt.device = optarg;
        } else if (c == 'c') {
            opt.chunk_size = atoi(optarg);
            if (opt.chunk_size < 1) {
                ERROR("Chunk size should larger than 0. You entered %d", opt.chunk_size);
                exit(EXIT_FAILURE);
            }
        } else if (c == 'p') {
            opt.overlap = atoi(optarg);
            if (opt.overlap < 1) {
                ERROR("Overlap should larger than 0. You entered %d", opt.overlap);
                exit(EXIT_FAILURE);
            }
        } else if (c == 'o') {
            opt.out_path = optarg;
            opt.out = fopen(opt.out_path, "w");
            if (opt.out == NULL) {
                fprintf(stderr,"Error in opening output file\n");
                exit(EXIT_FAILURE);
            }
        } else if (c == 'r') {
            opt.num_runners = atoi(optarg);
            if (opt.num_runners < 1) {
                ERROR("Number of runners should larger than 0. You entered %d", opt.num_runners);
                exit(EXIT_FAILURE);
            }
        } else if (c == 'V') {
            fprintf(stdout,"slorado %s\n",SLORADO_VERSION);
            exit(EXIT_SUCCESS);
        } else if (c == 'h'){
            fp_help = stdout;
        } else if(c == 0 && longindex == 7) { //debug break
            opt.debug_break = atoi(optarg);
        } else if(c == 0 && longindex == 8) { //sectional benchmark todo : warning for gpu mode
            yes_or_no(&opt.flag, SLORADO_PRF, long_options[longindex].name, optarg, 1);
        } else if(c == 0 && longindex == 9) { //accel
        #ifdef HAVE_ACC
            yes_or_no(&opt.flag, SLORADO_ACC, long_options[longindex].name, optarg, 1);
        #else
            WARNING("%s", "--accel has no effect when compiled for the CPU");
        #endif
        } else if(c == 0 && longindex == 14) { //sectional benchmark todo : warning for gpu mode
            yes_or_no(&opt.flag, SLORADO_EFQ, long_options[longindex].name, optarg, 1);
        }
    }

    // Incorrect number of arguments given
    if (argc - optind != 2 || fp_help == stdout) {
        print_help_msg(fp_help, opt);
        if(fp_help == stdout){
            exit(EXIT_SUCCESS);
        }
        exit(EXIT_FAILURE);
    }

    model = argv[optind++];

    if (model == NULL) {
        print_help_msg(fp_help, opt);
        if(fp_help == stdout){
            exit(EXIT_SUCCESS);
        }
        exit(EXIT_FAILURE);
    }

    data = argv[optind];

    if (data == NULL) {
        print_help_msg(fp_help, opt);
        if(fp_help == stdout){
            exit(EXIT_SUCCESS);
        }
        exit(EXIT_FAILURE);
    }

    // print summary
    fprintf(stderr,"\nslorado base-caller version %s\n", SLORADO_VERSION);
    fprintf(stderr,"model path:         %s\n", model);
    fprintf(stderr,"input path:         %s\n", data);
    fprintf(stderr,"output path:        %s\n", opt.out_path == NULL ? "stdout" : opt.out_path);
    fprintf(stderr,"device:             %s\n", opt.device);
    fprintf(stderr,"chunk size:         %d\n", opt.chunk_size);
    fprintf(stderr,"batch size:         %d\n", opt.batch_size);
    fprintf(stderr,"gpu batch size:     %d\n", opt.gpu_batch_size);
    fprintf(stderr,"no. threads:        %d\n", opt.num_thread);
    fprintf(stderr,"no. runners:        %d\n", opt.num_runners);
    fprintf(stderr,"overlap:            %d\n", opt.overlap);
    fprintf(stderr, "\n");

    //initialise the core data structure
    core_t* core = init_core(data, opt, model, realtime0);

    int32_t counter=0;

    //initialise a databatch
    db_t* db = init_db(core);

    ret_status_t status = {core->opt.batch_size,core->opt.batch_size_bytes};

    while (status.num_reads >= core->opt.batch_size || status.num_bytes>=core->opt.batch_size_bytes) {
        //load a databatch
        status = load_db(core, db);

        fprintf(stderr, "[%s::%.3f*%.2f] %d Entries (%.1fM bytes) loaded\n", __func__,
                realtime() - realtime0, cputime() / (realtime() - realtime0),
                status.num_reads,status.num_bytes/(1000.0*1000.0));
        //process a databatch
        process_db(core, db);

        fprintf(stderr, "[%s::%.3f*%.2f] %d Entries (%.1fM bytes) processed\n", __func__,
                realtime() - realtime0, cputime() / (realtime() - realtime0),
                status.num_reads,status.num_bytes/(1000.0*1000.0));

        //output print
        output_db(core, db);

        //free temporary
        free_db_tmp(db);

        if(opt.debug_break==counter){
            break;
        }
        counter++;
    }

    //free the databatch
    free_db(db);

    fprintf(stderr, "[%s] total entries: %ld", __func__,(long)core->total_reads);
    fprintf(stderr,"\n[%s] total bytes: %.1f M",__func__,core->sum_bytes/(float)(1000*1000));
    double total_time = core->ts.time_init_runners + core->load_db_time + core->process_db_time + core->output_time;
    fprintf(stderr, "\n[%s] Model initialization time: %.3f sec : %.2f %", __func__,core->ts.time_init_runners,core->ts.time_init_runners * 100 / total_time);
    fprintf(stderr, "\n[%s] Data loading time: %.3f sec : %.2f %", __func__,core->load_db_time,core->load_db_time*100/total_time);
    fprintf(stderr, "\n[%s] Data processing time: %.3f sec : %.2f %", __func__,core->process_db_time,core->process_db_time*100/total_time);
    //if((core->opt.flag&SLORADO_PRF)|| core->opt.flag & SLORADO_ACC){
            fprintf(stderr, "\n[%s]     - Parse time: %.3f sec",__func__, core->parse_time);
            fprintf(stderr, "\n[%s]     - Preprocess time: %.3f sec",__func__, core->preproc_time);
            fprintf(stderr, "\n[%s]     - Basecall+decode time: %.3f sec",__func__, core->basecall_time);
            fprintf(stderr, "\n[%s]          - Synchronisation time: %.3f sec",__func__, core->ts.time_sync);

    auto runner_ts = *core->runner_ts;

    for (size_t i = 0; i < runner_ts.size(); ++i) {
        fprintf(stderr, "\n[%s]          - Model Runner [%zu] time: %.3f",__func__, i, runner_ts[i]->time_basecall + runner_ts[i]->time_decode + runner_ts[i]->time_accept);
        fprintf(stderr, "\n[%s]             - Accept time: %.3f sec",__func__, runner_ts[i]->time_accept);
        fprintf(stderr, "\n[%s]             - Decode time: %.3f sec",__func__, runner_ts[i]->time_decode);
        fprintf(stderr, "\n[%s]                - beam_search time: %.3f sec",__func__, beam_searchT);
        fprintf(stderr, "\n\n[%s]              - cudaLSTM time: %.3f sec",__func__, cudaLSTM);            fprintf(stderr, "\n\n[%s]                     - Forward in ConvolutionImplT time: %.3f sec",__func__, convolutionImplT);
        fprintf(stderr, "\n[%s]                       - Forward in LinearCRFImpl time: %.3f sec",__func__, forward_l159);
        fprintf(stderr, "\n[%s]                       - Forward in LSTMStackImpl time: %.3f sec",__func__, forward_l536);
        fprintf(stderr, "\n\n[%s]                     - cudaLSTMImplT time: %.3f sec",__func__, cudaLSTMImplT);
        fprintf(stderr, "\n\n[%s]                     - cudaLSTMStackImplT time: %.3f sec",__func__, cudaLSTMStackImplT);
        fprintf(stderr, "\n\n[%s]                         - forward_cublas() time: %.3f sec",__func__, forward_cublasT);
        fprintf(stderr, "\n[%s]                           - host_transpose_f16() time: %.3f sec",__func__, host_transpose_f16T);
        fprintf(stderr, "\n[%s]                           - Iterate over RNN layers time: %.3f sec",__func__, rnnIterate);
        fprintf(stderr, "\n[%s]                             - Initialize Tensor time: %.3f sec",__func__, state_bufT); 
        fprintf(stderr, "\n[%s]                             - Transpose weights time: %.3f sec",__func__, weights_cpuT); 
        fprintf(stderr, "\n[%s]                                 - No of times called contiguous(): %.0f",__func__, weightCPUcalls);
        fprintf(stderr, "\n[%s]                                 - No of times Beam search    : %.0f",__func__, cont);
        fprintf(stderr, "\n[%s]                             - Transfer weights to GPU time: %.3f sec",__func__, weightsT);
        fprintf(stderr, "\n[%s]                             - Transfer bias to GPU time: %.3f sec",__func__, biasT);   
        fprintf(stderr, "\n[%s]                             - matmul_f16()  time: %.3f sec",__func__, matmul_f16T);
        fprintf(stderr, "\n\n[%s]                 - CudaCaller time: %.3f sec",__func__, CudaCallerT);
        fprintf(stderr, "\n[%s]                 - load_crf_model time: %.3f sec",__func__, CudaCallerT5);
        fprintf(stderr, "\n[%s]                         - Initialize crf_model time: %.3f sec",__func__, load_crf_modelT);
        fprintf(stderr, "\n[%s]                         - Populate crf_model time: %.3f sec",__func__, CudaCallerT5 - load_crf_modelT);
        fprintf(stderr, "\n[%s]              - call_chunks time: %.3f sec",__func__, call_chunksT);
        fprintf(stderr, "\n[%s]                 - NNTask while loop time: %.3f sec",__func__, NNTaskT2);
        fprintf(stderr, "\n[%s]                 -  cuda_thread_fnT time: %.3f sec",__func__, cuda_thread_fnT2);
        fprintf(stderr, "\n[%s]                  -  forward time: %.3f sec",__func__, cuda_thread_fnT5);
        fprintf(stderr, "\n[%s]                  -  synchronize time: %.3f sec",__func__, cuda_thread_fnT6);
        fprintf(stderr, "\n\n[%s]               - cublasGemmEx time: %.3f sec",__func__, cublasGemmExT);        
        fprintf(stderr, "\n\n[%s]     - Matmul count: %f ",__func__, matMul);
        fprintf(stderr, "\n\n[%s]     - Postprocess time: %.3f sec",__func__, core->postproc_time);
        fprintf(stderr, "\n[%s] Data output time: %.3f sec", __func__,core->output_time);
        fprintf(stderr, "\n[%s] Data output time: %.3f sec : %.2f %\n", __func__,core->output_time,core->output_time*100/total_time);
        fprintf(stderr,"\n");
        
        free_core(core,opt);

        if (opt.out != stdout) {
            fclose(opt.out);
        }

    return 0;
    }
}