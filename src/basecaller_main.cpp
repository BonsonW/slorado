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

//std::string generateSplitBar(const long* values, int size);   ////////////////////////////////
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
    fprintf(fp_help, "  -r INT                      number of runners [%d]\n", opt.num_runners);
    fprintf(fp_help, "  -h                          shows help message and exits\n");
    fprintf(fp_help, "  --verbose INT               verbosity level [%d]\n",(int)get_log_level());
    fprintf(fp_help, "  --version                   print version\n");
    fprintf(fp_help, "\nadvanced options:\n");
    fprintf(fp_help, "  --debug-break INT           break after processing the specified no. of batches\n");
    fprintf(fp_help, "  --emit-fastq=yes|no         emits fastq output format\n");
    fprintf(fp_help, "  --profile-cpu=yes|no        process section by section (used for profiling on CPU)\n");
#ifdef HAVE_ACC
    fprintf(fp_help,"   --accel=yes|no             Running on accelerator [%s]\n",(opt.flag&SLORADO_ACC?"yes":"no"));
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

/////////////////////////////////////////////////////////////////////////////

    //initialise the core data structure
    core_t* core = init_core(data, opt, model, realtime0);

    int32_t counter=0;

    //initialise a databatch
    db_t* db = init_db(core);

    ret_status_t status = {core->opt.batch_size,core->opt.batch_size_bytes};

    while (status.num_reads >= core->opt.batch_size || status.num_bytes>=core->opt.batch_size_bytes) {
        //load a databatch
        double realtime_d = realtime();
        
        status = load_db(core, db);

        fprintf(stderr, "[%s::%.3f*%.2f] %d Entries (%.1fM bytes) loaded\n", __func__,
                realtime() - realtime0, cputime() / (realtime() - realtime0),
                status.num_reads,status.num_bytes/(1000.0*1000.0));
        double realtime_p = realtime();
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
        //fprintf(stderr, "[%.3f]  Counter : %d \n", realtime() - realtime_d, counter);
        //fprintf(stderr, "[%.3f]  Counter : %d \n", realtime() - realtime_p, counter);
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

    ////Visualization////

    for (size_t i = 0; i < runner_ts.size(); ++i) {
            fprintf(stderr, "\n[%s]          - Model Runner [%zu] time: %.3f",__func__, i, runner_ts[i]->time_basecall + runner_ts[i]->time_decode + runner_ts[i]->time_accept);
            fprintf(stderr, "\n[%s]             - Accept time: %.3f sec",__func__, runner_ts[i]->time_accept);
            fprintf(stderr, "\n[%s]             - Decode time: %.3f sec",__func__, runner_ts[i]->time_decode);
            if(!isCUDA){
                fprintf(stderr, "\n[%s]                 - Beam search emplace time: %.3f sec",__func__, runner_ts[i]->time_beam_search_emplace);
                fprintf(stderr, "\n[%s]                 - Forward time: %.3f sec",__func__, time_forward);

                fprintf(stderr, "\n\n[%s]                     - Forward in ConvolutionImpl time: %.3f sec",__func__, forward_l62);
                fprintf(stderr, "\n[%s]                     - Forward in LinearCRFImpl time: %.3f sec",__func__, forward_l159);
                fprintf(stderr, "\n[%s]                     - Forward in CudaLSTMStackImpl time: %.3f sec",__func__, forward_l469);
                fprintf(stderr, "\n[%s]                     - Forward in LSTMStackImpl time: %.3f sec",__func__, forward_l536);
                fprintf(stderr, "\n[%s]                     - Forward in ClampImpl time: %.3f sec",__func__, forward_l577);
                fprintf(stderr, "\n[%s]                     - Forward in CRFModelImpl time: %.3f sec",__func__, forward_l642);
            }
            else{
                fprintf(stderr, "\n[%s]                 - CudaCaller time: %.3f sec",__func__, CudaCallerT);
                fprintf(stderr, "\n[%s]                 - ~CudaCallerT time: %.3f sec",__func__, NCudaCallerT);
                fprintf(stderr, "\n[%s]                 - NNTaskT time: %.3f sec",__func__, NNTaskT);
                fprintf(stderr, "\n[%s]                 - call_chunks time: %.3f sec",__func__, call_chunksT);
                fprintf(stderr, "\n[%s]                 - cuda_thread_fn time: %.3f sec",__func__, cuda_thread_fnT);
                fprintf(stderr, "\n[%s]                 - SubCudaCallerT time: %.3f sec",__func__, SubCudaCallerT);

            }
        
    }       
            if(!isCUDA){
                fprintf(stderr, "\n\n[%s]                         - x_flip time: %.3f sec",__func__, x_flipt);
                fprintf(stderr, "\n[%s]                         - rnn1 time: %.3f sec",__func__, rnn1t);
                fprintf(stderr, "\n[%s]                         - rnn2 time: %.3f sec",__func__, rnn2t);
                fprintf(stderr, "\n[%s]                         - rnn3 time: %.3f sec",__func__, rnn3t);
                fprintf(stderr, "\n[%s]                         - rnn4 time: %.3f sec",__func__, rnn4t);
                fprintf(stderr, "\n[%s]                         - rnn5 time: %.3f sec",__func__, rnn5t);

                fprintf(stderr, "\n\n[%s]                             - 'auto t1 = rnn1(x)' time: %.3f sec",__func__, rnn1tt1);
                fprintf(stderr, "\n[%s]                             - 'auto h1 = std::get<1>(t1)' time: %.3f sec",__func__, rnn1th1);
                fprintf(stderr, "\n[%s]                             - 'auto y1 = std::get<0>(t1)' time: %.3f sec",__func__, rnn1ty1);
                fprintf(stderr, "\n[%s]                             - 'x = y1.flip(1)' time: %.3f sec",__func__, rnn1tflip);
            }
            else{

            }

            fprintf(stderr, "\n[%s]     - Postprocess time: %.3f sec",__func__, core->postproc_time);
    //}
    fprintf(stderr, "\n[%s] Data output time: %.3f sec", __func__,core->output_time);

 //   fprintf(stderr, "\n[%s] Basecaller DB time: %.6f sec", __func__,core->basecall_db); //new

    fprintf(stderr, "\n[%s] Data output time: %.3f sec : %.2f %\n", __func__,core->output_time,core->output_time*100/total_time);
    fprintf(stderr,"\n");

    if (!isCUDA){
    long level0[] = {(long)core->ts.time_init_runners, (long)core->load_db_time, (long)core->process_db_time};
    std::string level0_Names[] = {"Model initialization time", "Data loading time", "Data processing time"};

    long level1[] = {(long)core->parse_time, (long)core->preproc_time, (long)core->basecall_time};
    std::string level1_Names[] = {"Parse time", "Preprocess time", "Basecall+decode time"};

    long level2[] = { (long)core->ts.time_sync, runner_ts[0]->time_basecall + runner_ts[0]->time_decode + runner_ts[0]->time_accept};
    std::string level2_Names[] = {"Synchronisation time", "Model Runner time"};

    long level3[] = {(long) runner_ts[0]->time_accept, (long)runner_ts[0]->time_decode};
    std::string level3_Names[] = {"Accept time", "Decode time"};

    long level4[] = {(long)runner_ts[0]->time_beam_search_emplace, (long)time_forward};
    std::string level4_Names[] = {"Beam search emplace time", "Forward function time"};

    //For different forward() functions    
    long level5[] = {forward_l62, forward_l159,forward_l469, forward_l536, forward_l577, forward_l642};
    std::string level5_Names[] = {"forward() in ConvolutionImpl", "forward() in LinearCRFImpl", "forward() in CudaLSTMStackImpl", "forward() in LSTMStackImpl", "forward() in ClampImpl", "forward() in CRFModelImpl"};

    long level6[] = {x_flipt, rnn1t, rnn2t, rnn3t, rnn4t, rnn5t};
    std::string level6_Names[] = {"x.flip time", "rnn1 time", "rnn2 time", "rnn3 time", "rnn4 time", "rnn5 time"};

    long level7[] = {rnn1tt1, rnn1th1, rnn1ty1, rnn1tflip};
    std::string level7_Names[] = {"'auto t1 = rnn1(x)' time", "'auto y1 = std::get<0>(t1)' time", "'auto h1 = std::get<1>(t1)' time", "'x = y1.flip(1)' time"};
    
    
    //, , "Synchronisation time", "Postprocess time", "Data output time:";

     //,, (long)core->postproc_time, (long)core->output_time};

    std::cout << "\n" << std::endl;
    std::cout << "---------------------------------------- Time Distribution -----------------------------------------" << std::endl;

    ///////////////////////////////////////////////////////////////////////////////////////////////////
////    long values[] = {(long)core->ts.time_init_runners, (long)core->load_db_time, (long)core->process_db_time, (long)core->parse_time, (long)core->preproc_time, (long)core->basecall_time, (long)core->ts.time_sync, (long)core->postproc_time, (long)core->output_time};


    generateSplitBar(level0, level0_Names, 3);
    std::cout << "\nIn Data processing" << std::endl;
    generateSplitBar(level1, level1_Names, 3);
    std::cout << "\nIn  Basecall" << std::endl;
    generateSplitBar(level2, level2_Names, 2);
    std::cout << "\nIn Model Runner" << std::endl;
    generateSplitBar(level3, level3_Names, 2);
    std::cout << "\nIn Decode" << std::endl;
    generateSplitBar(level4, level4_Names, 2);
    std::cout << "\nIn Time consumption for forward() functions" << std::endl;
    generateSplitBar(level5, level5_Names, 6);
    std::cout << "\nIn most time consuming forwad() function" << std::endl;
    generateSplitBar(level6, level6_Names, 6);
    std::cout << "\nIn rnn1" << std::endl;
    generateSplitBar(level7, level7_Names, 4);

    }
    //free the core data structure
    free_core(core,opt);

    if (opt.out != stdout) {
        fclose(opt.out);
    }

    return 0;
}

void generateSplitBar(const long* values, const std::string* names, int size) {
    int barLength = 100; // Length of the bar
    long sum = 0;

    std::string colors[] = {"\033[41m", "\033[45m", "\033[43m", "\033[44m", "\033[46m","\033[42m"};

    // std::string colorCode = "\033[" + std::to_string(41 + j) + "m"; // Set background color dynamically
    long sortedVal[size];
    std::string sortedValNames[size];

    for (int i = 0; i < size; i++) {
        sortedVal[i] = values[i];
        sortedValNames[i] = names[i];
        sum += values[i];
    }
    int len = 0;
    (size > 6)?  len = 6: len = size;

    for (int i = 0; i < len; i++) {
        for (int j = 0; j < len - i - 1; j++) {
            if (sortedVal[j] < sortedVal[j + 1]) {
                long temp = sortedVal[j];
                sortedVal[j] = sortedVal[j + 1];
                sortedVal[j + 1] = temp;

                std::string tempName = sortedValNames[j];
                sortedValNames[j] = sortedValNames[j+1];
                sortedValNames[j+1] = tempName;
            }
        }
    }

    std::string bar;

    // Append colored portions represented by colored spaces
    int currentPosition = 0;
    for (int i = 0; i < len; ++i) {

        int coloredLength = (sortedVal[i] * barLength) / sum; // Calculate length of colored portion
        std::string colorCode = colors[i]; // Set background color dynamically

        // Append colored portion with the respective color
        bar += colorCode;
        bar.append(coloredLength, ' ');
        bar += "\033[0m"; // Reset color to default

        currentPosition += coloredLength;
    }

    // Append remaining spaces
    bar.append(barLength - currentPosition, ' ');

    // Print the space bar
    std::cout << bar << std::endl;

    std::cout << "\n" << std::endl;
    // Print the value names with two spaces in the respective color in front
    for (int i = 0; i < len; ++i) {
        std::string colorCode = colors[i]; // Set background color dynamically

        // Print two spaces with the color code
        std::cout << colorCode << "  ";

        // Reset the color code
        std::cout << "\033[0m";

        // Print the value name
        std::cout << " : " << sortedValNames[i] <<  std::endl;
    }
    std::cout << "\n" << std::endl;
}

