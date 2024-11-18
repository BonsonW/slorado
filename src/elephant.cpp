/**
 * @file elephant.c
 * @brief common functions for slorado that depends on torch
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
#include "slorado.h"
#include "error.h"
#include "elephant.h"
#include "dorado/signal_prep_stitch_tensor_utils.h"

#ifdef HAVE_CUDA
#include <c10/cuda/CUDAGuard.h>
#endif

#ifdef HAVE_ROCM
#include <c10/hip/HIPGuard.h>
#endif

std::vector<std::string> parse_cuda_device_string(std::string device_arg) {
    std::vector<std::string> devices;

    if (device_arg == "cuda:all" || device_arg == "cuda:auto") {
        for (size_t i = 0; i < torch::cuda::device_count(); i++) {
            devices.push_back("cuda:" + std::to_string(i));
        }
        return devices;
    }

    std::string device_name = "";
    std::string delimiter = ":";
    size_t pos = device_arg.find(delimiter);
    device_name = device_arg.substr(0, pos + delimiter.length());
    device_arg.erase(0, pos + delimiter.length());

    delimiter = ",";
    while ((pos = device_arg.find(delimiter)) != std::string::npos) {
        devices.push_back(device_name + device_arg.substr(0, pos));
        device_arg.erase(0, pos + delimiter.length());
    }
    devices.push_back(device_name + device_arg.substr(0, pos));

    return devices;
}


/* initialise runners */
void init_runner(
    runner_t* runner,
    char *model_path,
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

#ifdef USE_GPU
    int64_t device_idx = device[device.size()-1] - '0'; // quick and dirty device index extraction
    runner->device_idx = device_idx;

#ifdef HAVE_CUDA
    c10::cuda::CUDAGuard device_guard(device_idx);
#endif
#ifdef HAVE_ROCM
    c10::hip::HIPGuard device_guard(device_idx);
#endif
    runner->gpubuf = openfish_gpubuf_init(chunk_size / runner->model_stride, batch_size, model_config.state_len);
#endif

    LOG_DEBUG("fully initialized model runner for device %s", device.c_str());
}

/* initialise runner_stat */
void init_runner_stat(runner_stat_t* time_stamps) {
    memset(time_stamps, 0, sizeof(runner_stat_t));
}



void init_runners(core_t* core, opt_t *opt, char *model){


    core->runners = new std::vector<runner_t*>();
    core->runner_stats = new std::vector<runner_stat_t*>();

    if (strcmp(opt->device, "cpu") == 0) {
        std::string device = opt->device;
        for (int i = 0; i < opt->num_runners; ++i) {
            core->runner_stats->push_back((runner_stat_t*)malloc(sizeof(runner_stat_t)));
            init_runner_stat((*core->runner_stats).back());

            core->runners->push_back(new runner_t());
            init_runner((*core->runners).back(), model, device, opt->chunk_size, opt->gpu_batch_size, torch::kF32);
        }
    } else {
#ifdef USE_GPU
        std::vector<std::string> devices;
        std::string device_args = std::string(opt->device);
        devices = parse_cuda_device_string(device_args);

        for (auto device: devices) {
            for (int i = 0; i < opt->num_runners; ++i) {
                core->runner_stats->push_back((runner_stat_t*)malloc(sizeof(runner_stat_t)));
                init_runner_stat((*core->runner_stats).back());
                core->runners->push_back(new runner_t());
                init_runner((*core->runners).back(), model, device, opt->chunk_size, opt->gpu_batch_size, torch::kF16);
            }
        }
#else
        fprintf(stderr, "Error. Please compile again for GPU\n");
        exit(EXIT_FAILURE);
#endif
    }

    auto adjusted_chunk_size = core->runners->front()->chunk_size;
    if (opt->chunk_size != adjusted_chunk_size) {
        LOG_DEBUG("Adjusting chunk size to %zu", adjusted_chunk_size);
        opt->chunk_size = adjusted_chunk_size;
    }
}

void free_runners(core_t *core) {

    for (size_t i = 0; i < core->runner_stats->size(); ++i) {
        free((*core->runner_stats)[i]);
    }

    for (size_t i = 0; i < core->runners->size(); ++i) {
        runner_t* runner = (*core->runners)[i];
#ifdef USE_GPU
#ifdef HAVE_CUDA
        c10::cuda::CUDAGuard device_guard(runner->device_idx);
#endif
#ifdef HAVE_ROCM
        c10::hip::HIPGuard device_guard(runner->device_idx);
#endif
        openfish_gpubuf_free(runner->gpubuf);
#endif
        delete runner;
    }

}

void init_elephant(db_t *db) {
    db->elephant = (elephant_t*) malloc(sizeof(elephant_t));
    MALLOC_CHK(db->elephant);
    db->elephant->tensors = new std::vector<std::vector<torch::Tensor>>(db->capacity_rec, std::vector<torch::Tensor>());
}

void free_elephant(db_t *db) {
    delete db->elephant->tensors;
    free(db->elephant);
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

        (*db->elephant->tensors)[i] = tensors;
        LOG_TRACE("%s","assigned tensors");
    }
}