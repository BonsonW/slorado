/**
 * @file torchbox.c
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
#include "error.h"
#include "misc.h"
#include "torchbox.h"
#include "dorado/tensor_chunk_utils.h"
#include "dorado/CRFModel.h"
#include "dorado/TxModel.h"

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
    core_t* core,
    runner_t* runner,
    char *model_path,
    const std::string &device,
    int batch_size,
    torch::ScalarType dtype
) {
    LOG_TRACE("initializing model runner for device %s", device.c_str());
    runner->device = device;

    runner->tensor_opts = torch::TensorOptions().dtype(dtype).device(device);
    if (core->model_config->tx != NULL) {
        runner->module = load_tx_model(*core->model_config, runner->tensor_opts);
    } else {
        runner->module = load_lstm_model(*core->model_config, runner->tensor_opts);
    }
    LOG_TRACE("%s", "model populated");

    runner->input_tensor = torch::zeros({batch_size, 1, (int64_t)core->chunk_size}, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    
    if (device != "cpu") {
#ifdef USE_GPU
        int64_t device_idx = device[device.size()-1] - '0'; // quick and dirty device index extraction
        runner->device_idx = device_idx;

#ifdef HAVE_CUDA
        c10::cuda::CUDAGuard device_guard(device_idx);
#endif
#ifdef HAVE_ROCM
        c10::hip::HIPGuard device_guard(device_idx);
#endif
        runner->gpubuf = openfish_gpubuf_init(core->chunk_size / core->model_stride, batch_size, core->model_config->state_len);
#endif        
    }

    LOG_DEBUG("fully initialized model runner for device %s", device.c_str());
}

/* initialise runner_stat */
void init_runner_stat(runner_stat_t *time_stamps) {
    memset(time_stamps, 0, sizeof(runner_stat_t));
}

void init_runners(core_t* core, opt_t *opt, char *model) {
    core->runners = new std::vector<runner_t *>();
    core->runner_stats = new std::vector<runner_stat_t *>();

    if (strcmp(opt->device, "cpu") == 0) {
        std::string device = opt->device;
        for (int i = 0; i < opt->num_runners; ++i) {
            core->runner_stats->push_back((runner_stat_t *)malloc(sizeof(runner_stat_t)));
            init_runner_stat((*core->runner_stats).back());

            core->runners->push_back(new runner_t());
            init_runner(core, (*core->runners).back(), model, device, opt->gpu_batch_size, torch::kF32);
        }
    } else {
#ifdef USE_GPU
        std::vector<std::string> devices;
        std::string device_args = std::string(opt->device);
        devices = parse_cuda_device_string(device_args);
        if (devices.size() < 1) {
            ERROR("%s", "Could not locate any cuda devices");
            exit(EXIT_FAILURE);
        }

        for (auto device: devices) {
            for (int i = 0; i < opt->num_runners; ++i) {
                core->runner_stats->push_back((runner_stat_t *)malloc(sizeof(runner_stat_t)));
                init_runner_stat((*core->runner_stats).back());
                core->runners->push_back(new runner_t());
                init_runner(core, (*core->runners).back(), model, device, opt->gpu_batch_size, torch::kF16);
            }
        }
#else
        ERROR("Invalid device: %s. Please compile again for GPU", opt->device);
        exit(EXIT_FAILURE);
#endif
    }

    auto adjusted_chunk_size = core->chunk_size;
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
        runner_t *runner = (*core->runners)[i];
        if (runner->device != "cpu") {
#ifdef USE_GPU
#ifdef HAVE_CUDA
            c10::cuda::CUDAGuard device_guard(runner->device_idx);
#endif
#ifdef HAVE_ROCM
            c10::hip::HIPGuard device_guard(runner->device_idx);
#endif
            openfish_gpubuf_free(runner->gpubuf);
#endif
        }
        delete runner;
    }

}

void init_tensor_db(db_t *db) {
    db->tensor_db = (tensor_db_t *)malloc(sizeof(tensor_db_t));
    MALLOC_CHK(db->tensor_db);
    db->tensor_db->tensors = new std::vector<std::vector<torch::Tensor>>(db->capacity_rec, std::vector<torch::Tensor>());
}

void free_tensor_db(db_t *db) {
    delete db->tensor_db->tensors;
    free(db->tensor_db);
}

torch::Tensor tensor_from_record(slow5_rec_t *rec) {
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt16);
    return torch::from_blob(rec->raw_signal, rec->len_raw_signal, options);
}

std::vector<chunk_t> create_chunks(size_t tensor_size, size_t chunk_size, size_t overlap) {
    size_t step = chunk_size - overlap;

    size_t n_chunks = tensor_size / step;
    n_chunks += tensor_size % step > 0 ? 1 : 0;

    std::vector<chunk_t> chunks;
    chunks.reserve(n_chunks);

    for (size_t i = 0; i < n_chunks; ++i) {
        size_t sig_pos = std::min(step * i, tensor_size - chunk_size);
        chunks.push_back({sig_pos, i, chunk_size, std::string(), std::string(), std::vector<uint8_t>()});
    }

    return chunks;
}

std::vector<torch::Tensor> tensor_chunks(torch::Tensor &signal, std::vector<chunk_t> &chunks, size_t chunk_size) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(chunks.size());

    for (size_t i = 0; i < chunks.size(); ++i) {
        torch::Tensor input_slice = signal.index({torch::indexing::Ellipsis, torch::indexing::Slice(chunks[i].input_offset, chunks[i].input_offset + chunk_size)});
        input_slice = input_slice.unsqueeze(0);
        size_t slice_size = input_slice.size(1);

        // repeat-pad non-full chunks
        if (slice_size != chunk_size) {
            int64_t quot = chunk_size / slice_size;
            int64_t rem = chunk_size % slice_size;
            input_slice = torch::concat(
                {
                    input_slice.repeat({1, quot}),
                    input_slice.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, rem)})
                },
                1
            );
        }
        tensors.push_back(input_slice);
    }

    return tensors;
}

void preprocess_signal(core_t *core, db_t *db, int32_t i) {
    slow5_rec_t *rec = db->slow5_rec[i];
    uint64_t len_raw_signal = rec->len_raw_signal;
    opt_t opt = core->opt;

    if (len_raw_signal > 0) {
        auto signal_norm_params = core->model_config->signal_norm_params;

        torch::Tensor signal = tensor_from_record(rec);

        scale_signal(signal, rec->range / rec->digitisation, rec->offset, signal_norm_params);

        std::vector<chunk_t> chunks = create_chunks(signal.size(0), core->chunk_size, opt.overlap);
        (*db->chunks)[i] = chunks;

        std::vector<torch::Tensor> tensors = tensor_chunks(signal, chunks, core->chunk_size);
        (*db->tensor_db->tensors)[i] = tensors;
    }
}