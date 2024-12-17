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
#include "error.h"
#include "elephant.h"
#include "dorado/signal_prep_stitch_tensor_utils.h"
#include "dorado/CRFModel.h"
#include "dorado/TxModel.h"

#ifdef HAVE_CUDA
#include <c10/cuda/CUDAGuard.h>
#include "NvInfer.h"
// #include "IEngine.h"
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

    exit(1);
    LOG_TRACE("initializing model runner for device %s", device.c_str());
    bool use_tx = is_tx_model_config(model_path);

    CRFModelConfig model_config;

    if (use_tx) {
        model_config = load_tx_model_config(model_path);
    } else {
        model_config = load_lstm_model_config(model_path);
    }
    model_config.model_path = std::string(model_path);
    
    LOG_TRACE("%s", "model config loaded");

    runner->model_stride = static_cast<size_t>(model_config.stride);

    runner->decoder_opts = DECODER_INIT;
    runner->decoder_opts.q_shift = model_config.qbias;
    runner->decoder_opts.q_scale = model_config.qscale;

    runner->device = device;
    runner->model_config = model_config;

    runner->tensor_opts = torch::TensorOptions().dtype(dtype).device(device);
    if (use_tx) {
        runner->module = load_tx_model(model_config, runner->tensor_opts);
    } else {
        runner->module = load_lstm_model(model_config, runner->tensor_opts);
    }

    LOG_TRACE("%s", "model populated");

    chunk_size -= chunk_size % runner->model_stride;
    runner->input_tensor = torch::zeros({batch_size, 1, chunk_size}, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    runner->chunk_size = runner->input_tensor.size(2);

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
        runner->gpubuf = openfish_gpubuf_init(chunk_size / runner->model_stride, batch_size, model_config.state_len);
#endif        
    }

    LOG_DEBUG("fully initialized model runner for device %s", device.c_str());
}

/* initialise runner_stat */
void init_runner_stat(runner_stat_t *time_stamps) {
    memset(time_stamps, 0, sizeof(runner_stat_t));
}

void init_runners(core_t* core, opt_t *opt, char *model){
    // todo: this should be in its own function
    torch::set_num_threads(opt->num_thread);

    core->runners = new std::vector<runner_t *>();
    core->runner_stats = new std::vector<runner_stat_t *>();

    if (strcmp(opt->device, "cpu") == 0) {
        std::string device = opt->device;
        for (int i = 0; i < opt->num_runners; ++i) {
            core->runner_stats->push_back((runner_stat_t *)malloc(sizeof(runner_stat_t)));
            init_runner_stat((*core->runner_stats).back());

            core->runners->push_back(new runner_t());
            init_runner((*core->runners).back(), model, device, opt->chunk_size, opt->gpu_batch_size, torch::kF32);
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
                init_runner((*core->runners).back(), model, device, opt->chunk_size, opt->gpu_batch_size, torch::kF16);
            }
        }
#else
        ERROR("Invalid device: %s. Please compile again for GPU", opt->device);
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

void init_elephant(db_t *db) {
    db->elephant = (elephant_t *)malloc(sizeof(elephant_t));
    MALLOC_CHK(db->elephant);
    db->elephant->tensors = new std::vector<std::vector<torch::Tensor>>(db->capacity_rec, std::vector<torch::Tensor>());
}

void free_elephant(db_t *db) {
    delete db->elephant->tensors;
    free(db->elephant);
}

void preprocess_signal(core_t *core, db_t *db, int32_t i) {
    slow5_rec_t *rec = db->slow5_rec[i];
    uint64_t len_raw_signal = rec->len_raw_signal;
    opt_t opt = core->opt;

    if (len_raw_signal > 0) {
        // quick and dirty model config, todo: should move this to core
        runner_t* runner = (*core->runners)[0];
        auto signal_norm_params = runner->model_config.signal_norm_params;

        torch::Tensor signal = tensor_from_record(rec).to(torch::kCPU);

        scale_signal(signal, rec->range / rec->digitisation, rec->offset, signal_norm_params);

        std::vector<Chunk *> chunks = chunks_from_tensor(signal, opt.chunk_size, opt.overlap);
        (*db->chunks)[i] = chunks;

        std::vector<torch::Tensor> tensors = tensor_as_chunks(signal, chunks, opt.chunk_size);
        (*db->elephant->tensors)[i] = tensors;
    }
}

bool load_network(std::string trtModelPath) {
    std::ifstream file(trtModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        auto msg = "Error, unable to read engine file";
        ERROR(msg);
        exit(EXIT_FAILURE)
    }

    // Create a runtime to deserialize the engine file.
    runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(logger)};
    if (!runtime) {
        return false;
    }

    // Set the device index
    cudaSetDevice(m_options.deviceIndex);
    checkCudaError();

    // Create an engine, a representation of the optimized model.
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }

    // The execution context contains all of the state associated with a
    // particular invocation
    context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!context) {
        return false;
    }

    // Storage for holding the input and output buffers
    // This will be passed to TensorRT for inference
    clearGpuBuffers();
    m_buffers.resize(m_engine->getNbIOTensors());

    m_output_lens.clear();
    m_input_dims.clear();
    m_output_dims.clear();
    m_io_tensor_names.clear();

    // Create a cuda stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    checkCudaError();

    // Allocate GPU memory for input and output buffers
    m_output_lens.clear();
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const auto tensor_name = m_engine->getIOTensorName(i);
        m_io_tensor_names.emplace_back(tensor_name);
        const auto tensor_type = m_engine->getTensorIOMode(tensor_name);
        const auto tensor_shape = m_engine->getTensorShape(tensor_name);
        const auto tensor_dtype = m_engine->getTensorDataType(tensor_name);

        if (tensor_type == nvinfer1::TensorIOMode::kINPUT) {
            // The implementation currently only supports inputs of type float
            if (m_engine->getTensorDataType(tensor_name) != nvinfer1::DataType::kHALF) {
                ERROR("%s", "the implementation currently only supports half float inputs");
                exit(EXIT_FAILURE)
            }

            // Don't need to allocate memory for inputs as we will be using the OpenCV
            // GpuMat buffer directly.

            // Store the input dims for later use
            m_input_dims.emplace_back(tensor_shape.d[1], tensor_shape.d[2], tensor_shape.d[3]);
            input_batch_size = tensor_shape.d[0];
        } else if (tensor_type == nvinfer1::TensorIOMode::kOUTPUT) {
            // The binding is an output
            uint32_t output_len = 1;
            m_output_dims.push_back(tensor_shape);

            for (int j = 1; j < tensor_shape.nbDims; ++j) {
                // We ignore j = 0 because that is the batch size, and we will take that
                // into account when sizing the buffer
                output_len *= tensor_shape.d[j];
            }

            m_output_lens.push_back(output_len);
            // Now size the output buffer appropriately, taking into account the max
            // possible batch size (although we could actually end up using less memory)
            cudaMallocAsync(&m_buffers[i], output_len * m_options.maxBatchSize * sizeof(T), stream);
            checkCudaError();
        } else {=
            ERROR("%s", "IO Tensor is neither an input or output");
            exit(EXIT_FAILURE)
        }
    }

    // Synchronize and destroy the cuda stream
    cudaStreamSynchronize(stream);
    checkCudaError();
    cudaStreamDestroy(stream);
    checkCudaError();

    return true;
}