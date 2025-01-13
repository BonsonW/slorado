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
#include <openfish/openfish.h>

#ifdef HAVE_CUDA
#include <c10/cuda/CUDAGuard.h>
#include "buffers.h"
// #include "IEngine.h"
#endif

#ifdef HAVE_ROCM
#include <c10/hip/HIPGuard.h>
#endif

struct DecodedChunk {
    std::string sequence;
    std::string qstring;
    std::vector<uint8_t> moves;
};

bool trt_infer(
    std::vector<torch::Tensor> tensors,
    std::vector<Chunk *> chunks,
    const int chunk_size,
    const core_t* core,
    const int runner_idx
) {
    FILE *fp;
    runner_t* runner = (*core->runners)[runner_idx];

#ifdef USE_GPU
#ifdef HAVE_CUDA
    c10::cuda::CUDAGuard device_guard(runner->device_idx);
#endif
#ifdef HAVE_ROCM
    c10::hip::HIPGuard device_guard(runner->device_idx);
#endif
#endif
    BufferManager buffers(runner->engine);

    for (int32_t i = 0, e = runner->engine->getNbIOTensors(); i < e; i++) {
        auto const name = runner->engine->getIOTensorName(i);
        runner->context->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }
    
    LOG_DEBUG("%s", "copying input");
    fprintf(stderr, "input sizes: %ld, %ld, %ld\n", runner->input_dims.d[0], runner->input_dims.d[1], runner->input_dims.d[2]);
    // read the input data into the managed buffers
    void *host_input_buffer = static_cast<float *>(buffers.getHostBuffer(runner->io["input"]));
    for (size_t i = 0; i < tensors.size(); ++i) {
        memcpy((void *)(host_input_buffer + (i * runner->chunk_size)), (void *)tensors[i].data_ptr(), tensors[i].numel() * (sizeof(float) / 2));
    }
    // auto d = runner->input_dims.d[0] * runner->input_dims.d[1] * runner->input_dims.d[2];

    // fp = fopen("input_trt_C1.blob", "w");
    // F_CHK(fp, "input_trt_C1.blob");
    // if (fwrite(host_input_buffer, sizeof(float) / 2, d, fp) != d) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);

    LOG_DEBUG("%s", "moving buffers");

    // Create CUDA stream for the execution of this inference
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    checkCudaError();

    // Asynchronously copy data from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    // Asynchronously enqueue the inference work
    if (!runner->context->enqueueV3(stream)) {
        return false;
    }

    // Asynchronously copy data from device output buffers to host output buffers
    // buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete
    cudaStreamSynchronize(stream);
    checkCudaError();

    // Release stream
    cudaStreamDestroy(stream);
    checkCudaError();

    LOG_DEBUG("%s", "inference done");

    // get scores
    void *scores_TNC = buffers.getDeviceBuffer(runner->io.at("output"));

    fprintf(stderr, "output sizes: %ld, %ld, %ld\n", runner->output_dims.d[0], runner->output_dims.d[1], runner->output_dims.d[2]);
    // auto k = runner->output_dims.d[0] * runner->output_dims.d[1] * runner->output_dims.d[2];

    // fp = fopen("scores_trt_C1.blob", "w");
    // F_CHK(fp, "scores_trt_C1.blob");
    // if (fwrite(scores_TNC, sizeof(float) / 2, k, fp) != k) {
    //     fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
    //     exit(EXIT_FAILURE);
    // }
    // fclose(fp);
    // exit(0);

    const int T = runner->output_dims.d[0];
    const int N = runner->output_dims.d[1];
    const int C = runner->output_dims.d[2];
    const int state_len = runner->model_config.state_len;
    int nthreads = core->opt.num_thread / core->runners->size();

    uint8_t *moves;
    char *sequence;
    char *qstring;

    LOG_DEBUG("%s", "decoding scores");

    if (runner->device == "cpu") {
        openfish_decode_cpu(T, N, C, nthreads, scores_TNC, state_len, &runner->decoder_opts, &moves, &sequence, &qstring);
    } else {
#ifdef USE_GPU
        openfish_decode_gpu(T, N, C, scores_TNC, state_len, &runner->decoder_opts, runner->gpubuf, &moves, &sequence, &qstring);
#else
        ERROR("Invalid device: %s. Please compile again for GPU", runner->device.c_str());
        exit(EXIT_FAILURE);
#endif
    }

    std::vector<DecodedChunk> decoded_chunks(chunks.size());
    for (size_t chunk = 0; chunk < decoded_chunks.size(); ++chunk) {
        size_t idx = chunk * T;
        decoded_chunks[chunk] = {
            std::string(sequence + idx),
            std::string(qstring + idx),
            std::vector<uint8_t>(moves + idx, moves + idx + T),
        };

        if (decoded_chunks[chunk].sequence.size() == 0) {
            ERROR("%s", "empty sequence returned by decoder");
            exit(EXIT_FAILURE);
        }

        if (decoded_chunks[chunk].qstring.size() == 0) {
            ERROR("%s", "empty qstring returned by decoder");
            exit(EXIT_FAILURE);
        }

        size_t seq_size = decoded_chunks[chunk].sequence.size();
        size_t qstr_size = decoded_chunks[chunk].qstring.size();
        if (seq_size != qstr_size) {
            ERROR("mismatch sequence size of %zu with qstring size of %zu", seq_size, qstr_size);
            exit(EXIT_FAILURE);
        }
    }

    for (size_t i = 0; i < chunks.size(); ++i) {
        chunks[i]->seq = decoded_chunks[i].sequence;
        chunks[i]->qstring = decoded_chunks[i].qstring;
        chunks[i]->moves = decoded_chunks[i].moves;
    }

    free(moves);
    free(sequence);
    free(qstring);

    // Check and print the output of the inference
    return true;
}

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

static void get_io_names(runner_t* runner) {
    int32_t nbindings = runner->engine.get()->getNbIOTensors();
    ASSERT(nbindings == 2);

    for (int32_t b = 0; b < nbindings; ++b) {
        auto const binding_name = runner->engine.get()->getIOTensorName(b);
        nvinfer1::Dims dims = runner->engine.get()->getTensorShape(binding_name);
        if (runner->engine.get()->getTensorIOMode(binding_name) == nvinfer1::TensorIOMode::kINPUT) {
            LOG_DEBUG("%s", "found input");
            // sample::gLogInfo << "Found input: " << binding_name << " shape=" << dims
            //                  << " dtype=" << static_cast<int32_t>(runner->engine.get()->getTensorDataType(binding_name))
            //                  << std::endl;
            runner->io["input"] = binding_name;
        } else {
            LOG_DEBUG("%s", "found output");
            // sample::gLogInfo << "Found output: " << binding_name << " shape=" << dims
            //                  << " dtype=" << static_cast<int32_t>(runner->engine.get()->getTensorDataType(binding_name))
            //                  << std::endl;
            runner->io["output"] = binding_name;
        }
    }
}

static bool load_trt_network(runner_t* runner, char *trt_model_path) {
    std::ifstream file(trt_model_path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        ERROR("%s", "unable to read engine file");
        return false;
    }

    runner->logger = std::make_unique<Logger>(Severity::kINFO);

    // create a runtime to deserialize the engine file.
    runner->runtime = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(*runner->logger)
    );
    if (!runner->runtime) {
        ERROR("%s", "unable to initialise nvinfer runtime");
        return false;
    }

    // create an engine, a representation of the optimized model.
    runner->engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runner->runtime->deserializeCudaEngine(buffer.data(), buffer.size())
    );
    if (!runner->engine) {
        ERROR("%s", "unable to initialise nvinfer cuda engine");
        return false;
    }

    get_io_names(runner);

    runner->input_dims = runner->engine.get()->getTensorShape(runner->io["input"].c_str());
    runner->output_dims = runner->engine.get()->getTensorShape(runner->io["output"].c_str());

    runner->context = std::unique_ptr<nvinfer1::IExecutionContext>(runner->engine->createExecutionContext());
    if (!runner->context) {
        return false;
    }

    return true;
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
    // if (use_tx) {
    //     runner->module = load_tx_model(model_config, runner->tensor_opts);
    // } else {
    //     runner->module = load_lstm_model(model_config, runner->tensor_opts);
    // }

    LOG_TRACE("%s", "model populated");

    chunk_size -= chunk_size % runner->model_stride;
    // runner->input_tensor = torch::zeros({batch_size, 1, chunk_size}, torch::TensorOptions().dtype(dtype).device(torch::kCPU));
    runner->chunk_size = chunk_size;

    char *trt_model_path = "/data/bonwon/models_trt/420_sup.trt";

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
        load_trt_network(runner, trt_model_path);
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
