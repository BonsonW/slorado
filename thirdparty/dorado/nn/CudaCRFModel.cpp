#include "CudaCRFModel.h"

#include "dorado/decode/GPUDecoder.h"
#include "error.h"
#include "../utils/cuda_utils.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "toml.h"
#include <torch/torch.h>

#include <algorithm>
#include <cassert>
#include <limits>

using namespace std::chrono_literals;

class CudaCaller {
public:
    CudaCaller(const CRFModelConfig &model_config,
               int chunk_size,
               int batch_size,
               const std::string &device,
               float memory_limit_fraction,
               bool exclusive_gpu_access)
            : m_config(model_config),
              m_device(device),
              m_exclusive_gpu_access(exclusive_gpu_access) {
        m_decoder_options = DecoderOptions();
        m_decoder_options.q_shift = model_config.qbias;
        m_decoder_options.q_scale = model_config.qscale;
        m_decoder = std::make_unique<GPUDecoder>();
        m_num_input_features = model_config.num_features;
        // adjust chunk size to be a multiple of the stride
        m_out_chunk_size = chunk_size / model_config.stride;
        m_in_chunk_size = m_out_chunk_size * model_config.stride;

        m_options = torch::TensorOptions().dtype(GPUDecoder::dtype).device(device);
        assert(m_options.device().is_cuda());

        torch::InferenceMode guard;
        m_module = load_crf_model(model_config, m_options);
        
        // Batch size will be rounded up to a multiple of batch_size_granularity, regardless of
        // user choice. This makes sure batch size is compatible with GPU kernels.
        if (batch_size == 0) {
            m_batch_size = auto_batch_size(model_config, chunk_size, memory_limit_fraction);
        } else {
            int batch_size_granularity = get_batch_size_granularity(model_config, m_options);
            m_batch_size = pad_to(batch_size, batch_size_granularity);
            // Warmup
            auto input =
                    torch::empty({m_batch_size, m_num_input_features, m_in_chunk_size}, m_options);
            m_module->forward(input);
            torch::cuda::synchronize(m_options.device().index());
        }

        c10::cuda::CUDAGuard device_guard(m_options.device());
        c10::cuda::CUDACachingAllocator::emptyCache();

        start_threads();
    }

    void start_threads() {
        m_cuda_thread.reset(new std::thread(&CudaCaller::cuda_thread_fn, this));
    }

    ~CudaCaller() {
        m_terminate.store(true);
        m_input_cv.notify_one();
        if (m_cuda_thread && m_cuda_thread->joinable()) {
            m_cuda_thread->join();
        }
    }

    static int get_batch_size_granularity(const CRFModelConfig &model_config,
                                          const torch::TensorOptions &options) {
        // TODO: we may want to use different numbers based on model type and GPU arch
        return 64;
    }

    int auto_batch_size(const CRFModelConfig &model_config,
                        int chunk_size_in,
                        float memory_limit_fraction) {
#ifdef DORADO_TX2
        return 256;
#else
        int64_t available = available_memory(m_options.device()) * memory_limit_fraction;

        int granularity = get_batch_size_granularity(model_config, m_options);

        // Determine size of working memory for CRFModel divided by (batch_size * chunk_size)
        // These values have been derermined by running dorado with different models and
        // reporting the actual allocation size per chunk-timestep.
        int64_t crfmodel_bytes_per_chunk_timestep;
        if (model_config.decomposition) {
            auto out_features = model_config.out_features;
            std::unordered_map<int, int64_t> out_features_map{{128, 2312}, {256, 8712}};
            crfmodel_bytes_per_chunk_timestep = out_features_map[out_features];
            if (crfmodel_bytes_per_chunk_timestep == 0) {
                return granularity;
            }
        } else {
            std::unordered_map<int, int64_t> insize_map{
                    {96, 960}, {128, 1280}, {384, 2816}, {768, 9728}, {1024, 10240}};
            crfmodel_bytes_per_chunk_timestep = insize_map[model_config.insize];
            if (crfmodel_bytes_per_chunk_timestep == 0) {
                return granularity;
            }
        }

        // Determine size of working memory for decoder divided by (batch_size * chunk_size)
        // Decoder needs roughly (beam_width * 4) + num_states + 10 extra bytes
        // where num_states = 4^(state_len+1)
        // See `dorado::GPUDecoder::gpu_part()`, block beginning with `if (!initialized) {`
        // for more details.
        int64_t decode_bytes_per_chunk_timestep =
                10 + m_decoder_options.beam_width * 4 + (1 << (model_config.state_len * 2 + 2));

        auto bytes_per_chunk_timestep =
                decode_bytes_per_chunk_timestep + crfmodel_bytes_per_chunk_timestep;
        int64_t chunk_size_out = chunk_size_in / model_config.stride;
        available = available - 1.0e9f;  // Allow 1GB for model weights, etc.
        if (available < 0) {
            return granularity;
        }

        const int64_t max_batch_size_limit = 10240;
        const int max_batch_size = std::min(available / (bytes_per_chunk_timestep * chunk_size_out),
                                            max_batch_size_limit);
        if (max_batch_size < pad_to(128, granularity) + granularity) {
            return granularity;
        }

        c10::cuda::CUDAGuard device_guard(m_options.device());

        int best_batch_size = granularity;
        float best_time = std::numeric_limits<float>::max();
        const int chunk_size = std::min(chunk_size_in, model_config.stride * 300);
        for (int batch_size = granularity; batch_size <= max_batch_size;
             batch_size += granularity) {
            auto input =
                    torch::empty({batch_size, model_config.num_features, chunk_size}, m_options);

            float time = std::numeric_limits<float>::max();
            for (int i = 0; i < 2; ++i) {  // run twice to eliminate outliers
                cudaEvent_t start, stop;
                handle_cuda_result(cudaEventCreate(&start));
                handle_cuda_result(cudaEventCreate(&stop));
                handle_cuda_result(cudaEventRecord(start));
                m_module->forward(input);
                handle_cuda_result(cudaEventRecord(stop));
                handle_cuda_result(cudaEventSynchronize(stop));
                float ms = 0;
                handle_cuda_result(cudaEventElapsedTime(&ms, start, stop));
                time = std::min(time, ms / batch_size);
                handle_cuda_result(cudaEventDestroy(start));
                handle_cuda_result(cudaEventDestroy(stop));
            }

            if (time < best_time) {
                best_time = time;
                best_batch_size = batch_size;
            }
        }
        return best_batch_size;
#endif
    }

    struct NNTask {
        NNTask(torch::Tensor input_, torch::Tensor &output_, int num_chunks_)
                : input(input_), out(output_), num_chunks(num_chunks_) {}
        torch::Tensor input;
        torch::Tensor &out;
        std::mutex mut;
        std::condition_variable cv;
        bool done{false};
        int num_chunks;
    };

    std::vector<DecodedChunk> call_chunks(torch::Tensor &input,
                                          torch::Tensor &output,
                                          int num_chunks,
                                          c10::cuda::CUDAStream stream) {
        c10::cuda::CUDAStreamGuard stream_guard(stream);

        if (num_chunks == 0) {
            return std::vector<DecodedChunk>();
        }

        auto task = std::make_shared<NNTask>(input, output, num_chunks);
        {
            std::lock_guard<std::mutex> lock(m_input_lock);
            m_input_queue.push_front(task);
        }
        m_input_cv.notify_one();

        LOG_DEBUG("%s", "running nn");
        std::unique_lock<std::mutex> lock(task->mut);
        while (!task->done) {
            task->cv.wait(lock);
        }

        LOG_DEBUG("%s", "decoding chunks");
        return m_decoder->cpu_part(output);
    }

    void cuda_thread_fn() {
        torch::InferenceMode guard;
        c10::cuda::CUDAGuard device_guard(m_options.device());
        auto stream = c10::cuda::getCurrentCUDAStream(m_options.device().index());

        while (true) {
            std::unique_lock<std::mutex> input_lock(m_input_lock);
            while (m_input_queue.empty() && !m_terminate.load()) {
                m_input_cv.wait_for(input_lock, 100ms);
            }
            
            if (m_input_queue.empty() && m_terminate.load()) {
                return;
            }

            auto task = m_input_queue.back();
            m_input_queue.pop_back();
            input_lock.unlock();

            auto gpu_lock = acquire_gpu_lock(m_options.device().index(),
                                                            m_exclusive_gpu_access);
            std::unique_lock<std::mutex> task_lock(task->mut);

            auto run_basecalling = [&]() {
                auto scores = m_module->forward(task->input.to(m_options.device(), true));
                task->out.copy_(m_decoder->gpu_part(scores, task->num_chunks, m_decoder_options));
                stream.synchronize();
            };

            try {
                run_basecalling();
            } catch (c10::Error &e) {
                c10::cuda::CUDACachingAllocator::emptyCache();
                run_basecalling();
            }
            task->done = true;
            task_lock.unlock();
            task->cv.notify_one();
        }
    }

    void terminate() {
        m_terminate.store(true);
        m_input_cv.notify_one();
        if (m_cuda_thread && m_cuda_thread->joinable()) {
            m_cuda_thread->join();
        }
        m_cuda_thread.reset();
    }

    void restart() {
        // This can be called more than one, via multiple runners.
        if (m_terminate.load()) {
            m_terminate.store(false);
            start_threads();
        }
    }

    const CRFModelConfig m_config;
    std::string m_device;
    torch::TensorOptions m_options;
    std::unique_ptr<GPUDecoder> m_decoder;
    DecoderOptions m_decoder_options;
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
    std::atomic<bool> m_terminate{false};
    std::deque<std::shared_ptr<NNTask>> m_input_queue;
    std::mutex m_input_lock;
    std::condition_variable m_input_cv;
    std::unique_ptr<std::thread> m_cuda_thread;
    int m_num_input_features, m_batch_size, m_in_chunk_size, m_out_chunk_size;
    bool m_exclusive_gpu_access;
};

std::shared_ptr<CudaCaller> create_cuda_caller(const CRFModelConfig &model_config,
                                               int chunk_size,
                                               int batch_size,
                                               const std::string &device,
                                               float memory_limit_fraction,
                                               bool exclusive_gpu_access) {
    return std::make_shared<CudaCaller>(model_config, chunk_size, batch_size, device,
                                        memory_limit_fraction, exclusive_gpu_access);
}

CudaModelRunner::CudaModelRunner(std::shared_ptr<CudaCaller> caller)
        : m_caller(caller),
          m_stream(c10::cuda::getStreamFromPool(false, m_caller->m_options.device().index())) {
    auto opts = torch::TensorOptions().device(torch::kCPU).pinned_memory(true);
    m_input = torch::empty(
            {caller->m_batch_size, caller->m_num_input_features, caller->m_in_chunk_size},
            opts.dtype(m_caller->m_options.dtype()));
    m_output = torch::empty({3, caller->m_batch_size, caller->m_out_chunk_size},
                            opts.dtype(torch::kInt8));
}

void CudaModelRunner::accept_chunk(int chunk_idx, const torch::Tensor &chunk) {
    m_input.index_put_({chunk_idx, torch::indexing::Ellipsis}, chunk);
}

std::vector<DecodedChunk> CudaModelRunner::call_chunks(int num_chunks) {
    auto decoded_chunks = m_caller->call_chunks(m_input, m_output, num_chunks, m_stream);
    return decoded_chunks;
}

const CRFModelConfig &CudaModelRunner::config() const { return m_caller->m_config; }
size_t CudaModelRunner::model_stride() const { return m_caller->m_config.stride; }
size_t CudaModelRunner::chunk_size() const { return m_input.size(2); }
size_t CudaModelRunner::batch_size() const { return m_input.size(0); }
void CudaModelRunner::terminate() { m_caller->terminate(); }
void CudaModelRunner::restart() { m_caller->restart(); }
