#pragma once

#include "CRFModel.h"
#include "../decode/decode_cpu.h"
#include "../decode/decode_gpu.h"

#include "toml.h"
#include "error.h"
#include <torch/torch.h>

#include <string>

class ModelRunnerBase {
public:
    virtual void accept_chunk(int chunk_idx, at::Tensor slice) = 0;
    virtual std::vector<DecodedChunk> call_chunks(int num_chunks) = 0;
    virtual size_t model_stride() const = 0;
    virtual size_t chunk_size() const = 0;
};

using Runner = std::shared_ptr<ModelRunnerBase>;

class ModelRunner : public ModelRunnerBase {
public:
    template <typename T>
    ModelRunner(const std::string &model_path,
                const std::string &device,
                int chunk_size,
                int batch_size,
                T dtype);
                
    void accept_chunk(int num_chunks, at::Tensor slice) final {
        m_input.index_put_({num_chunks, 0}, slice);
    }

    std::vector<DecodedChunk> call_chunks(int num_chunks) final {
        torch::InferenceMode guard;
        auto scores = (m_module->forward(m_input.to(m_options.device_opt().value())));
#ifdef USE_CUDA_LSTM
        return decode_gpu(scores, num_chunks, m_decoder_options, m_device, m_model_config);
#else
        return decode_cpu(scores, num_chunks, m_decoder_options, m_device, m_model_config);
#endif
    }

    size_t model_stride() const final { return m_model_stride; }
    size_t chunk_size() const final { return m_input.size(2); }

private:
    std::string m_device;
    torch::Tensor m_input;
    torch::TensorOptions m_options;
    DecoderOptions m_decoder_options;
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
    size_t m_model_stride;
    CRFModelConfig m_model_config;
};

template <typename T>
ModelRunner::ModelRunner(const std::string &model_path,
                            const std::string &device,
                            int chunk_size,
                            int batch_size,
                            T dtype) {
    const auto model_config = load_crf_model_config(model_path);
    m_model_stride = static_cast<size_t>(model_config.stride);

    m_decoder_options = DecoderOptions();
    m_decoder_options.q_shift = model_config.qbias;
    m_decoder_options.q_scale = model_config.qscale;
    m_device = device;
    m_model_config = model_config;

    LOG_DEBUG("initialized model runner for device %s", device.c_str());

#ifdef USE_GPU
    m_options = torch::TensorOptions().dtype(dtype).device(device); //todo
    m_module = load_crf_model(model_path, model_config, batch_size, chunk_size, m_options);
    chunk_size -= chunk_size % m_model_stride;
    m_input = torch::zeros({batch_size, 1, chunk_size}, torch::TensorOptions().dtype(dtype).device(torch::kCPU)); //todo
#else
    m_options = torch::TensorOptions().dtype(dtype).device(device); //todo
    m_module = load_crf_model(model_path, model_config, batch_size, chunk_size, m_options);
    chunk_size -= chunk_size % m_model_stride;
    m_input = torch::zeros({batch_size, 1, chunk_size}, torch::TensorOptions().dtype(dtype).device(torch::kCPU)); //todo
#endif
}
