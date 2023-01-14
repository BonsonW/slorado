#pragma once

#include "../decode/Decoder.h"
#include "CRFModel.h"

#include <toml.hpp>
#include <torch/torch.h>

#include <string>

class ModelRunnerBase {
public:
    virtual void accept_chunk(int chunk_idx, at::Tensor slice) = 0;
    virtual torch::Tensor call_chunks() = 0;
    virtual std::vector<DecodedChunk> decode_chunks(const torch::Tensor &scores, int num_chunks) = 0;
    virtual size_t model_stride() const = 0;
    virtual size_t chunk_size() const = 0;
};

using Runner = std::shared_ptr<ModelRunnerBase>;

template <typename T>
class ModelRunner : public ModelRunnerBase {
public:
    ModelRunner(const std::string &model_path,
                const std::string &device,
                int chunk_size,
                int batch_size);
    void accept_chunk(int chunk_idx, at::Tensor slice) final;
    torch::Tensor call_chunks() final;
    std::vector<DecodedChunk> decode_chunks(const torch::Tensor &scores, int num_chunks) final;
    size_t model_stride() const final { return m_model_stride; }
    size_t chunk_size() const final { return m_input.size(2); }

private:
    std::string m_device;
    torch::Tensor m_input;
    torch::TensorOptions m_options;
    std::unique_ptr<T> m_decoder;
    DecoderOptions m_decoder_options;
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
    size_t m_model_stride;
};

template <typename T>
ModelRunner<T>::ModelRunner(const std::string &model_path,
                            const std::string &device,
                            int chunk_size,
                            int batch_size) {
    const auto model_config = load_crf_model_config(model_path);
    m_model_stride = static_cast<size_t>(model_config.stride);

    m_decoder_options = DecoderOptions();
    m_decoder_options.q_shift = model_config.qbias;
    m_decoder_options.q_scale = model_config.qscale;
    m_decoder = std::make_unique<T>();

#ifdef USE_GPU
    if (device == "cpu") {
        m_options = torch::TensorOptions().dtype(torch::kF32).device(device); //todo
        m_module = load_crf_model(model_path, model_config, batch_size, chunk_size, m_options);
        chunk_size -= chunk_size % m_model_stride;
        m_input = torch::zeros({batch_size, 1, chunk_size}, torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU)); //todo
    } else {
#ifdef USE_KOI
        m_options = torch::TensorOptions().dtype(torch::kF16).device(device); //todo
        m_module = load_crf_model(model_path, model_config, batch_size, chunk_size, m_options);
        chunk_size -= chunk_size % m_model_stride;
        m_input = torch::zeros({batch_size, 1, chunk_size}, torch::TensorOptions().dtype(torch::kF16).device(torch::kCPU)); //todo
#else
        m_options = torch::TensorOptions().dtype(torch::kF32).device(device); //todo
        m_module = load_crf_model(model_path, model_config, batch_size, chunk_size, m_options);
        chunk_size -= chunk_size % m_model_stride;
        m_input = torch::zeros({batch_size, 1, chunk_size}, torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU)); //todo
#endif
    }
#else
    m_options = torch::TensorOptions().dtype(torch::kF32).device(device); //todo
    m_module = load_crf_model(model_path, model_config, batch_size, chunk_size, m_options);
    chunk_size -= chunk_size % m_model_stride;
    m_input = torch::zeros({batch_size, 1, chunk_size}, torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU)); //todo
#endif
}

template<typename T> torch::Tensor ModelRunner<T>::call_chunks() {
    torch::InferenceMode guard;
    return m_module->forward(m_input.to(m_options.device_opt().value()));
}

template<typename T> std::vector<DecodedChunk> ModelRunner<T>::decode_chunks(const torch::Tensor &scores, int num_chunks) {
    return m_decoder->beam_search(scores, num_chunks, m_decoder_options);
}

template<typename T> void ModelRunner<T>::accept_chunk(int num_chunks, at::Tensor slice) {
    m_input.index_put_({num_chunks, 0}, slice);
}

