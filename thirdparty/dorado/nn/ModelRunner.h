#pragma once

#include "../decode/Decoder.h"
#include "CRFModel.h"
#include "../decode/CPUDecoder.h"

#include "toml.h"
#include "error.h"
#include <torch/torch.h>

#include <string>

class ModelRunnerBase {
public:
    virtual void accept_chunk(int chunk_idx, const torch::Tensor& slice) = 0;
    virtual std::vector<DecodedChunk> call_chunks(int num_chunks) = 0;
    virtual size_t model_stride() const = 0;
    virtual size_t chunk_size() const = 0;
    virtual const CRFModelConfig &config() const = 0;
    virtual size_t batch_size() const = 0;
    virtual void terminate() = 0;
    virtual void restart() = 0;
};

using Runner = std::shared_ptr<ModelRunnerBase>;

template <typename T>
class ModelRunner final : public ModelRunnerBase {
public:
    ModelRunner(const CRFModelConfig &model_config,
                const std::string &device,
                int chunk_size,
                int batch_size);
    void accept_chunk(int chunk_idx, const torch::Tensor& slice) final;
    std::vector<DecodedChunk> call_chunks(int num_chunks) final;
    const CRFModelConfig &config() const final { return m_config; };
    size_t model_stride() const final { return m_config.stride; }
    size_t chunk_size() const final { return m_input.size(2); }
    size_t batch_size() const final { return m_input.size(0); }
    void terminate() final {}
    void restart() final {}

private:
    std::string m_device;
    const CRFModelConfig m_config;
    torch::Tensor m_input;
    torch::TensorOptions m_options;
    std::unique_ptr<T> m_decoder;
    DecoderOptions m_decoder_options;
    torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
};

template <typename T>
ModelRunner<T>::ModelRunner(const CRFModelConfig &model_config,
                            const std::string &device,
                            int chunk_size,
                            int batch_size)
        : m_config(model_config) {
    m_decoder_options = DecoderOptions();
    m_decoder_options.q_shift = model_config.qbias;
    m_decoder_options.q_scale = model_config.qscale;
    m_decoder = std::make_unique<T>();

    m_options = torch::TensorOptions().dtype(T::dtype).device(device);
    m_module = load_crf_model(model_config, m_options);

    // adjust chunk size to be a multiple of the stride
    chunk_size -= chunk_size % model_config.stride;

    m_input = torch::zeros({batch_size, model_config.num_features, chunk_size},
                           torch::TensorOptions().dtype(T::dtype).device(torch::kCPU));
}

template <typename T>
std::vector<DecodedChunk> ModelRunner<T>::call_chunks(int num_chunks) {
    torch::InferenceMode guard;
    auto scores = m_module->forward(m_input.to(m_options.device_opt().value()));
    auto decoded_chunks = m_decoder->beam_search(scores, num_chunks, m_decoder_options);
    return decoded_chunks;
}

template <typename T>
void ModelRunner<T>::accept_chunk(int chunk_idx, const torch::Tensor &chunk) {
    m_input.index_put_({chunk_idx, torch::indexing::Ellipsis}, chunk);
}

