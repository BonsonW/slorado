#pragma once

#include <string>
#include <torch/torch.h>
#include <toml.h>
#include "../decode/Decoder.h"
#include "CRFModel.h"


class ModelRunnerBase {
    public:
        virtual void accept_chunk(int chunk_idx, at::Tensor slice) = 0;
        virtual std::vector<DecodedChunk> call_chunks(int num_chunks) = 0;
};

using Runner = std::shared_ptr<ModelRunnerBase>;

template<typename T> class ModelRunner : public ModelRunnerBase {
    public:
        ModelRunner(const std::string &model, const std::string &device, int chunk_size, int batch_size);
        void accept_chunk(int chunk_idx, at::Tensor slice) final;
        std::vector<DecodedChunk> call_chunks(int num_chunks) final;
    private:
        std::string m_device;
        torch::Tensor m_input;
        torch::TensorOptions m_options;
        std::unique_ptr<T> m_decoder;
        DecoderOptions m_decoder_options;
        torch::nn::ModuleHolder<torch::nn::AnyModule> m_module{nullptr};
};

static inline void toml_error(const char* msg, const char* msg1)
{
    fprintf(stderr, "ERROR: %s%s\n", msg, msg1?msg1:"");
    exit(1);
}


template<typename T> ModelRunner<T>::ModelRunner(const std::string &model, const std::string &device, int chunk_size, int batch_size) {
    //todo:

    FILE* fp;
    char errbuf[200];

    // 1. Read and parse toml file
    std::string model_path = model + "/config.toml";
    fp = fopen(model_path.c_str(), "r");
    if (!fp) {
        fprintf(stderr,"cannot open %s - %s",model_path.c_str(), strerror(errno));
        exit(1);
    }
    toml_table_t* conf = toml_parse_file(fp, errbuf, sizeof(errbuf));
    fclose(fp);
    if (!conf) {
        toml_error("cannot parse - ", errbuf);
    }
    toml_table_t* qscore = toml_table_in(conf, "qscore");
    if (!qscore) {
        toml_error("missing [qscore]", "");
    }
    toml_datum_t bias = toml_double_in(qscore, "bias");
    if (!bias.ok) {
        toml_error("cannot read bias", "");
    }
    double qbias = bias.u.d;
    //fprintf(stderr, "bias: %f\n", qbias);

    toml_datum_t scale = toml_double_in(qscore, "scale");
    if (!scale.ok) {
        toml_error("cannot read scale", "");
    }
    double qscale = scale.u.d;
    //fprintf(stderr,"scale: %f\n", qscale);
    toml_free(conf);


    m_decoder_options = DecoderOptions();
    m_decoder_options.q_shift = qbias;
    m_decoder_options.q_scale = qscale;

    m_decoder = std::make_unique<T>();

#ifdef USE_GPU
    if (device == "cpu") {
        m_options = torch::TensorOptions().dtype(torch::kF32).device(device); //todo
        m_module = load_crf_model(model, batch_size, chunk_size, m_options);
        m_input = torch::zeros({batch_size, 1, chunk_size}, torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU)); //todo
    } else {
        m_options = torch::TensorOptions().dtype(torch::kF16).device(device); //todo
        m_module = load_crf_model(model, batch_size, chunk_size, m_options);
        m_input = torch::zeros({batch_size, 1, chunk_size}, torch::TensorOptions().dtype(torch::kF16).device(torch::kCPU)); //todo
    }
#else
    m_options = torch::TensorOptions().dtype(torch::kF32).device(device); //todo
    m_module = load_crf_model(model, batch_size, chunk_size, m_options);
    m_input = torch::zeros({batch_size, 1, chunk_size}, torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU)); //todo
#endif
}

template<typename T> std::vector<DecodedChunk> ModelRunner<T>::call_chunks(int num_chunks) {
    torch::InferenceMode guard;
    auto scores = m_module->forward(m_input.to(m_options.device_opt().value()));
    return m_decoder->beam_search(scores, num_chunks, m_decoder_options);
}

template<typename T> void ModelRunner<T>::accept_chunk(int num_chunks, at::Tensor slice) {
    m_input.index_put_({num_chunks, 0}, slice);
}
