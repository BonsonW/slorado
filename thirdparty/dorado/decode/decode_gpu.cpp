#include "decode_gpu.h"
#include "../nn/CRFModel.h"
#include "decode.h"
#include "error.h"

#ifdef USE_CUDA_LSTM
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
extern "C" {
#include "koi.h"
}
#endif

torch::Tensor decode_gpu_single(torch::Tensor scores, int num_chunks, const runner_t *runner) {
    const auto options = runner->m_decoder_options;
    const auto device = runner->m_device;
#ifdef USE_CUDA_LSTM
    long int N = scores.sizes()[0];
    long int C = scores.sizes()[2];

    // init
    auto chunks = runner->koi_chunks;
    auto chunk_results = runner->koi_chunk_results;
    auto aux = runner->koi_aux;
    auto path = runner->koi_path;
    auto moves_sequence_qstring = runner->koi_moves_sequence_qstring;

    moves_sequence_qstring.index({torch::indexing::Slice()}) = 0.0;
    auto moves = moves_sequence_qstring[0];
    auto sequence = moves_sequence_qstring[1];
    auto qstring = moves_sequence_qstring[2];

    host_back_guide_step(chunks.data_ptr(), chunk_results.data_ptr(), N, scores.data_ptr(), C,
                         aux.data_ptr(), path.data_ptr(), moves.data_ptr(), NULL,
                         sequence.data_ptr(), qstring.data_ptr(), options.q_scale, options.q_shift,
                         options.beam_width, options.beam_cut, options.blank_score);

    host_beam_search_step(chunks.data_ptr(), chunk_results.data_ptr(), N, scores.data_ptr(), C,
                          aux.data_ptr(), path.data_ptr(), moves.data_ptr(), NULL,
                          sequence.data_ptr(), qstring.data_ptr(), options.q_scale, options.q_shift,
                          options.beam_width, options.beam_cut, options.blank_score);

    host_compute_posts_step(chunks.data_ptr(), chunk_results.data_ptr(), N, scores.data_ptr(), C,
                            aux.data_ptr(), path.data_ptr(), moves.data_ptr(), NULL,
                            sequence.data_ptr(), qstring.data_ptr(), options.q_scale,
                            options.q_shift, options.beam_width, options.beam_cut,
                            options.blank_score);

    host_run_decode(chunks.data_ptr(), chunk_results.data_ptr(), N, scores.data_ptr(), C,
                    aux.data_ptr(), path.data_ptr(), moves.data_ptr(), NULL, sequence.data_ptr(),
                    qstring.data_ptr(), options.q_scale, options.q_shift, options.beam_width,
                    options.beam_cut, options.blank_score, options.move_pad);

    return moves_sequence_qstring.reshape({3, N, -1});
#else
    return torch::empty({1});
#endif
}

std::vector<DecodedChunk> collect_gpu_decoded_chunks(torch::Tensor moves_sequence_qstring_cpu) {
#ifdef USE_CUDA_LSTM
    assert(moves_sequence_qstring_cpu.device() == torch::kCPU);
    auto moves_cpu = moves_sequence_qstring_cpu[0];
    auto sequence_cpu = moves_sequence_qstring_cpu[1];
    auto qstring_cpu = moves_sequence_qstring_cpu[2];
    int N = moves_cpu.size(0);
    int T = moves_cpu.size(1);

    std::vector<DecodedChunk> called_chunks;

    for (int idx = 0; idx < N; idx++) {
        std::vector<uint8_t> mov((uint8_t *)moves_cpu[idx].data_ptr(),
                                 (uint8_t *)moves_cpu[idx].data_ptr() + T);
        auto num_bases = moves_cpu[idx].sum().item<int>();
        std::string seq((char *)sequence_cpu[idx].data_ptr(),
                        (char *)sequence_cpu[idx].data_ptr() + num_bases);
        std::string qstr((char *)qstring_cpu[idx].data_ptr(),
                         (char *)qstring_cpu[idx].data_ptr() + num_bases);

        called_chunks.emplace_back(DecodedChunk{std::move(seq), std::move(qstr), std::move(mov)});
    }

    return called_chunks;
#else
    return std::vector<DecodedChunk>();
#endif
}

std::vector<DecodedChunk> decode_gpu( const torch::Tensor &scores, int num_chunks, const runner_t *runner) {
    return collect_gpu_decoded_chunks(decode_gpu_single(scores, num_chunks, runner).to(torch::kCPU));
}
