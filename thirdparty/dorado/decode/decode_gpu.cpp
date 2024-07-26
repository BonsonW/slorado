#include "decode_gpu.h"
#include "../nn/CRFModel.h"
#include "decode.h"
#include "error.h"

torch::Tensor decode_gpu_single(torch::Tensor scores, int num_chunks, const runner_t *runner) {
    ERROR("%s", "not implemented yet");
    exit(EXIT_FAILURE);
    return torch::empty({1});
}

std::vector<DecodedChunk> collect_gpu_decoded_chunks(torch::Tensor moves_sequence_qstring_cpu) {
    ERROR("%s", "not implemented yet");
    exit(EXIT_FAILURE);
    return std::vector<DecodedChunk>();
}

std::vector<DecodedChunk> decode_gpu(const torch::Tensor& scores, const int num_chunks, const core_t *core, const int runner_idx) {
    const runner_t *runner = (*core->runners)[runner_idx];
    return collect_gpu_decoded_chunks(decode_gpu_single(scores, num_chunks, runner).to(torch::kCPU));
}
