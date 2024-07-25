#include "decode_gpu.h"
#include "../nn/CRFModel.h"
#include "decode.h"
#include "error.h"

torch::Tensor decode_gpu_single(torch::Tensor scores, int num_chunks, const runner_t *runner) {
    return torch::empty({1});
}

std::vector<DecodedChunk> collect_gpu_decoded_chunks(torch::Tensor moves_sequence_qstring_cpu) {
    return std::vector<DecodedChunk>();
}

std::vector<DecodedChunk> decode_gpu( const torch::Tensor &scores, int num_chunks, const runner_t *runner) {
    return collect_gpu_decoded_chunks(decode_gpu_single(scores, num_chunks, runner).to(torch::kCPU));
}
