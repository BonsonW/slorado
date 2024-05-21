#include "CPUDecoder.h"

#include "beam_search.h"
#include "error.h"

#include <math.h>

#include <vector>

at::Tensor scan(const torch::Tensor& Ms,
                const float fixed_stay_score,
                const torch::Tensor& idx,
                const torch::Tensor& v0) {
    const int T = Ms.size(0);
    const int N = Ms.size(1);
    const int C = Ms.size(2);

    torch::Tensor alpha = Ms.new_full({T + 1, N, C}, -1E38);
    alpha[0] = v0;

    for (int t = 0; t < T; t++) {
        auto scored_steps = torch::add(alpha.index({t, torch::indexing::Slice(), idx}), Ms[t]);
        auto scored_stay = torch::add(alpha.index({t, torch::indexing::Slice()}), fixed_stay_score)
                                   .unsqueeze(-1);
        auto scored_transitions = torch::cat({scored_stay, scored_steps}, -1);

        alpha[t + 1] = torch::logsumexp(scored_transitions, -1);
    }

    return alpha;
}

torch::Tensor forward_scores(const torch::Tensor& scores, const float fixed_stay_score) {
    const int T = scores.size(0);  // Signal len
    const int N = scores.size(1);  // Num batches
    const int C = scores.size(2);  // 4^state_len * 4 = 4^(state_len + 1)

    const int n_base = 4;
    const int state_len = std::log(C) / std::log(n_base) - 1;

    // Transition scores reshaped so that the 4 scores for each predecessor state are arranged along the
    // innermost dimension.
    const torch::Tensor Ms = scores.reshape({T, N, -1, n_base});

    // Number of states per timestep.
    const int num_states = pow(n_base, state_len);

    // Guide values at first timestep.
    const auto v0 = Ms.new_full({{N, num_states}}, 0.0f);

    // For each state, the indices of the 4 states that could precede it via a step transition.
    const auto idx = torch::arange(num_states)
                             .repeat_interleave(n_base)
                             .reshape({n_base, -1})
                             .t()
                             .contiguous();

    return scan(Ms, fixed_stay_score, idx, v0);
}

torch::Tensor backward_scores(const torch::Tensor& scores, const float fixed_stay_score) {
    const int N = scores.size(1);  // Num batches
    const int C = scores.size(2);  // 4^state_len * 4 = 4^(state_len + 1)

    const int n_base = 4;

    const int state_len = std::log(C) / std::log(n_base) - 1;

    // Number of states per timestep.
    const int num_states = pow(n_base, state_len);

    // Guide values at last timestep.
    const torch::Tensor vT = scores.new_full({N, num_states}, 0.0f);

    const auto idx = torch::arange(num_states)
                             .repeat_interleave(n_base)
                             .reshape({n_base, -1})
                             .t()
                             .contiguous();
    auto idx_T = idx.flatten().argsort().reshape(idx.sizes());

    const auto Ms_T = scores.index({torch::indexing::Slice(), torch::indexing::Slice(), idx_T});

    // For each state, the indices of the 4 states that could succeed it via a step transition.
    idx_T = torch::bitwise_right_shift(idx_T, 2);

    return scan(Ms_T.flip(0), fixed_stay_score, idx_T.to(torch::kInt64), vT).flip(0);
}

typedef struct {
    const DecoderOptions *options;
    const torch::Tensor *scores_cpu;
    std::vector<torch::Tensor> *scores;
    std::vector<torch::Tensor> *bwd;
    std::vector<torch::Tensor> *posts;
    int32_t tid;
    int32_t start;
    int32_t end;
} scores_guide_thread_arg_t;

void* pthread_scores_guide_single(void* voidargs) {
    scores_guide_thread_arg_t* args = (scores_guide_thread_arg_t*)voidargs;
    const DecoderOptions *options = args->options;

    using Slice = torch::indexing::Slice;
    // we slice on the second index because we want to split the batch up not the chunk up
    // (first index is timestep in chunk, second index is chunk in batch)
    auto t_scores = args->scores_cpu->index({Slice(), Slice(args->start, args->end)});

    torch::Tensor t_fwd = forward_scores(t_scores, options->blank_score);
    torch::Tensor t_bwd = backward_scores(t_scores, options->blank_score);

    torch::Tensor t_posts = torch::softmax(t_fwd + t_bwd, -1);

    t_scores = t_scores.transpose(0, 1);
    t_bwd = t_bwd.transpose(0, 1).contiguous();
    t_posts = t_posts.transpose(0, 1).contiguous();

    int tid = args->tid;
    (*args->scores)[tid] = t_scores;
    (*args->bwd)[tid] = t_bwd;
    (*args->posts)[tid] = t_posts;

    pthread_exit(0);
}

typedef struct {
    const DecoderOptions *options;
    std::vector<torch::Tensor> *scores;
    std::vector<torch::Tensor> *bwd;
    std::vector<torch::Tensor> *posts;
    std::vector<DecodedChunk> *chunk_results;
    int32_t tid;
    int32_t start;
    int32_t end;
} decode_thread_arg_t;

void* pthread_single_beam_search(void* voidargs) {
    decode_thread_arg_t* args = (decode_thread_arg_t*)voidargs;
    const DecoderOptions *options = args->options;

    int i = 0;
    int tid = args->tid;
    for (int c = args->start; c < args->end; c++, i++) {
        auto decode_result = beam_search_decode(
                args->scores[tid][i], args->bwd[tid][i], args->posts[tid][i], options->beam_width, options->beam_cut,
                options->blank_score, options->q_shift, options->q_scale,
                options->temperature, 1.0f);
        (*args->chunk_results)[c] = DecodedChunk{
                std::get<0>(decode_result),
                std::get<1>(decode_result),
                std::get<2>(decode_result),
        };
    }

    pthread_exit(0);
}

std::vector<DecodedChunk> beam_search_cpu(const torch::Tensor& _scores,
                                                  const int num_chunks,
                                                  const DecoderOptions& options,
                                                  std::string &device) {
    const auto scores_cpu = _scores.to(torch::kCPU).to(CPUDecoder::dtype).transpose(0, 1);
    int num_threads = std::min(num_chunks, 1);
    int chunks_per_thread = num_chunks / num_threads;
    int num_threads_with_one_more_chunk = num_chunks % num_threads;

    std::vector<torch::Tensor> scores(num_threads);
    std::vector<torch::Tensor> bwd(num_threads);
    std::vector<torch::Tensor> posts(num_threads);

    std::vector<DecodedChunk> chunk_results(num_chunks);

    // create threads
    pthread_t tids[num_threads];
    scores_guide_thread_arg_t scores_pt_args[num_threads];
    decode_thread_arg_t pt_args[num_threads];
    int32_t t, ret;

    // prep tensors
    for (t = 0; t < num_threads; t++) {
        scores_pt_args[t].tid = t;
        scores_pt_args[t].scores = &scores;
        scores_pt_args[t].bwd = &bwd;
        scores_pt_args[t].posts = &posts;
        scores_pt_args[t].start = t * chunks_per_thread + std::min(t, num_threads_with_one_more_chunk);
        scores_pt_args[t].end = pt_args[t].start + chunks_per_thread + int(t < num_threads_with_one_more_chunk);
        scores_pt_args[t].scores_cpu = &scores_cpu;
        scores_pt_args[t].options = &options;
    }
    
    for (t = 0; t < num_threads; t++) {
        ret = pthread_create(&tids[t], NULL, pthread_scores_guide_single, (void*)(&scores_pt_args[t]));
        NEG_CHK(ret);
    }

    for (t = 0; t < num_threads; t++) {
        ret = pthread_join(tids[t], NULL);
        NEG_CHK(ret);
    }

    // decode
    for (t = 0; t < num_threads; t++) {
        pt_args[t].tid = t;
        pt_args[t].scores = &scores;
        pt_args[t].bwd = &bwd;
        pt_args[t].posts = &posts;
        pt_args[t].start = t * chunks_per_thread + std::min(t, num_threads_with_one_more_chunk);
        pt_args[t].end = pt_args[t].start + chunks_per_thread + int(t < num_threads_with_one_more_chunk);
        pt_args[t].chunk_results = &chunk_results;
        pt_args[t].options = &options;
    }

    for (t = 0; t < num_threads; t++) {
        ret = pthread_create(&tids[t], NULL, pthread_single_beam_search, (void*)(&pt_args[t]));
        NEG_CHK(ret);
    }

    for (t = 0; t < num_threads; t++) {
        ret = pthread_join(tids[t], NULL);
        NEG_CHK(ret);
    }

    return chunk_results;
}

std::vector<DecodedChunk> CPUDecoder::beam_search(const torch::Tensor& scores,
                                                  const int num_chunks,
                                                  const DecoderOptions& options,
                                                  std::string &device) {
    return beam_search_cpu(scores, num_chunks, options, device);
}