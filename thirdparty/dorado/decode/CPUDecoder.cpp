#include "CPUDecoder.h"

#include "beam_search.h"
#include "error.h"

#include <math.h>

#include <vector>

void backward_scan(const float *scores_in, float *out, const int chunk, const int T, const int N, const int num_states) {
    const int kNumBases = 4;
    const int kNumTransitions = kNumBases + 1;
    const float kFixedStayScore = 2.0f;

    const int ts_states = num_states * kNumBases;

    const float* const chunk_in = scores_in + chunk * ts_states; // should be half float (for GPU impl)
    float* const chunk_out = out + chunk * (T+1) * num_states;
    float* const alpha_init = chunk_out + num_states * T;
    for (int state = 0; state < num_states; ++state) { // (for GPU impl) its 1 thread per state, but below we iterate through all the states on 1 thread
        alpha_init[state] = 0.0f;
    }

    for (int ts = 0; ts < T; ++ts) {
        // threadgroup_barrier(mem_flags::mem_device); // synchronize all threads before next time step (for GPU impl)
        const float* const ts_in = chunk_in + N * ts_states * (T - ts - 1);
        float* const ts_alpha_in = alpha_init - num_states * ts;
        float* const ts_alpha_out = ts_alpha_in - num_states;

        for (int state = 0; state < num_states; ++state) { // we should have 1 thread for each state (for GPU impl)
            const int stay_state_idx = state;
            const int step_state_idx_a = (state * kNumBases) % num_states;
            const int step_trans_idx_a = step_state_idx_a * kNumBases +
                ((state * kNumBases) / num_states);

            float vals[kNumTransitions];
            float max_val = vals[0] = ts_alpha_in[stay_state_idx] + kFixedStayScore;
            for (int base = 0; base < kNumBases; ++base) {
                vals[base + 1] = ts_alpha_in[step_state_idx_a + base] +
                    ts_in[step_trans_idx_a + base * kNumBases];
                max_val = max_val > vals[base + 1] ? max_val : vals[base + 1];
            }
            float sum = 0.0f;
            for (int i = 0; i < kNumTransitions; ++i) {
                sum += exp(vals[i] - max_val);
            }
            ts_alpha_out[state] = max_val + log(sum);
        }
    }
}

void forward_scan(const float *scores_in, const float *bwd, float *out, const int chunk, const int _T, const int N, const int num_states) {
    const int T = _T+1; 
    constexpr int kNumBases = 4;
    constexpr int kNumTransitions = kNumBases + 1;
    constexpr float kFixedStayScore = 2.0f;
    
    const int kMsb = num_states / kNumBases;
    const int ts_states = num_states * kNumBases;

    // This batch element's scores.
    const float* const chunk_scores = scores_in + chunk * ts_states;

    // Alternating forward guide buffers used for successive time steps.
    constexpr int kMaxStates = 1024;
    float ts_fwd[2][kMaxStates]; // threadgroup

    // The forward guide input for the first step is 0.
    for (int state = 0; state < num_states; ++state) {
        ts_fwd[0][state] = 0.0f;
    }
    // threadgroup_barrier(mem_flags::mem_threadgroup); // ------------------------------------------------------------------

    for (int ts = 0; ts < T; ++ts) {
        // We read forward guide values written to TG memory in the previous step as
        // inputs to this step.  However, there has already been a TG barrier since
        // they were written.
        const int ts_idx = (chunk * T + ts) * num_states;

        // This time step's scores.
        const float* const ts_scores = chunk_scores + N * ts_states * ts;

        // Alternating TG buffer twiddling.
        const float* const ts_alpha_in = ts_fwd[ts & 1];
        float* const ts_alpha_out = ts_fwd[(ts & 1) ^ 1];

        // Calculate the next time step's forward guide from this time step's scores
        // and forward guide.  It's written to threadgroup memory for use in the
        // next iteration.
        for (int state = 0; state < num_states; ++state) { // we should have 1 thread for each state (for GPU impl)
            const int stay_state_idx = state;
            const int step_state_idx_a = state / kNumBases;
            const int step_trans_idx_a = state * kNumBases;
            float vals[kNumTransitions];
            float fwd_max_val = vals[0] = ts_alpha_in[stay_state_idx] + kFixedStayScore;
            for (int base = 0; base < kNumBases; ++base) {
                vals[base + 1] = ts_alpha_in[step_state_idx_a + base * kMsb] + 
                    ts_scores[step_trans_idx_a + base];
                fwd_max_val = std::max(fwd_max_val, vals[base + 1]);
            }
            float fwd_sum = 0.0f;
            for (int i = 0; i < kNumTransitions; ++i) {
                fwd_sum += exp(vals[i] - fwd_max_val);
            }
            ts_alpha_out[state] = fwd_max_val + log(fwd_sum);

            // Load the forward guide value calculated in the last time step for use
            // in this time step's posterior probability calculation.
            const float fwd_val = ts_alpha_in[state];

            // Calculate fwd/bwd guide product in log space.
            const float val = fwd_val + bwd[ts_idx + state];
            out[ts_idx + state] = val;
        }
    }
}

void softmax(const float *fwd, float *out, const int chunk, const int _T, const int num_states) {
    const int T = _T+1; 
    for (int ts = 0; ts < T; ++ts) {
        const int ts_idx = (chunk * T + ts) * num_states;

        float max_val = fwd[ts_idx];
        for (int state = 0; state < num_states; ++state) {
            
            max_val = max_val > fwd[ts_idx + state] ? max_val : fwd[ts_idx + state];
        }

        float exp_sum = 0;
        float exp_vals[num_states];
        for (int state = 0; state < num_states; ++state) {
            const float val = fwd[ts_idx + state];
            const float exp_val = exp(val - max_val);
            exp_vals[state] = exp_val;
            exp_sum += exp_val;
        }

        for (int state = 0; state < num_states; ++state) {
            const float exp_val = exp_vals[state];

            // Write out the posterior probability 
            out[ts_idx + state] = (float)(exp_val / exp_sum);
        }
    }
}

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