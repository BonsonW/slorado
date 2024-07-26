#include "decode_cpu.h"
#include "../nn/CRFModel.h"
#include "beam_search.h"
#include "error.h"

#include <math.h>
#include <vector>

using Slice = torch::indexing::Slice;

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
        // threadgroup_barrier(mem_flags::medevice); // synchronize all threads before next time step (for GPU impl)
        const float* const ts_in = chunk_in + N * ts_states * (T - ts - 1);
        float* const ts_alpha_in = alpha_init - num_states * ts;
        float* const ts_alpha_out = ts_alpha_in - num_states;

        for (int state = 0; state < num_states; ++state) { // we should have 1 thread for each state (for GPU impl)
            const int stay_state_idx = state;
            const int step_state_idx_a = (state * kNumBases) % num_states;
            const int step_trans_idx_a = step_state_idx_a * kNumBases +
                ((state * kNumBases) / num_states);

            float vals[kNumTransitions];
            vals[0] = ts_alpha_in[stay_state_idx] + kFixedStayScore;
            float max_val = vals[0];
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
                // todo: this is a bandaid for indexing past the actual T dimension of scores
                // need to verify with actual MetalTxCaller impl output,
                // otherwise output remains exactly the same for this impl whether it indexes past or not
                float ts_score = ts < _T ? ts_scores[step_trans_idx_a + base] : 0.0f;

                vals[base + 1] = ts_alpha_in[step_state_idx_a + base * kMsb] + ts_score;
                fwd_max_val = fwd_max_val > vals[base + 1] ? fwd_max_val : vals[base + 1];
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

typedef struct {
    const DecoderOptions *options;
    const torch::Tensor *scores_TNC;
    torch::Tensor *bwd_NTC;
    torch::Tensor *fwd_NTC;
    torch::Tensor *post_NTC;
    std::vector<DecodedChunk> *chunk_results;
    int32_t start;
    int32_t end;
    const CRFModelConfig *config;
} decode_thread_arg_t;

void* pthread_single_beam_search(void* voidargs) {
    decode_thread_arg_t* args = (decode_thread_arg_t*)voidargs;
    const DecoderOptions *options = args->options;

    const int n_base = 4; // should honor model config
    const int m_states = std::pow(n_base, args->config->state_len);

    const float *scores_in = (float *)args->scores_TNC->data_ptr();
    float *bwd_out = (float *)args->bwd_NTC->data_ptr();
    float *fwd_out = (float *)args->fwd_NTC->data_ptr();
    float *post_out = (float *)args->post_NTC->data_ptr();

    const int T = args->scores_TNC->size(0);
    const int N = args->scores_TNC->size(1);

    for (int c = args->start; c < args->end; c++) {
        backward_scan(scores_in, bwd_out, c, T, N, m_states);
        forward_scan(scores_in, bwd_out, fwd_out, c, T, N, m_states);
        softmax(fwd_out, post_out, c, T, m_states);

        auto bwd = args->bwd_NTC->index({c});
        auto post = args->post_NTC->index({c});
        auto scores = args->scores_TNC->index({Slice(), c});

        LOG_TRACE("bwd dimensions: %ld, %ld", scores.size(0), scores.size(1));
        LOG_TRACE("bwd dimensions: %ld, %ld", bwd.size(0), bwd.size(1));
        LOG_TRACE("post dimensions: %ld, %ld", post.size(0), post.size(1));

        auto decode_result = beam_search_decode(
                scores, bwd, post, options->beam_width, options->beam_cut,
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

std::vector<DecodedChunk> decode_cpu(const torch::Tensor& scores, const int num_chunks, const core_t *core, const int runner_idx) {
    const runner_t *runner = (*core->runners)[runner_idx];
    
    const auto options = runner->decoder_opts;
    const auto device = runner->device;
    const auto config = runner->model_config;

    const auto scores_TNC = scores.to(torch::kCPU).to(DTYPE_CPU).transpose(0, 1).contiguous();
    const int T = scores_TNC.size(0);
    const int N = scores_TNC.size(1);
    const int C = scores_TNC.size(2);
    const int n_base = 4;
    const int m_states = std::pow(n_base, config.state_len);

    LOG_TRACE("scores tensor dim: %d, %d, %d", T, N, C);

    torch::Tensor bwd_NTC = torch::empty({N, T + 1, m_states}).to(DTYPE_CPU).contiguous();
    torch::Tensor fwd_NTC = torch::empty({N, T + 1, m_states}).to(DTYPE_CPU).contiguous();
    torch::Tensor post_NTC = torch::empty({N, T + 1, m_states}).to(DTYPE_CPU).contiguous();
    
    std::vector<DecodedChunk> chunk_results(num_chunks);

    // create threads
    const int num_threads = std::min(num_chunks, 4);
    const int chunks_per_thread = num_chunks / num_threads;
    const int num_threads_with_one_more_chunk = num_chunks % num_threads;

    pthread_t tids[num_threads];
    decode_thread_arg_t pt_args[num_threads];
    int32_t t, ret;

    // set the data structures
    for (t = 0; t < num_threads; t++) {
        pt_args[t].start = t * chunks_per_thread + std::min(t, num_threads_with_one_more_chunk);
        pt_args[t].end = pt_args[t].start + chunks_per_thread + int(t < num_threads_with_one_more_chunk);
        pt_args[t].scores_TNC = &scores_TNC;
        pt_args[t].bwd_NTC = &bwd_NTC;
        pt_args[t].fwd_NTC = &fwd_NTC;
        pt_args[t].post_NTC = &post_NTC;
        pt_args[t].chunk_results = &chunk_results;
        pt_args[t].options = &options;
        pt_args[t].config = &config;
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
