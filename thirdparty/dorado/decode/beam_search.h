#pragma once

#include "torch/torch.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// 16 bit state supports 7-mers with 4 bases.
typedef int16_t state_t;

typedef struct beam_element {
    state_t state;
    uint8_t prev_element_index;
    bool stay;
} beam_element_t;

void generate_sequence(
    const uint8_t* moves,
    const int32_t* states,
    const float* qual_data,
    const float shift,
    const float scale,
    const size_t num_ts,
    const size_t seq_len,
    float* base_probs,
    float* total_probs,
    char* sequence,
    char* qstring
);

float beam_search(
    const float* const scores,
    size_t scores_block_stride,
    const float* const back_guide,
    const float* const posts,
    const int num_state_bits,
    const size_t num_ts,
    const size_t max_beam_width,
    const float beam_cut,
    const float fixed_stay_score,
    int32_t* states,
    uint8_t* moves,
    float* qual_data,
    float score_scale,
    float posts_scale,
    beam_element_t* beam_vector
);