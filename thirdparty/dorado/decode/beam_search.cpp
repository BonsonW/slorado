#include "beam_search.h"

#include "fast_hash.h"
#include "error.h"

#include <math.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>

//#define REMOVE_FIXED_BEAM_STAYS

// 16 bit state supports 7-mers with 4 bases.
typedef int16_t state_t;

constexpr int NUM_BASE_BITS = 2;
constexpr int NUM_BASES = 1 << NUM_BASE_BITS;

// This is the data we need to retain for the whole beam
struct BeamElement {
    state_t state;
    uint8_t prev_element_index;
    bool stay;
};

// This is the data we need to retain for only the previous timestep (block) in the beam
//  (and what we construct for the new timestep)
struct BeamFrontElement {
    uint32_t hash;
    state_t state;
    uint8_t prev_element_index;
    bool stay;
};

float log_sum_exp(float x, float y) {
    float abs_diff = std::abs(x - y);
    return std::max(x, y) + ((abs_diff < 17.0f) ? (std::log1p(std::exp(-abs_diff))) : 0.0f);
}

int get_num_states(size_t num_trans_states) {
    if (num_trans_states % NUM_BASES != 0) {
        throw std::runtime_error("Unexpected number of transition states in beam search decode.");
    }
    return int(num_trans_states / NUM_BASES);
}

static inline void generate_sequence(
    const uint8_t* moves,
    const int32_t* states,
    const float* qual_data,
    const float shift,
    const float scale,
    const size_t num_blocks,
    const size_t seq_len,
    float* base_probs,
    float* total_probs,
    char* sequence,
    char* qstring
) {
    size_t seq_pos = 0;

    const char alphabet[4] = {'A', 'C', 'G', 'T'};

    for (size_t i = 0; i < seq_len; ++i) {
        base_probs[i] = 0;
        total_probs[i] = 0;
    }

    for (size_t blk = 0; blk < num_blocks; ++blk) {
        int state = states[blk];
        int move = int(moves[blk]);
        int base = state & 3;
        int offset = (blk == 0) ? 0 : move - 1;
        int probPos = int(seq_pos) + offset;

        // get the probability for the called base.
        base_probs[probPos] += qual_data[blk * NUM_BASES + base];

        // accumulate the total probability for all possible bases at this position, for normalization.
        for (size_t k = 0; k < NUM_BASES; ++k) {
            total_probs[probPos] += qual_data[blk * NUM_BASES + k];
        }

        if (blk == 0) {
            sequence[seq_pos++] = char(base);
        } else {
            for (int j = 0; j < move; ++j) {
                sequence[seq_pos++] = char(base);
            }
        }
    }

    for (size_t i = 0; i < seq_len; ++i) {
        sequence[i] = alphabet[int(sequence[i])];
        base_probs[i] = 1.0f - (base_probs[i] / total_probs[i]);
        base_probs[i] = -10.0f * log10f(base_probs[i]);
        float qscore = base_probs[i] * scale + shift;
        if (qscore > 50.0f) qscore = 50.0f;
        if (qscore < 1.0f) qscore = 1.0f;
        qstring[i] = char(33.5f + qscore);
    }
}

// Incorporates NUM_NEW_BITS into a Castagnoli CRC32, aka CRC32C
// (not the same polynomial as CRC32 as used in zip/ethernet).
template <int NUM_NEW_BITS>
uint32_t crc32c(uint32_t crc, uint32_t new_bits) {
    // Note that this is the reversed polynomial.
    constexpr uint32_t POLYNOMIAL = 0x82f63b78u;
    for (int i = 0; i < NUM_NEW_BITS; ++i) {
        auto b = (new_bits ^ crc) & 1;
        crc >>= 1;
        if (b) {
            crc ^= POLYNOMIAL;
        }
        new_bits >>= 1;
    }
    return crc;
}

template <typename T, typename U>
float beam_search(
    const T* const scores,
    size_t scores_block_stride,
    const float* const back_guide,
    const U* const posts,
    const int num_state_bits,
    const size_t num_blocks,
    const size_t max_beam_width,
    const float beam_cut,
    const float fixed_stay_score,
    int32_t* states,
    uint8_t* moves,
    float* qual_data,
    float score_scale,
    float posts_scale,
    BeamElement* beam_vector,
    T* sorted_back_guides
) {
    const size_t num_states = 1ull << num_state_bits;
    const auto states_mask = static_cast<state_t>(num_states - 1);

    if (max_beam_width > 256) {
        throw std::range_error("Beamsearch max_beam_width cannot be greater than 256.");
    }

    // Some values we need
    constexpr uint32_t CRC_SEED = 0x12345678u;
    const float log_beam_cut = (beam_cut > 0.0f) ? logf(beam_cut) : std::numeric_limits<float>::max();

    // Create the previous and current beam fronts
    // Each existing element can be extended by one of NUM_BASES, or be a stay.
    const size_t max_beam_candidates = (NUM_BASES + 1) * max_beam_width;

    BeamFrontElement current_beam_front[max_beam_candidates];
    BeamFrontElement prev_beam_front[max_beam_candidates];

    float current_scores[max_beam_candidates];
    float prev_scores[max_beam_candidates];

    // Find the score an initial element needs in order to make it into the beam
    T beam_init_threshold = std::numeric_limits<T>::lowest();
    if (max_beam_width < num_states) {
        // Copy the first set of back guides and sort to extract max_beam_width highest elements
        std::memcpy(sorted_back_guides, back_guide, num_states * sizeof(T));

        // Note we don't need a full sort here to get the max_beam_width highest values
        std::nth_element(sorted_back_guides,
                         sorted_back_guides + max_beam_width - 1, sorted_back_guides + num_states,
                         std::greater<T>());
        beam_init_threshold = sorted_back_guides[max_beam_width - 1];
    }

    // Initialise the beam
    for (size_t state = 0, beam_element = 0; state < num_states && beam_element < max_beam_width;
         state++) {
        if (back_guide[state] >= beam_init_threshold) {
            // Note that this first element has a prev_element_index of 0
            prev_beam_front[beam_element] = {crc32c<32>(CRC_SEED, uint32_t(state)),
                                             static_cast<state_t>(state), 0, false};
            prev_scores[beam_element] = 0.0f;
            ++beam_element;
        }
    }

    // Copy this initial beam front into the beam persistent state
    size_t current_beam_width = std::min(max_beam_width, num_states);
    for (size_t element_idx = 0; element_idx < current_beam_width; ++element_idx) {
        beam_vector[element_idx].state = prev_beam_front[element_idx].state;
        beam_vector[element_idx].prev_element_index =
                (prev_beam_front)[element_idx].prev_element_index;
        beam_vector[element_idx].stay = prev_beam_front[element_idx].stay;
    }

    // Iterate through blocks, extending beam
    for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const T* const block_scores = scores + (block_idx * scores_block_stride);
        // Retrieves the given score as a float, multiplied by score_scale.
        const auto fetch_block_score = [block_scores, score_scale](size_t idx) {
            return static_cast<float>(block_scores[idx]) * score_scale;
        };
        const float* const block_back_scores = back_guide + ((block_idx + 1) << num_state_bits);

        /*  kmer transitions order:
         *  N^K , N array
         *  Elements stored as resulting kmer and modifying action (stays have a fixed score and are not computed).
         *  Kmer index is lexicographic with most recent base in the fastest index
         *
         *  E.g.  AGT has index (4^2, 4, 1) . (0, 2, 3) == 11
         *  The modifying action is
         *    0: Remove A from beginning
         *    1: Remove C from beginning
         *    2: Remove G from beginning
         *    3: Remove T from beginning
         *
         *  Transition (movement) ACGTT (111) -> CGTTG (446) has index 446 * 4 + 0 = 1784
         */

        float max_score = std::numeric_limits<float>::lowest();

        // Essentially a k=1 Bloom filter, indicating the presence of steps with particular
        // sequence hashes.  Avoids comparing stay hashes against all possible progenitor
        // states where none of them has the requisite sequence hash.
        const uint32_t HASH_PRESENT_BITS = 4096;
        const uint32_t HASH_PRESENT_MASK = HASH_PRESENT_BITS - 1;
        std::bitset<HASH_PRESENT_BITS> step_hash_present;  // Default constructor zeros content.

        // Generate list of candidate elements for this timestep (block).
        // As we do so, update the maximum score.
        size_t new_elem_count = 0;
        for (size_t prev_elem_idx = 0; prev_elem_idx < current_beam_width; ++prev_elem_idx) {
            const auto& previous_element = prev_beam_front[prev_elem_idx];

            // Expand all the possible steps
            for (int new_base = 0; new_base < NUM_BASES; new_base++) {
                state_t new_state =
                        (state_t((previous_element.state << NUM_BASE_BITS) & states_mask) |
                         state_t(new_base));
                const auto move_idx = static_cast<state_t>(
                        (new_state << NUM_BASE_BITS) +
                        (((previous_element.state << NUM_BASE_BITS) >> num_state_bits)));
                float new_score = prev_scores[prev_elem_idx] + fetch_block_score(move_idx) +
                                  static_cast<float>(block_back_scores[new_state]);
                uint32_t new_hash = crc32c<NUM_BASE_BITS>(previous_element.hash, new_base);

                step_hash_present[new_hash & HASH_PRESENT_MASK] = true;

                // Add new element to the candidate list
                current_beam_front[new_elem_count] = {new_hash, new_state, (uint8_t)prev_elem_idx,
                                                      false};
                current_scores[new_elem_count] = new_score;
                max_score = std::max(max_score, new_score);
                ++new_elem_count;
            }
        }

        for (size_t prev_elem_idx = 0; prev_elem_idx < current_beam_width; ++prev_elem_idx) {
            const auto& previous_element = prev_beam_front[prev_elem_idx];
            // Add the possible stay.
            const float stay_score = prev_scores[prev_elem_idx] + fixed_stay_score +
                                     static_cast<float>(block_back_scores[previous_element.state]);
            current_beam_front[new_elem_count] = {previous_element.hash, previous_element.state,
                                                  (uint8_t)prev_elem_idx, true};
            current_scores[new_elem_count] = stay_score;
            max_score = std::max(max_score, stay_score);

            // Determine whether the path including this stay duplicates another sequence ending in
            // a step.
            if (step_hash_present[previous_element.hash & HASH_PRESENT_MASK]) {
                size_t stay_elem_idx = (current_beam_width << NUM_BASE_BITS) + prev_elem_idx;
                // latest base is in smallest bits
                int stay_latest_base = int(previous_element.state & 3);

                // Go through all the possible step extensions that match this destination base with the stay and compare
                // their hashes, merging if we find any.
                for (size_t prev_elem_comp_idx = 0; prev_elem_comp_idx < current_beam_width;
                     prev_elem_comp_idx++) {
                    size_t step_elem_idx = (prev_elem_comp_idx << NUM_BASE_BITS) | stay_latest_base;
                    if (current_beam_front[stay_elem_idx].hash ==
                        current_beam_front[step_elem_idx].hash) {
                        if (current_scores[stay_elem_idx] > current_scores[step_elem_idx]) {
                            // Fold the step into the stay
                            const float folded_score = log_sum_exp(current_scores[stay_elem_idx],
                                                                   current_scores[step_elem_idx]);
                            current_scores[stay_elem_idx] = folded_score;
                            max_score = std::max(max_score, folded_score);
                            // The step element will end up last, sorted by score
                            current_scores[step_elem_idx] = std::numeric_limits<float>::lowest();
                        } else {
                            // Fold the stay into the step
                            const float folded_score = log_sum_exp(current_scores[stay_elem_idx],
                                                                   current_scores[step_elem_idx]);
                            current_scores[step_elem_idx] = folded_score;
                            max_score = std::max(max_score, folded_score);
                            // The stay element will end up last, sorted by score
                            current_scores[stay_elem_idx] = std::numeric_limits<float>::lowest();
                        }
                    }
                }
            }

            ++new_elem_count;
        }

        // Starting point for finding the cutoff score is the beam cut score
        float beam_cutoff_score = max_score - log_beam_cut;

        auto get_elem_count = [new_elem_count, &beam_cutoff_score, &current_scores]() {
            // Count the elements which meet the beam cutoff.
            size_t elem_count = 0;
            const float* score_ptr = current_scores;
            for (int i = int(new_elem_count); i; --i) {
                if (*score_ptr >= beam_cutoff_score) {
                    ++elem_count;
                }
                ++score_ptr;
            }
            return elem_count;
        };

        // Count the elements which meet the min score
        size_t elem_count = get_elem_count();

        if (elem_count > max_beam_width) {
            // Need to find a score which doesn't return too many scores, but doesn't reduce beam width too much
            size_t min_beam_width =
                    (max_beam_width * 8) / 10;  // 80% of beam width is the minimum we accept.
            float low_score = beam_cutoff_score;
            float hi_score = max_score;
            int num_guesses = 1;
            constexpr int MAX_GUESSES = 10;
            while ((elem_count > max_beam_width || elem_count < min_beam_width) &&
                   num_guesses < MAX_GUESSES) {
                if (elem_count > max_beam_width) {
                    // Make a higher guess
                    low_score = beam_cutoff_score;
                    beam_cutoff_score = (beam_cutoff_score + hi_score) / 2.0f;  // binary search.
                } else {
                    // Make a lower guess
                    hi_score = beam_cutoff_score;
                    beam_cutoff_score = (beam_cutoff_score + low_score) / 2.0f;  // binary search.
                }
                elem_count = get_elem_count();
                ++num_guesses;
            }
            // If we made 10 guesses and didn't find a suitable score, a couple of things may have happened:
            // 1: we just haven't completed the binary search yet (there is a good score in there somewhere but we didn't find it.)
            //  - in this case we should just pick the higher of the two current search limits to get the top N elements)
            // 2: there is no good score, as max_score returns more than beam_width elements (i.e. more than the whole beam width has max_score)
            //  - in this case we should just take max_beam_width of the top-scoring elements
            // 3: there is no good score as all the elements from <80% of the beam to >100% have the same score.
            //  - in this case we should just take the hi_score and accept it will return us less than 80% of the beam
            if (num_guesses == MAX_GUESSES) {
                beam_cutoff_score = hi_score;
                elem_count = get_elem_count();
            }

            // Clamp the element count to the max beam width in case of failure 2 from above.
            elem_count = std::min(elem_count, max_beam_width);
        }

        size_t write_idx = 0;
        for (size_t read_idx = 0; read_idx < new_elem_count; ++read_idx) {
            if (current_scores[read_idx] >= beam_cutoff_score) {
                if (write_idx < max_beam_width) {
                    prev_beam_front[write_idx] = current_beam_front[read_idx];
                    prev_scores[write_idx] = current_scores[read_idx];
                    ++write_idx;
                } else {
                    break;
                }
            }
        }

        // At the last timestep, we need to ensure the best path corresponds to element 0.
        // The other elements don't matter.
        if (block_idx == num_blocks - 1) {
            float best_score = std::numeric_limits<float>::lowest();
            size_t best_score_index = 0;
            for (size_t i = 0; i < elem_count; i++) {
                if (prev_scores[i] > best_score) {
                    best_score = prev_scores[i];
                    best_score_index = i;
                }
            }
            std::swap(prev_beam_front[0], prev_beam_front[best_score_index]);
            std::swap(prev_scores[0], prev_scores[best_score_index]);
        }

        size_t beam_offset = (block_idx + 1) * max_beam_width;
        for (size_t i = 0; i < elem_count; ++i) {
            // Remove backwards contribution from score
            prev_scores[i] -= float(block_back_scores[prev_beam_front[i].state]);

            // Copy this new beam front into the beam persistent state
            beam_vector[beam_offset + i].state = prev_beam_front[i].state;
            beam_vector[beam_offset + i].prev_element_index = prev_beam_front[i].prev_element_index;
            beam_vector[beam_offset + i].stay = prev_beam_front[i].stay;
        }

        current_beam_width = elem_count;
    }

    // Extract final score
    const float final_score = prev_scores[0];

    // Note that we don't emit the seed state at the front of the beam, hence the -1 offset when copying the path
    uint8_t element_index = 0;
    for (size_t beam_idx = num_blocks; beam_idx != 0; --beam_idx) {
        size_t beam_addr = beam_idx * max_beam_width + element_index;
        states[beam_idx - 1] = int32_t(beam_vector[beam_addr].state);
        moves[beam_idx - 1] = beam_vector[beam_addr].stay ? 0 : 1;
        element_index = beam_vector[beam_addr].prev_element_index;
    }
    moves[0] = 1;  // Always step in the first event

    // new compute
    // int shifted_states[2 * NUM_BASES];

    // Compute per-base qual data
    // for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
    //     int state = states[block_idx];
    //     states[block_idx] = states[block_idx] % NUM_BASES;
    //     int base_to_emit = states[block_idx];

    //     // Compute a probability for this block, based on the path kmer. See the following explanation:
    //     // https://git.oxfordnanolabs.local/machine-learning/notebooks/-/blob/master/bonito-basecaller-qscores.ipynb
    //     const U* const timestep_posts = posts + ((block_idx + 1) << num_state_bits);
    //     const auto fetch_post = [timestep_posts, posts_scale](size_t idx) {
    //         return static_cast<float>(timestep_posts[idx]) * posts_scale;
    //     };

    //     float block_prob = fetch_post(state);

    //     // Get indices of left- and right-shifted kmers
    //     int l_shift_idx = state >> NUM_BASE_BITS;
    //     int r_shift_idx = (state << NUM_BASE_BITS) % num_states;
    //     int msb = int(num_states) >> NUM_BASE_BITS;
    //     int l_shift_state, r_shift_state;
    //     for (int shift_base = 0; shift_base < NUM_BASES; ++shift_base) {
    //         l_shift_state = l_shift_idx + msb * shift_base;
    //         shifted_states[2 * shift_base] = l_shift_state;

    //         r_shift_state = r_shift_idx + shift_base;
    //         shifted_states[2 * shift_base + 1] = r_shift_state;
    //     }

    //     // Add probabilities for unique states
    //     int candidate_state;
    //     for (size_t state_idx = 0; state_idx < 2 * NUM_BASES; ++state_idx) {
    //         candidate_state = shifted_states[state_idx];
    //         // don't double-count this shifted state if it matches the current state
    //         bool count_state = (candidate_state != state);
    //         // or any other shifted state that we've seen so far
    //         if (count_state) {
    //             for (size_t inner_state = 0; inner_state < state_idx; ++inner_state) {
    //                 if (shifted_states[inner_state] == candidate_state) {
    //                     count_state = false;
    //                     break;
    //                 }
    //             }
    //         }
    //         if (count_state) {
    //             block_prob += fetch_post(candidate_state);
    //         }
    //     }
        
    //     if (block_prob < 0.0f) block_prob = 0.0f;
    //     else if (block_prob > 1.0f) block_prob = 1.0f;

    //     block_prob = std::pow(block_prob, 0.4f);  // Power fudge factor

    //     // Calculate a placeholder qscore for the "wrong" bases
    //     const float wrong_base_prob = (1.0f - block_prob) / 3.0f;

    //     for (size_t base = 0; base < NUM_BASES; base++) {
    //         qual_data[block_idx * NUM_BASES + base] =
    //                 (int(base) == base_to_emit ? block_prob : wrong_base_prob);
    //     }
    // }

    // old compute
    int hp_states[4] = {0, 0, 0,
                        0};  // What state index are the four homopolymers (A is always state 0)
    hp_states[3] = int(num_states) - 1;  // homopolymer T is always the last state. (11b per base)
    hp_states[1] = hp_states[3] / 3;     // calculate hp C from hp T (01b per base)
    hp_states[2] = hp_states[1] * 2;     // calculate hp G from hp C (10b per base)

    // Compute per-base qual data
    for (size_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        int state = states[block_idx];
        states[block_idx] = states[block_idx] % NUM_BASES;
        int base_to_emit = states[block_idx];

        // Compute a probability for this block, based on the path kmer. See the following explanation:
        // https://git.oxfordnanolabs.local/machine-learning/notebooks/-/blob/master/bonito-basecaller-qscores.ipynb
        const float* timestep_posts = posts + ((block_idx + 1) * num_states);

        // For states which are homopolymers, we don't want to count the states more than once
        bool is_hp = state == hp_states[0] || state == hp_states[1] || state == hp_states[2] ||
                     state == hp_states[3];
        float block_prob = float(timestep_posts[state]) * (is_hp ? -1.0f : 1.0f);

        // Add in left-shifted kmers
        int l_shift_idx = state / NUM_BASES;
        int msb = int(num_states) / NUM_BASES;
        for (int shift_base = 0; shift_base < NUM_BASES; shift_base++) {
            block_prob += float(timestep_posts[l_shift_idx + msb * shift_base]);
        }

        // Add in the right-shifted kmers
        int r_shift_idx = (state * NUM_BASES) % num_states;
        for (int shift_base = 0; shift_base < NUM_BASES; shift_base++) {
            block_prob += float(timestep_posts[r_shift_idx + shift_base]);
        }
        if (block_prob < 0.0f) block_prob = 0.0f;
        else if (block_prob > 1.0f) block_prob = 1.0f;
        block_prob = powf(block_prob, 0.4f);  // Power fudge factor

        // Calculate a placeholder qscore for the "wrong" bases
        float wrong_base_prob = (1.0f - block_prob) / 3.0f;

        for (size_t base = 0; base < NUM_BASES; base++) {
            qual_data[block_idx * NUM_BASES + base] =
                    (int(base) == base_to_emit ? block_prob : wrong_base_prob);
        }
    }

    return final_score;
}

std::tuple<std::string, std::string, std::vector<uint8_t>> beam_search_decode(
        const torch::Tensor& scores_t,
        const torch::Tensor& back_guides_t,
        const torch::Tensor& posts_t,
        const size_t max_beam_width,
        float beam_cut,
        float fixed_stay_score,
        float q_shift,
        float q_scale,
        float temperature,
        float byte_score_scale) {
    const int num_blocks = int(scores_t.size(0));
    const int num_states = get_num_states(scores_t.size(1));
    const int num_state_bits = static_cast<int>(std::log2(num_states));
    if (1 << num_state_bits != num_states) {
        throw std::runtime_error("num_states must be an integral power of 2");
    }
    BeamElement* beam_vector = (BeamElement*)malloc(max_beam_width * (num_blocks + 1) * sizeof(BeamElement));
    MALLOC_CHK(beam_vector);

    float* sorted_back_guides = (float*)malloc(num_states * sizeof(float));
    MALLOC_CHK(sorted_back_guides);

    int32_t* states = (int32_t*)malloc(num_blocks * sizeof(int32_t));
    MALLOC_CHK(states);
    
    uint8_t* moves = (uint8_t*)malloc(num_blocks * sizeof(uint8_t));
    MALLOC_CHK(moves);

    float* qual_data = (float*)malloc(num_blocks * NUM_BASES * sizeof(float));
    MALLOC_CHK(qual_data);

    // Posterior probabilities and back guides must be floats regardless of scores type.
    if (posts_t.dtype() != torch::kFloat32 || back_guides_t.dtype() != torch::kFloat32) {
        throw std::runtime_error(
                "beam_search_decode: mismatched tensor types provided for posts and "
                "guides");
    }

    // back guides and posts should be contiguous
    auto back_guides_contig = back_guides_t.expect_contiguous();
    auto posts_contig = posts_t.expect_contiguous();
    // scores_t may come from a tensor with chunks interleaved, but make sure the last dimension is contiguous
    auto scores_block_contig = (scores_t.stride(1) == 1) ? scores_t : scores_t.contiguous();
    const size_t scores_block_stride = scores_block_contig.stride(0);
    if (scores_t.dtype() == torch::kFloat32) {
        const auto scores = scores_block_contig.data_ptr<float>();
        const auto back_guides = back_guides_contig->data_ptr<float>();
        const auto posts = posts_contig->data_ptr<float>();

        beam_search<float, float>(scores, scores_block_stride, back_guides, posts, num_state_bits, num_blocks,
                           max_beam_width, beam_cut, fixed_stay_score, states, moves, qual_data, 1.0f, 1.0f, beam_vector, sorted_back_guides);
    } else if (scores_t.dtype() == torch::kInt8) {
        // const auto scores = scores_block_contig.data_ptr<int8_t>();
        // const auto back_guides = back_guides_contig->data_ptr<float>();
        // const auto posts = posts_contig->data_ptr<float>();
        // const float posts_scale = static_cast<float>(1.0 / 32767.0);
        // beam_search<int8_t, int16_t>(scores, scores_block_stride, back_guides, posts, num_states, num_blocks,
        //                     max_beam_width, beam_cut, fixed_stay_score, states, moves, qual_data, byte_score_scale,
        //                              posts_scale);
    } else {
        throw std::runtime_error(std::string("beam_search_decode: unsupported tensor type ") +
                                 std::string(scores_t.dtype().name()));
    }
    
    size_t seq_len = 0;
    for (int i = 0; i < num_blocks; ++i) {
        seq_len += moves[i];
    }

    float* base_probs = (float*)malloc(num_blocks * sizeof(float));
    MALLOC_CHK(base_probs);

    float* total_probs = (float*)malloc(num_blocks * sizeof(float));
    MALLOC_CHK(total_probs);

    char* sequence = (char*)malloc(num_blocks * sizeof(char));
    MALLOC_CHK(sequence);

    char* qstring = (char*)malloc(num_blocks * sizeof(char));
    MALLOC_CHK(qstring);

    generate_sequence(moves, states, qual_data, q_shift, q_scale, num_blocks, seq_len, base_probs, total_probs, sequence, qstring);
    
    sequence[seq_len] = '\0';
    qstring[seq_len] = '\0';
    std::string sequence_str(sequence);
    std::string qstring_str(qstring);
    std::vector<uint8_t> moves_vec(num_blocks);
    std::copy(moves, moves + num_blocks, moves_vec.begin());

    free(beam_vector);
    free(sorted_back_guides);
    free(qual_data);
    free(states);
    free(moves);
    free(base_probs);
    free(total_probs);
    free(sequence);
    free(qstring);

    return std::make_tuple(sequence_str, qstring_str, moves_vec);
}
