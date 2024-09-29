#include "beam_search.h"

#include "fast_hash.h"
#include "error.h"
#include "misc.h"
#include <math.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>

constexpr int NUM_BASE_BITS = 2;
constexpr int NUM_BASES = 1 << NUM_BASE_BITS;

constexpr uint64_t HASH_PRESENT_BITS = 4096;
constexpr uint64_t HASH_PRESENT_MASK = HASH_PRESENT_BITS - 1;

// our kmer
typedef uint16_t state_t;

// represents a potential state (k-mer) for a single beam
struct BeamElement {
    state_t state;                  // the k-mer it represents
    uint8_t prev_element_index;     // points to the element it transitioned from, an element may branch into multiple k-mers
    bool stay;                      // false: it's a stay, true: it's a step
};

// this is the data we need to retain for only the previous timestep (block) in the beam
// (and what we construct for the new timestep)
struct BeamFrontElement {
    uint64_t hash;
    state_t state;
    uint8_t prev_element_index;
    bool stay;
};

float log_sum_exp(float x, float y, float t) {
    float abs_diff = fabsf(x - y) / t;
    return fmaxf(x, y) + ((abs_diff < 17.0f) ? (log1pf(expf(-abs_diff)) * t) : 0.0f);
}

inline int get_num_states(size_t num_trans_states) {
    if (num_trans_states % NUM_BASES != 0) {
        throw std::runtime_error("Unexpected number of transition states in beam search decode.");
    }
    return int(num_trans_states / NUM_BASES);
}

static inline std::tuple<std::string, std::string> generate_sequence(
    const std::vector<uint8_t>& moves,
    const std::vector<int32_t>& states,
    const std::vector<float>& qual_data,
    float shift,
    float scale
) {
    size_t seqPos = 0;
    size_t num_blocks = moves.size();
    size_t seqLen = accumulate(moves.begin(), moves.end(), 0);

    std::string sequence(seqLen, 'N');
    std::string qstring(seqLen, '!');
    std::array<char, 4> alphabet = {'A', 'C', 'G', 'T'};
    std::vector<float> baseProbs(seqLen), totalProbs(seqLen);

    for (size_t blk = 0; blk < num_blocks; ++blk) {
        int state = states[blk];
        int move = int(moves[blk]);
        int base = state & 3;
        int offset = (blk == 0) ? 0 : move - 1;
        int probPos = int(seqPos) + offset;

        // get the probability for the called base.
        baseProbs[probPos] += qual_data[blk * alphabet.size() + base];

        // accumulate the total probability for all possible bases at this position, for normalization.
        for (size_t k = 0; k < alphabet.size(); ++k) {
            totalProbs[probPos] += qual_data[blk * alphabet.size() + k];
        }

        if (blk == 0) {
            sequence[seqPos++] = char(base);
        } else {
            for (int j = 0; j < move; ++j) {
                sequence[seqPos++] = char(base);
            }
        }
    }

    for (size_t i = 0; i < seqLen; ++i) {
        sequence[i] = alphabet[int(sequence[i])];
        baseProbs[i] = 1.0f - (baseProbs[i] / totalProbs[i]);
        baseProbs[i] = -10.0f * log10f(baseProbs[i]);
        float qscore = baseProbs[i] * scale + shift;
        qscore = std::min(50.0f, qscore);
        qscore = std::max(1.0f, qscore);
        qstring[i] = char(33.5f + qscore);
    }

    return make_tuple(sequence, qstring);
}

template <typename T>
static inline void add_candidate_steps(
    size_t block_idx,
    state_t states_mask,
    std::bitset<HASH_PRESENT_BITS>& step_hash_present,
    const T* const scores,
    size_t scores_block_stride,
    const float* const back_guide,
    int num_state_bits,
    std::vector<BeamFrontElement>& prev_beam_front,
    std::vector<BeamFrontElement>& current_beam_front,
    std::vector<float>& current_scores,
    std::vector<float>& prev_scores,
    const size_t current_beam_width,
    float score_scale,
    float& max_score,
    size_t& new_elem_count,
    const T* const block_scores,
    const float* const block_back_scores
) {
    // retrieves the given score as a float, multiplied by score_scale
    // score of the transition
    const auto fetch_block_score = [block_scores, score_scale](size_t idx) {
        return static_cast<float>(block_scores[idx]) * score_scale;
    };

    for (size_t prev_elem_idx = 0; prev_elem_idx < current_beam_width; ++prev_elem_idx) {
            const auto& previous_element = prev_beam_front[prev_elem_idx];

            // expand all the possible steps
            for (size_t new_base = 0; new_base < NUM_BASES; new_base++) {
                
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
                
                // shift the state (k-mer) left and append the new base to the end of bitset
                // transition to a new k-mer (see explanation above)
                state_t new_state = (state_t((previous_element.state << NUM_BASE_BITS) & states_mask) | new_base);
                
                // get the score of this transition (see explanation above)
                const auto move_idx = static_cast<state_t>(
                        (new_state << NUM_BASE_BITS) +
                        (((previous_element.state << NUM_BASE_BITS) >> num_state_bits)));
                
                float new_score = prev_scores[prev_elem_idx]                                // score of prev elem
                                    + fetch_block_score(move_idx)                           // + score of transitioning from prev state to current state
                                    + static_cast<float>(block_back_scores[new_state]);     // + score of this state being in this timestep
                                  
                // generate hash from previous element and new state
                uint64_t new_hash = chainfasthash64(previous_element.hash, new_state);
                step_hash_present[new_hash & HASH_PRESENT_MASK] = true;

                // add new element to candidates
                current_beam_front[new_elem_count] = {
                    new_hash,
                    new_state,
                    (uint8_t)prev_elem_idx,
                    false // this is never a stay, these are possible steps
                };
                
                // update scores
                current_scores[new_elem_count] = new_score;
                max_score = std::max(max_score, new_score);
                ++new_elem_count;
            }
        }
}

static inline void add_candidate_stays(
    size_t block_idx,
    std::bitset<HASH_PRESENT_BITS>& step_hash_present,
    std::vector<BeamFrontElement>& prev_beam_front,
    std::vector<BeamFrontElement>& current_beam_front,
    std::vector<float>& current_scores,
    std::vector<float>& prev_scores,
    const size_t current_beam_width,
    const float fixed_stay_score,
    const float temperature,
    float& max_score,
    size_t& new_elem_count,
    const float* const block_back_scores
) {
    for (size_t prev_elem_idx = 0; prev_elem_idx < current_beam_width; ++prev_elem_idx) {
        const auto& previous_element = prev_beam_front[prev_elem_idx];
        
        // score for possible stay
        // if it's a stay, it's a kmer that repeats in the sequence
        const float stay_score = prev_scores[prev_elem_idx]                                         // score of prev elem
                                + fixed_stay_score                                                  // + some static score for possible stays
                                + static_cast<float>(block_back_scores[previous_element.state]);    // + score of previous state being in this timestep
                                    
        // add stay to candidates
        // since we will always have the step as a candidate (from our previous step) in our current_beam_front and current_scores, we only need to create candidate stay beam elements
        current_beam_front[new_elem_count] = {
            previous_element.hash,
            previous_element.state,
            (uint8_t)prev_elem_idx,
            true // this is always stay, a stay represents a beam_element that has not changed since the last timestep
        };                                
        current_scores[new_elem_count] = stay_score;
        
        // update max score
        max_score = std::max(max_score, stay_score);

        // determine whether the path including this stay duplicates another sequence ending in a step equal to the state of the stay
        if (step_hash_present[previous_element.hash & HASH_PRESENT_MASK]) {
        
            // left shift by 2 and then add the previous elem idx
            size_t stay_elem_idx = (current_beam_width << NUM_BASE_BITS) + prev_elem_idx;
            
            // latest base is in smallest bits
            int stay_latest_base = int(previous_element.state & 3);

            // go through all the possible step extensions that match this destination base with the stay and compare their hashes, merging if we find any
            for (size_t prev_elem_comp_idx = 0; prev_elem_comp_idx < current_beam_width; prev_elem_comp_idx++) {
            
                // it's a step if it's a previous kmer with a repeated base
                size_t step_elem_idx = (prev_elem_comp_idx << NUM_BASE_BITS) | stay_latest_base;
                
                // compare hashes of step extension and possible stay, if they're equal, we merge
                if (current_beam_front[stay_elem_idx].hash == current_beam_front[step_elem_idx].hash) {
                    
                    // new score for the stay
                    const float folded_score = log_sum_exp(current_scores[stay_elem_idx], current_scores[step_elem_idx], temperature);
                    
                    // merge: 
                    //      update the better scoring beam element
                    //      sort the worse scoring one last
                    if (current_scores[stay_elem_idx] > current_scores[step_elem_idx]) {
                        current_scores[stay_elem_idx] = folded_score;
                        current_scores[step_elem_idx] = std::numeric_limits<float>::lowest();
                    } else {
                        current_scores[step_elem_idx] = folded_score;
                        current_scores[stay_elem_idx] = std::numeric_limits<float>::lowest();
                    }
                    
                    // update max score
                    max_score = std::max(max_score, folded_score);
                }
            }
        }

        // update new elem count
        ++new_elem_count;
    }
}

template <typename T>
static inline void init_beams(
    const float* const back_guide,
    std::vector<BeamElement>& beam_vector,
    std::vector<BeamFrontElement>& prev_beam_front,
    std::vector<float>& prev_scores,
    const size_t current_beam_width,
    const size_t max_beam_width,
    const size_t num_states
) {
    // find the score a state needs to make it into the first set of beam elements
    T beam_init_threshold = std::numeric_limits<T>::lowest();
    if (max_beam_width < num_states) {
        // copy the first set of back guides and sort to extract max_beam_width highest elements
        std::vector<T> sorted_back_guides(num_states);
        std::memcpy(sorted_back_guides.data(), back_guide, num_states * sizeof(T));

        // note that we don't need a full sort here to get the max_beam_width highest values
        std::nth_element(sorted_back_guides.begin(), sorted_back_guides.begin() + max_beam_width - 1, sorted_back_guides.end(), std::greater<T>());
        beam_init_threshold = sorted_back_guides[max_beam_width - 1];
    }

    // initialise all beams
    // go through all state scores in the first block of the back_guide
    constexpr uint64_t HASH_SEED = 0x880355f21e6d1965ULL;
    for (size_t state = 0, beam_element = 0; state < num_states && beam_element < max_beam_width; state++) {
    
        if (back_guide[state] >= beam_init_threshold) {
        
            // note that this first element has a prev_element_index of 0
            prev_beam_front[beam_element] = {chainfasthash64(HASH_SEED, state), static_cast<state_t>(state), 0, false};
            
            prev_scores[beam_element] = 0.0f;
            ++beam_element;
        }
    }

    // copy all beam fronts into the beam persistent state
    for (size_t element_idx = 0; element_idx < current_beam_width; ++element_idx) {
        beam_vector[element_idx].state                  = prev_beam_front[element_idx].state;
        beam_vector[element_idx].prev_element_index     = prev_beam_front[element_idx].prev_element_index;
        beam_vector[element_idx].stay                   = prev_beam_front[element_idx].stay;
    }
}

static inline float find_cutoff(
    const std::vector<float>& current_scores,
    const float prev_beam_cutoff,
    const size_t new_elem_count,
    const size_t max_beam_width,
    size_t& elem_count,
    const float max_score
) {
    float beam_cutoff_score = prev_beam_cutoff;
    // count the elements which meet the beam cutoff
    auto get_elem_count = [new_elem_count, &beam_cutoff_score, &current_scores]() {
        size_t elem_count = 0;
        const float* score_ptr = current_scores.data();
        for (int i = new_elem_count; i; --i) {
            if (*score_ptr >= beam_cutoff_score) {
                ++elem_count;
            }
            ++score_ptr;
        }
        return elem_count;
    };

    // count the elements which meet the min score
    elem_count = get_elem_count();

    if (elem_count > max_beam_width) {
        // binary search to find a score which doesn't return too many scores, but doesn't reduce beam width too much
        size_t min_beam_width = (max_beam_width * 8) / 10;  // 80% of beam width is the minimum we accept.
        float low_score = beam_cutoff_score;
        float hi_score = max_score;
        int num_guesses = 1;
        constexpr int MAX_GUESSES = 10;
        
        while ((elem_count > max_beam_width || elem_count < min_beam_width) &&
                num_guesses < MAX_GUESSES) {
            if (elem_count > max_beam_width) {
                // Make a higher guess
                low_score = beam_cutoff_score;
                beam_cutoff_score = (beam_cutoff_score + hi_score) / 2.0f;
            } else {
                // Make a lower guess
                hi_score = beam_cutoff_score;
                beam_cutoff_score = (beam_cutoff_score + low_score) / 2.0f;
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

        // clamp the element count to the max beam width in case of failure 2 from above.
        elem_count = std::min(elem_count, max_beam_width);
    }

    return beam_cutoff_score;
}

void compute_qual_base_data(
    std::vector<float>& qual_data,
    std::vector<int32_t>& states,
    const float* const posts,
    const size_t num_blocks,
    const size_t num_states,
    const size_t num_state_bits
) {
    int shifted_states[2 * NUM_BASES];
    
    for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int state = states[block_idx];
        states[block_idx] = states[block_idx] % NUM_BASES;
        int base_to_emit = states[block_idx];

        // compute a probability for this block, based on the path kmer. See the following explanation:
        // https://git.oxfordnanolabs.local/machine-learning/notebooks/-/blob/master/bonito-basecaller-qscores.ipynb << link is broken
        const float* timestep_posts = posts + ((block_idx + 1) << num_state_bits);

        float block_prob = float(timestep_posts[state]);

        // get indices of left-and right-shifted kmers
        int l_shift_idx = state >> NUM_BASE_BITS;
        int r_shift_idx = (state << NUM_BASE_BITS) % num_states;
        int msb = int(num_states) >> NUM_BASE_BITS;
        int l_shift_state, r_shift_state;
        
        // candidates are shifted kmers with one of the bases appended to their respecitve shifted side
        for (int shift_base = 0; shift_base < NUM_BASES; ++shift_base) {
            l_shift_state = l_shift_idx + msb * shift_base;
            shifted_states[2 * shift_base] = l_shift_state;

            r_shift_state = r_shift_idx + shift_base;
            shifted_states[2 * shift_base + 1] = r_shift_state;
        }

        // add probabilities for unique states
        int candidate_state;
        for (size_t state_idx = 0; state_idx < 2 * NUM_BASES; ++state_idx) {
            candidate_state = shifted_states[state_idx];
            
            // don't double-count this shifted state if it matches the current state
            // or any other shifted state that we've seen so far
            bool count_state = (candidate_state != state);
            
            // check all states we've seen
            if (count_state) {
                for (size_t inner_state = 0; inner_state < state_idx; ++inner_state) {
                    if (shifted_states[inner_state] == candidate_state) {
                        count_state = false;
                        break;
                    }
                }
            }
            
            // add to block prob
            if (count_state) {
                block_prob += float(timestep_posts[candidate_state]);
            }
        }
        
        // clamp block_prob
        if (block_prob < 0.0f) block_prob = 0.0f;
        else if (block_prob > 1.0f) block_prob = 1.0f;
        
        block_prob = std::pow(block_prob, 0.4f); // power fudge factor

        // calculate a placeholder qscore for the "wrong" bases
        float wrong_base_prob = (1.0f - block_prob) / 3.0f;
        
        // record qual data
        for (size_t base = 0; base < NUM_BASES; base++) {
            qual_data[block_idx * NUM_BASES + base] =
                    (int(base) == base_to_emit ? block_prob : wrong_base_prob);
        }
    }
}

template <typename T>
float beam_search(
    const T* const scores,
    size_t scores_block_stride,
    const float* const back_guide,
    const float* const posts,
    int num_state_bits,
    size_t num_blocks,
    size_t max_beam_width,
    float beam_cut,
    float fixed_stay_score,
    std::vector<int32_t>& states,
    std::vector<uint8_t>& moves,
    std::vector<float>& qual_data,
    float temperature,
    float score_scale
) {
    if (max_beam_width > 256) {
        throw std::range_error("Beamsearch max_beam_width cannot be greater than 256.");
    }

    // create the beam, we need to keep beam_width elements for each block, plus the initial state
    std::vector<BeamElement> beam_vector(max_beam_width * (num_blocks + 1));

    const size_t num_states = 1 << num_state_bits;
    const state_t states_mask = static_cast<state_t>(num_states - 1);
    
    const float log_beam_cut = (beam_cut > 0.0f) ? (temperature * logf(beam_cut)) : std::numeric_limits<float>::max();
    size_t max_beam_candidates = (NUM_BASES + 1) * max_beam_width;

    std::vector<BeamFrontElement> current_beam_front(max_beam_candidates);
    std::vector<BeamFrontElement> prev_beam_front(max_beam_candidates);

    std::vector<float> current_scores(max_beam_candidates);
    std::vector<float> prev_scores(max_beam_candidates);

    size_t current_beam_width = std::min(max_beam_width, num_states);
    init_beams<T>(back_guide, beam_vector, prev_beam_front, prev_scores, current_beam_width, max_beam_width, num_states);

    // extend beams
    for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        // a k=1 Bloom filter, indicating the presence of steps with particular sequence hashes
        // avoids comparing stay hashes against all possible progenitor states where none of them has the requisite sequence hash
        std::bitset<HASH_PRESENT_BITS> step_hash_present;  // default constructor zeros content

        // generate list of candidate elements for this timestep (block)
        // as we do so, update the maximum score
        size_t new_elem_count = 0;
        float max_score = std::numeric_limits<float>::lowest();

        const T* const block_scores = scores + (block_idx * scores_block_stride);
        const float* const block_back_scores = back_guide + ((block_idx + 1) << num_state_bits);

        add_candidate_steps<T>(block_idx, states_mask, step_hash_present, scores, scores_block_stride, back_guide, num_state_bits,
                            prev_beam_front, current_beam_front, current_scores, prev_scores, current_beam_width, score_scale,
                            max_score, new_elem_count, block_scores, block_back_scores);

        add_candidate_stays(block_idx, step_hash_present, prev_beam_front, current_beam_front, current_scores,
                            prev_scores, current_beam_width, fixed_stay_score, temperature, max_score, new_elem_count, block_back_scores);

        // find a suitable cutoff score
        float initial_cutoff = max_score - log_beam_cut;
        size_t elem_count = 0;
        float beam_cutoff_score = find_cutoff(current_scores, initial_cutoff, new_elem_count, max_beam_width, elem_count, max_score);

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

        // at the last timestep, we need to ensure the best path corresponds to element 0
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
        
        // copy this new beam front into the beam persistent state
        size_t beam_offset = (block_idx + 1) * max_beam_width;
        for (size_t i = 0; i < elem_count; ++i) {
            // remove backwards contribution from score
            prev_scores[i] -= float(block_back_scores[prev_beam_front[i].state]);

            beam_vector[beam_offset + i].state = prev_beam_front[i].state;
            beam_vector[beam_offset + i].prev_element_index = prev_beam_front[i].prev_element_index;
            beam_vector[beam_offset + i].stay = prev_beam_front[i].stay;
        }

        // adjust current beam width
        current_beam_width = elem_count;
    }

    // extract final score
    const float final_score = prev_scores[0];

    // write out sequence bases and move table
    moves.resize(num_blocks);
    states.resize(num_blocks);

    // note that we don't emit the seed state at the front of the beam, hence the -1 offset when copying the path
    uint8_t element_index = 0;
    for (size_t beam_idx = num_blocks; beam_idx != 0; --beam_idx) {
        size_t beam_addr = beam_idx * max_beam_width + element_index;
        states[beam_idx - 1] = int32_t(beam_vector[beam_addr].state);
        moves[beam_idx - 1] = beam_vector[beam_addr].stay ? 0 : 1;
        element_index = beam_vector[beam_addr].prev_element_index;
    }
    moves[0] = 1;  // always step in the first event

    // compute qual data
    compute_qual_base_data(qual_data, states, posts, num_blocks, num_states, num_state_bits);

    return final_score;
}

std::tuple<std::string, std::string, std::vector<uint8_t>> beam_search_decode(
    const torch::Tensor& scores_t,
    const torch::Tensor& back_guides_t,
    const torch::Tensor& posts_t,
    size_t max_beam_width,
    float beam_cut,
    float fixed_stay_score,
    float q_shift,
    float q_scale,
    float temperature,
    float byte_score_scale
) {
    const int num_blocks = int(scores_t.size(0));
    const int num_states = get_num_states(scores_t.size(1));
    const int num_state_bits = static_cast<int>(std::log2(num_states));
    if (1 << num_state_bits != num_states) {
        throw std::runtime_error("num_states must be an integral power of 2");
    }

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

    std::vector<int32_t> states(num_blocks);
    std::vector<uint8_t> moves(num_blocks);
    std::vector<float> qual_data(num_blocks * NUM_BASES);

    const size_t scores_block_stride = scores_block_contig.stride(0);
    if (scores_t.dtype() == torch::kFloat32) {
        const auto scores = scores_block_contig.data_ptr<float>();
        const auto back_guides = back_guides_contig->data_ptr<float>();
        const auto posts = posts_contig->data_ptr<float>();

        beam_search<float>(scores, scores_block_stride, back_guides, posts, num_state_bits,
                           num_blocks, max_beam_width, beam_cut, fixed_stay_score, states, moves,
                           qual_data, temperature, 1.0f);
    } else if (scores_t.dtype() == torch::kInt8) {
        const auto scores = scores_block_contig.data_ptr<int8_t>();
        const auto back_guides = back_guides_contig->data_ptr<float>();
        const auto posts = posts_contig->data_ptr<float>();
        beam_search<int8_t>(scores, scores_block_stride, back_guides, posts, num_state_bits,
                            num_blocks, max_beam_width, beam_cut, fixed_stay_score, states, moves,
                            qual_data, temperature, byte_score_scale);
    } else {
        throw std::runtime_error(std::string("beam_search_decode: unsupported tensor type ") +
                                 std::string(scores_t.dtype().name()));
    }

    std::string sequence, qstring;
    std::tie(sequence, qstring) = generate_sequence(moves, states, qual_data, q_shift, q_scale);

    return std::make_tuple(sequence, qstring, moves);
}
