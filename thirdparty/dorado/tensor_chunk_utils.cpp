
#include <cstdint>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <string>
#include <utility>

#include "error.h"
#include "tensor_chunk_utils.h"

#define EPS (1e-9f)
#define DEFAULT_TRIM_THRESHOLD (2.4f)
#define DEFAULT_TRIM_WINDOW_SIZE (40)
#define DEFAULT_TRIM_MIN_ELEMENTS (3)

using Slice = torch::indexing::Slice;

int trim(const torch::Tensor& signal, float threshold, int window_size, int min_elements) {
    const int min_trim = 10;
    const int num_samples = static_cast<int>(signal.size(0)) - min_trim;
    const int num_windows = num_samples / window_size;

    // Access via raw pointers because of torch indexing overhead.
    const auto signal_f32 = signal.to(torch::ScalarType::Float);
    assert(signal_f32.is_contiguous());
    const float* const signal_f32_ptr = signal_f32.data_ptr<float>();

    bool seen_peak = false;
    for (int pos = 0; pos < num_windows; ++pos) {
        const int start = pos * window_size + min_trim;
        const int end = start + window_size;
        assert(start < signal.size(0));
        assert(end <= signal.size(0));  // end is exclusive

        const auto num_large_enough = std::count_if(&signal_f32_ptr[start], &signal_f32_ptr[end], [threshold](float elem) { return elem > threshold; });

        if (num_large_enough > min_elements || seen_peak) {
            seen_peak = true;
            if (signal_f32_ptr[end - 1] > threshold) {
                continue;
            }
            if (end >= num_samples) {
                return min_trim;
            } else {
                return end;
            }
        }
    }

    return min_trim;
}

std::pair<float, float> normalisation(QuantileScalingParams& params, torch::Tensor& x) {
    auto quantiles = quantile_counting(x, torch::tensor({params.quantile_a, params.quantile_b}));
    float q20 = quantiles[0].item<float>();
    float q90 = quantiles[1].item<float>();
    float shift = std::max(10.0f, params.shift_multiplier * (q20 + q90));
    float scale = std::max(1.0f, params.scale_multiplier * (q90 - q20));
    return std::make_pair(shift, scale);
}

std::pair<float, float> med_mad(torch::Tensor &x, float factor=1.4826){
    torch::Tensor med = x.median();
    torch::Tensor mad = torch::median(torch::abs(x - med)) * factor + EPS;

    return {med.item<float>(), mad.item<float>()};
}

int determine_rna_adapter_pos(torch::Tensor &signal) {
    assert(signal.dtype() == at::kShort);

    const int kWindowSize = 250;
    const int kStride = 50;
    const int16_t kMedianDiff = 125;
    const int16_t kMedianDiffForDiffOnlyCheck = 150;

    const int16_t kMinMedianForRNASignal = 700;

    int signal_len = static_cast<int>(signal.size(0));
    const int16_t* signal_data_ptr = static_cast<int16_t*>(signal.data_ptr());

    // Check the median value change over 5 windows.
    std::array<int16_t, 5> medians = {0, 0, 0, 0, 0};
    std::array<int32_t, 5> window_pos = {0, 0, 0, 0, 0};
    int median_pos = 0;
    int break_point = 0;
    const int signal_start = 1000;
    const int signal_end = 3 * signal_len / 4;
    for (int i = signal_start; i < signal_end; i += kStride) {
        auto slice = at::from_blob(const_cast<int16_t*>(&signal_data_ptr[i]),
                                   {static_cast<int>(std::min(kWindowSize, signal_len - i))},
                                   at::TensorOptions().dtype(at::kShort));
        int16_t median = slice.median().item<int16_t>();
        medians[median_pos % medians.size()] = median;
        // Since the medians are stored in a circular buffer, we need
        // to store the actual window positions for the median values
        // as well to check that maximum median value came from a window
        // after that of the minimum median value.
        window_pos[median_pos % window_pos.size()] = median_pos;
        auto minmax = std::minmax_element(medians.begin(), medians.end());
        // The range of raw signal values is within the range of [-500, 3000] (TODO: they're
        // likely are non-negative but need to confirm that). So the median values lie
        // in the same range, and any difference between the median values
        // will not result in an overflow with the int16_t data type.
        int16_t min_median = *minmax.first;
        int16_t max_median = *minmax.second;
        auto min_pos = std::distance(medians.begin(), minmax.first);
        auto max_pos = std::distance(medians.begin(), minmax.second);

        if ((median_pos >= static_cast<int>(medians.size()) &&
             window_pos[max_pos] > window_pos[min_pos]) &&
            (((max_median > kMinMedianForRNASignal) && (max_median - min_median > kMedianDiff)) ||
             (max_median - min_median > kMedianDiffForDiffOnlyCheck))) {
            break_point = i;
            break;
        }
        ++median_pos;
    }

    return break_point;
}

void scale_signal(core_t *core, torch::Tensor &signal, float scaling, float offset, SignalNormalisationParams &scaling_params) {
    auto strategy = scaling_params.strategy;
    float scale = 1.0f;
    float shift = 0.0f;
    int trim_start = 0;
    bool is_rna_model = is_rna(core->model_config->sample_type);

    if (is_rna_model) {
        const bool has_rna_based_adapters = false;
        if (!has_rna_based_adapters) {
            trim_start = determine_rna_adapter_pos(signal);
            if (int64_t(trim_start) < signal.size(0)) {
                signal = signal.index({Slice(trim_start, at::indexing::None)});
                // read->read_common.rna_adapter_end_signal_pos = 0;
            } else {
                // TODO: handle this case
                // If RNA adapter isn't trimmed, track where the adapter signal is ending
                // so it can be used during polyA estimation.
                // read->read_common.rna_adapter_end_signal_pos = trim_start;
                // Since we're not actualy trimming the signal, reset the trim value to 0.
                trim_start = 0;
            }
        }
    }

    if (strategy == ScalingStrategy::PA) {
        auto stdn = scaling_params.standarisation;
        if (stdn.standardise) {
            scale = stdn.stdev / scaling;
            shift = (stdn.mean / scaling) - offset;
        } else {
            scale = 1.f / scaling;
            shift = -1.f * offset;
        }
        signal = ((signal.to(torch::kFloat) - shift) / scale).to(torch::kFloat16);

    } else {
        auto t1 = strategy == ScalingStrategy::QUANTILE ? normalisation(scaling_params.quantile, signal) : med_mad(signal);
        shift = std::get<0>(t1);
        scale = std::get<1>(t1);

        signal = ((signal.to(torch::kFloat) - shift) / scale).to(torch::kFloat16);
    }

    if (!is_rna_model) {
        if (trim_start == 0 && scaling_params.standarisation.standardise) {
            trim_start = 10;
        } else if (trim_start == 0) {
            // 8000 value may be changed in future. Currently this is found to work well.
            int max_samples = std::min(8000, (int)(signal.size(0) / 2));
            trim_start = trim(
                signal.index({Slice(torch::indexing::None, max_samples)}),
                DEFAULT_TRIM_THRESHOLD,
                DEFAULT_TRIM_WINDOW_SIZE,
                DEFAULT_TRIM_MIN_ELEMENTS
            );
        }

        if ((size_t)(trim_start) < (size_t)(signal.size(0))) {
            signal = signal.index({Slice(trim_start, torch::indexing::None)});
        }
    }
}

int div_round_closest(const int n, const int d) {
    return ((n < 0) ^ (d < 0)) ? ((n - d/2)/d) : ((n + d/2)/d);
}

void stitch_chunks(chunk_db_t *chunk_db, size_t i, std::string &sequence, std::string &qstring) {
    std::vector<chunk_res> &chunks = (*chunk_db->chunks_res)[i];
    // Calculate the chunk down sampling, round to closest int.
    int down_sampling = div_round_closest(chunks[0].raw_chunk_size, chunks[0].moves.size());

    int start_pos = 0;
    std::vector<std::string> sequences;
    std::vector<std::string> qstrings;
    for (size_t i = 0; i < chunks.size() - 1; i++){
        chunk_res_t &current_chunk = chunks[i];
        chunk_res_t &next_chunk = chunks[i+1];
        int overlap_size = (current_chunk.raw_chunk_size + current_chunk.input_offset) - (next_chunk.input_offset);
        int overlap_down_sampled = overlap_size / down_sampling;
        int mid_point = overlap_down_sampled / 2;

        int current_chunk_bases_to_trim = 0;
        for (int i = current_chunk.moves.size() - 1; i > (int)(current_chunk.moves.size() - mid_point); i--){
            current_chunk_bases_to_trim += (int) current_chunk.moves[i];
        }

        int current_chunk_seq_len = current_chunk.seq.size();
        int end_pos = current_chunk_seq_len - current_chunk_bases_to_trim;
        int trimmed_len = end_pos - start_pos;
        sequences.push_back(current_chunk.seq.substr(start_pos, trimmed_len));
        qstrings.push_back(current_chunk.qstring.substr(start_pos, trimmed_len));

        start_pos = 0;
        for (int i=0; i < mid_point; i++){
            start_pos += (int) next_chunk.moves[i];
        }
    }

    //append the final read
    sequences.push_back(chunks[chunks.size() - 1].seq.substr(start_pos));
    qstrings.push_back(chunks[chunks.size() - 1].qstring.substr(start_pos));

    // Set the read seq and qstring
    sequence = std::accumulate(sequences.begin(), sequences.end(), std::string(""));
    qstring = std::accumulate(qstrings.begin(), qstrings.end(), std::string(""));
}

std::vector<torch::Tensor> load_tensors(const std::string& dir, const std::vector<std::string>& tensors) {
    auto weights = std::vector<torch::Tensor>();
    for (auto tensor : tensors) {
        auto path = dir + "/" + tensor;
        torch::load(weights, path);
    }

    return weights;
}

torch::Tensor quantile(const torch::Tensor t, const torch::Tensor q) {
    assert(q.dtype() == torch::kF32);

    auto tmp = t.clone();

    // auto [qval, qidx] = q.sort();
    auto q_sorted = q.sort();
    auto qval = std::get<0>(q_sorted);
    auto qidx = std::get<1>(q_sorted);

    auto res = torch::empty_like(q);

    auto start = tmp.data_ptr<float>();
    auto end = tmp.data_ptr<float>() + tmp.size(0);

    for (int i = 0; i < q.size(0); i++) {
        auto m = tmp.data_ptr<float>() +
                 static_cast<size_t>((tmp.size(0) - 1) * qval[i].item<float>());
        std::nth_element(start, m, end);
        res[qidx[i]] = *m;
        start = m;
    }

    return res;
}

torch::Tensor quantile_counting(const torch::Tensor t, const torch::Tensor q) {
    assert(q.dtype() == torch::kF32);

    auto p = t.data_ptr<int16_t>();
    auto range_min = t.min().item<int16_t>();
    auto range_max = t.max().item<int16_t>();

    int size = t.size(0);

    std::vector<int> counts(range_max - range_min + 1, 0);
    for (int i = 0; i < size; ++i) {
        counts[p[i] - range_min]++;
    }

    std::partial_sum(counts.begin(), counts.end(), counts.begin());

    auto res = torch::empty_like(q);

    for (size_t idx = 0; idx < (size_t)q.numel(); idx++) {
        int threshold = q[idx].item<float>() * (size - 1);
        for (size_t i = 0; i <= counts.size(); ++i) {
            if (counts[i] > threshold) {
                res[idx] = (int16_t)i + range_min;
                break;
            }
        }
    }

    return res;
}