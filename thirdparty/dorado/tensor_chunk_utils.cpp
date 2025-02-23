
#include <cstdint>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <string>
#include <utility>

#include "slorado.h"
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

void scale_signal(torch::Tensor &signal, float scaling, float offset, SignalNormalisationParams &scaling_params) {
    auto strategy = scaling_params.strategy;
    float scale = 1.0f;
    float shift = 0.0f;
    int trim_start = 0;

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

torch::Tensor tensor_from_record(slow5_rec_t *rec) {
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt16);
    return torch::from_blob(rec->raw_signal, rec->len_raw_signal, options);
}

std::vector<chunk_t> chunks_from_tensor(torch::Tensor &tensor, size_t chunk_size, int overlap) {
    std::vector<chunk_t> chunks;

    size_t raw_size = tensor.size(0);
    size_t offset = 0;
    size_t chunk_in_read_idx = 0;
    size_t signal_chunk_step = chunk_size - overlap;
    chunks.push_back({offset, chunk_in_read_idx++, chunk_size, std::string(), std::string(), std::vector<uint8_t>()});

    while (offset + chunk_size < raw_size) {
        offset = std::min(offset + signal_chunk_step, raw_size - chunk_size);
        chunks.push_back({offset, chunk_in_read_idx++, chunk_size, std::string(), std::string(), std::vector<uint8_t>()});
    }

    return chunks;
}

std::vector<torch::Tensor> tensor_as_chunks(torch::Tensor &signal, std::vector<chunk_t> &chunks, size_t chunk_size) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(chunks.size());

    for (size_t i = 0; i < chunks.size(); ++i) {
        auto input_slice = signal.index({torch::indexing::Ellipsis, torch::indexing::Slice(chunks[i].input_offset, chunks[i].input_offset + chunk_size)});
        if (input_slice.ndimension() == 1) {
            input_slice = input_slice.unsqueeze(0);
        }
        size_t slice_size = input_slice.size(1);

        // repeat-pad any non-full chunks
        if (slice_size != chunk_size) {
            auto t0 = std::div((int)chunk_size, (int)slice_size);
            auto n = t0.quot;
            auto overhang = t0.rem;
            input_slice = torch::concat(
                {
                    input_slice.repeat({1, n}),
                    input_slice.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, overhang)})
                },
                1
            );
        }
        tensors.push_back(input_slice);
    }

    return tensors;
}

int div_round_closest(const int n, const int d) {
    return ((n < 0) ^ (d < 0)) ? ((n - d/2)/d) : ((n + d/2)/d);
}

void stitch_chunks(std::vector<chunk_t> &chunks, std::string &sequence, std::string &qstring) {
    // Calculate the chunk down sampling, round to closest int.
    int down_sampling = div_round_closest(chunks[0].raw_chunk_size, chunks[0].moves.size());

    int start_pos = 0;
    std::vector<std::string> sequences;
    std::vector<std::string> qstrings;
    for (size_t i = 0; i < chunks.size() - 1; i++){
        chunk_t &current_chunk = chunks[i];
        chunk_t &next_chunk = chunks[i+1];
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