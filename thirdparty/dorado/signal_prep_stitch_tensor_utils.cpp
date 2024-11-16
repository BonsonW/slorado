
#include <cstdint>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <string>
#include <utility>

#include <torch/torch.h>

#include "dorado/Chunk.h"
#include "slorado.h"
#include "error.h"
#include "signal_prep_stitch_tensor_utils.h"

#define EPS 1e-9f;

std::pair<float, float> normalisation(torch::Tensor& x) {
    //Calculate shift and scale factors for normalisation.
    auto quantiles = quantile_counting(x, torch::tensor({0.2, 0.9}));
    float q20 = quantiles[0].item<float>();
    float q90 = quantiles[1].item<float>();
    float shift = std::max(10.0f, 0.51f * (q20 + q90));
    float scale = std::max(1.0f, 0.53f * (q90 - q20));
    return std::make_pair(shift, scale);
}

std::pair<float, float> calculate_med_mad(torch::Tensor &x, float factor=1.4826){
    torch::Tensor med = x.median();
    torch::Tensor mad = torch::median(torch::abs(x - med)) * factor + EPS;

    return {med.item<float>(), mad.item<float>()};
}

void scale_signal(torch::Tensor &signal, float scaling, float offset) {
    auto t1 = normalisation(signal);
    auto shift = std::get<0>(t1);
    auto scale = std::get<1>(t1);

    signal = ((signal.to(torch::kFloat) - shift) / scale).to(torch::kFloat16);

    scale = scaling * scale;
    shift = scaling * (shift + offset);

    float threshold = shift + scale * 2.4;

    // 8000 value may be changed in future. Currently this is found to work well.
    int trim_start = trim(signal.index({torch::indexing::Slice(torch::indexing::None, 8000)}), threshold);
    signal = signal.index({torch::indexing::Slice(trim_start, torch::indexing::None)});
}

int trim(
    torch::Tensor signal,
    int window_size,
    float threshold,
    int min_elements,
    int max_samples,
    float max_trim
) {
    int min_trim = 10;
    bool seen_peak = false;
    int num_samples = std::min(max_samples, static_cast<int>(signal.size(0)) - min_trim);
    int num_windows = num_samples / window_size;

    for (int pos = 0; pos < num_windows; pos++) {
        int start = pos * window_size + min_trim;
        int end = start + window_size;

        auto window = signal.index({torch::indexing::Slice(start, end)});
        auto elements = window > threshold;

        if ((elements.sum().item<int>() > min_elements) || seen_peak) {
            seen_peak = true;
            if (window[-1].item<float>() > threshold) {
                continue;
            }
            if (end >= num_samples || end >= (max_trim * signal.size(0))) {
                return min_trim;
            } else {
                return end;
            }
        }
    }

    return min_trim;
}

torch::Tensor tensor_from_record(slow5_rec_t *rec) {
    std::vector<int16_t> tmp(rec->raw_signal,rec->raw_signal+rec->len_raw_signal);
    std::vector<int16_t> floatTmp(tmp.begin(), tmp.end());

    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt16);
    return torch::from_blob(floatTmp.data(), floatTmp.size(), options).clone().to("cpu");
}

std::vector<Chunk *> chunks_from_tensor(torch::Tensor &tensor, int chunk_size, int overlap) {
    std::vector<Chunk *> chunks;

    size_t raw_size = tensor.size(0);
    size_t offset = 0;
    size_t chunk_in_read_idx = 0;
    size_t signal_chunk_step = chunk_size - overlap;
    chunks.push_back(new Chunk(offset, chunk_in_read_idx++, chunk_size));

    while (offset + chunk_size < raw_size) {
        offset = std::min(offset + signal_chunk_step, raw_size - chunk_size);
        chunks.push_back(new Chunk(offset, chunk_in_read_idx++, chunk_size));
    }

    return chunks;
}

std::vector<torch::Tensor> tensor_as_chunks(torch::Tensor &signal, std::vector<Chunk *> &chunks, size_t chunk_size) {
    std::vector<torch::Tensor> tensors;

    for (size_t i = 0; i < chunks.size(); ++i) {
        auto input_slice = signal.index({torch::indexing::Ellipsis, torch::indexing::Slice(chunks[i]->input_offset, chunks[i]->input_offset + chunk_size)});
        size_t slice_size;
        if (input_slice.ndimension() == 1) slice_size = input_slice.size(0);
        else slice_size = input_slice.sizes()[1];

        // repeat-pad any non-full chunks
        // Stereo and Simplex encoding need to be treated differently
        if (slice_size != chunk_size) {
            if (input_slice.ndimension() == 1) {
                auto t0 = std::div((int)chunk_size, (int)slice_size);
                auto n = t0.quot;
                auto overhang = t0.rem;
                input_slice = torch::concat({input_slice.repeat({n}), input_slice.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, overhang)})});
            } else if (input_slice.ndimension() == 2) {
                auto t0 = std::div((int)chunk_size, (int)slice_size);
                auto n = t0.quot;
                auto overhang = t0.rem;
                input_slice = torch::concat(
                            {input_slice.repeat({1, n}),
                             input_slice.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, overhang)})},
                            1);
            }
        }
        tensors.push_back(input_slice);
    }

    return tensors;
}


int div_round_closest(const int n, const int d)
{
    return ((n < 0) ^ (d < 0)) ? ((n - d/2)/d) : ((n + d/2)/d);
}

void stitch_chunks(std::vector<Chunk *> &chunks, std::string &sequence, std::string &qstring) {
    // Calculate the chunk down sampling, round to closest int.
    int down_sampling = div_round_closest(chunks[0]->raw_chunk_size, chunks[0]->moves.size());

    std::vector<uint8_t> moves = chunks[0]->moves;

    int start_pos = 0;
    std::vector<std::string> sequences;
    std::vector<std::string> qstrings;
    for (size_t i = 0; i < chunks.size() - 1; i++){
        Chunk &current_chunk = *chunks[i];
        Chunk &next_chunk = *chunks[i+1];
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
    sequences.push_back(chunks[chunks.size() - 1]->seq.substr(start_pos));
    qstrings.push_back(chunks[chunks.size() - 1]->qstring.substr(start_pos));

    // Set the read seq and qstring
    sequence = std::accumulate(sequences.begin(), sequences.end(), std::string(""));
    qstring = std::accumulate(qstrings.begin(), qstrings.end(), std::string(""));
}



std::vector<torch::Tensor> load_tensors(const std::string& dir,
                                        const std::vector<std::string>& tensors) {
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