
#include <cstdint>
#include <stdlib.h>
#include <vector>

#include "signal_prep.h"

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