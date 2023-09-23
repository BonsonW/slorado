
#include <cstdint>
#include <stdlib.h>
#include <vector>

#include "signal_prep.h"

#define EPS 1e-9f;

using Slice = torch::indexing::Slice;

std::pair<float, float> normalisation(torch::Tensor& x, SignalNormalisationParams& scaling_params) {
    //Calculate shift and scale factors for normalisation.
    auto quantiles = quantile_counting(x, torch::tensor({scaling_params.quantile_a, scaling_params.quantile_b}));
    float qa = quantiles[0].item<float>();
    float qb = quantiles[1].item<float>();
    float shift = std::max(10.0f, scaling_params.shift_multiplier * (qa + qb));
    float scale = std::max(1.0f, scaling_params.scale_multiplier * (qb - qa));
    return std::make_pair(shift, scale);
}

std::pair<float, float> med_mad(torch::Tensor &x, float factor=1.4826){
    torch::Tensor med = x.median();
    torch::Tensor mad = torch::median(torch::abs(x - med)) * factor + EPS;

    return {med.item<float>(), mad.item<float>()};
}

void scale_signal(torch::Tensor &signal, float scaling, float offset, SignalNormalisationParams& scaling_params) {
    auto t1 = scaling_params.quantile_scaling
                ? normalisation(signal, scaling_params)
                : med_mad(signal);

    auto shift = std::get<0>(t1);
    auto scale = std::get<1>(t1);

    signal = ((signal.to(torch::kFloat) - shift) / scale).to(torch::kFloat16);

    // 8000 value may be changed in future. Currently this is found to work well.
    int max_samples = std::min(8000, static_cast<int>(signal.size(0) / 2));

    int trim_start = trim(signal.index({Slice(torch::indexing::None, max_samples)}));
    signal = signal.index({Slice(trim_start, torch::indexing::None)});
}

int trim(const torch::Tensor &signal, float threshold, int window_size, int min_elements) {
    const int min_trim = 10;
    const int num_samples = static_cast<int>(signal.size(0)) - min_trim;
    const int num_windows = num_samples / window_size;

    // Access via raw pointers because of torch indexing overhead.
    const auto signal_f32 = signal.to(torch::kFloat32);
    assert(signal_f32.is_contiguous());
    const float *const signal_f32_ptr = signal_f32.data_ptr<float>();

    bool seen_peak = false;
    for (int pos = 0; pos < num_windows; ++pos) {
        const int start = pos * window_size + min_trim;
        const int end = start + window_size;
        assert(start < signal.size(0));
        assert(end <= signal.size(0));  // end is exclusive

        const auto num_large_enough =
                std::count_if(&signal_f32_ptr[start], &signal_f32_ptr[end],
                              [threshold](float elem) { return elem > threshold; });

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

torch::Tensor tensor_from_record(slow5_rec_t *rec) {
    std::vector<int16_t> tmp(rec->raw_signal, rec->raw_signal+rec->len_raw_signal);
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