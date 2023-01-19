
/**
 * @file signal_prep.c
 * @brief signal preprocessing methods for slorado
 * @author Bonson Wong (bonson.ym@gmail.com)

MIT License

Copyright (c) 2022 Bonson Wong (bonson.ym@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


******************************************************************************/

#include <cstdint>
#include <stdlib.h>
#include <torch/torch.h>
#include <vector>

#include "Chunk.h"
#include "slorado.h"
#include "signal_prep.h"
#include "error.h"
#include "utils/tensor_utils.h"

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

    LOG_TRACE("%s", "fucksdkadkf");

    signal = (signal - shift) / scale;

    scale = scaling * scale;
    shift = scaling * (shift + offset);

    float threshold = shift + scale * 2.4;

    // 8000 value may be changed in future. Currently this is found to work well.
    int trim_start = trim(signal.index({torch::indexing::Slice(torch::indexing::None, 8000)}), threshold);
    signal = signal.index({torch::indexing::Slice(trim_start, torch::indexing::None)});
}

int trim(torch::Tensor signal,
                     int window_size,
                     float threshold,
                     int min_elements,
                     int max_samples,
                     float max_trim) {
    int min_trim = 10;
    bool seen_peak = false;
    int num_samples = std::min(max_samples, static_cast<int>(signal.size(0)));
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
        torch::Tensor signal_chunk = signal.index({ torch::indexing::Slice(chunks[i]->input_offset, chunks[i]->input_offset + chunk_size) });
        size_t slice_size = signal_chunk.size(0);
    
        if (slice_size != chunk_size) {
            signal_chunk = torch::constant_pad_nd(signal_chunk, c10::IntArrayRef{ 0, int(chunk_size - slice_size) }, 0);
        }
        tensors.push_back(signal_chunk);
    }

    return tensors;
}