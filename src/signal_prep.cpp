
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
#include <slow5/slow5.h>
#include <torch/torch.h>
#include <vector>

#include "Chunk.h"
#include "slorado.h"

#define EPS 1e-9f;

std::pair<float, float> calculate_med_mad(torch::Tensor &x, float factor=1.4826){
    torch::Tensor med = x.median();
    torch::Tensor mad = torch::median(torch::abs(x - med)) * factor + EPS;

    return {med.item<float>(), mad.item<float>()};
}

slow5_rec_t *read_file_to_record(char *file_path) {
    slow5_file_t *sp = slow5_open(file_path, "r");
    if (sp == NULL) {
        fprintf(stderr, "Error in opening slow5 file %s\n", file_path);
        exit(EXIT_FAILURE);
    }

    int ret = 0;

    ret = slow5_idx_load(sp);
    if (ret < 0) {
        fprintf(stderr, "Error in loading index for slow5 file %s\n", file_path);
        exit(EXIT_FAILURE);
    }

    slow5_rec_t *rec = NULL;

    ret = slow5_get("r3", &rec, sp);
    if (ret < 0) {
        fprintf(stderr, "Error when fetching the read for slow5 file %s\n", file_path);
        exit(EXIT_FAILURE);
    }

    slow5_idx_unload(sp);
    slow5_close(sp);

    return rec;
}

int trim_signal(torch::Tensor signal, int window_size, float threshold_factor, int min_elements) {
    int min_trim = 10;
    signal = signal.index({torch::indexing::Slice(min_trim, torch::indexing::None)});

    int trim_start = -(window_size * 100);

    torch::Tensor trimmed = signal.index({torch::indexing::Slice(trim_start, torch::indexing::None)});
    std::pair<float, float> med_mad = calculate_med_mad(trimmed);

    float threshold = med_mad.first + med_mad.second * threshold_factor;

    int64_t signal_len = signal.size(0);
    int num_windows = signal_len / window_size;

    bool seen_peak = false;

    for (int pos = 0; pos < num_windows; pos++) {
        int start = pos * window_size;
        int end = start + window_size;

        torch::Tensor window = signal.index({torch::indexing::Slice(start, end)});
        torch::Tensor elements = window > threshold;


        if ((elements.sum().item<int>() > min_elements) || seen_peak) {
            seen_peak = true;
            if (window[-1].item<float>() > threshold) {
                continue;
            }
            return std::min(end + min_trim, (int) signal.size(0));
        }
    }

    return min_trim;
}

void scale_signal(torch::Tensor &signal) {
    std::pair<float, float> med_mad = calculate_med_mad(signal);
    float med = med_mad.first;
    float mad = med_mad.second;

    signal = (signal - med) / std::max(1.0f, mad);
}

torch::Tensor tensor_from_record(slow5_rec_t *rec) {
    std::vector<int16_t> tmp(rec->raw_signal,rec->raw_signal+rec->len_raw_signal);
    std::vector<float> floatTmp(tmp.begin(), tmp.end());
    
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
    return torch::from_blob(floatTmp.data(), floatTmp.size(), options).clone().to("cpu");
}

std::vector<Chunk> chunks_from_tensor(torch::Tensor &tensor, int chunk_size, int overlap) {
    std::vector<Chunk> chunks;

    size_t raw_size = tensor.size(0);
    size_t offset = 0;
    size_t chunk_in_read_idx = 0;
    size_t signal_chunk_step = chunk_size - overlap;
    chunks.push_back(Chunk(offset, chunk_in_read_idx++, chunk_size));

    while (offset + chunk_size < raw_size) {
        offset = std::min(offset + signal_chunk_step, raw_size - chunk_size);
        chunks.push_back(Chunk(offset, chunk_in_read_idx++, chunk_size));
    }

    return chunks;
}