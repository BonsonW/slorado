/* @file signal_prep.h
**
** methods for preparing a signal for the base calling step
** @@
******************************************************************************/

#ifndef SIGNAL_PREP_H
#define SIGNAL_PREP_H

#include <slow5/slow5.h>

#include "utils/tensor_utils.h"
#include "nn/CRFModel.h"
#include "Chunk.h"

torch::Tensor tensor_from_record(slow5_rec_t *rec);
std::pair<float, float> normalisation(torch::Tensor& x);
int trim(
    const torch::Tensor &signal,
    float threshold = 2.4,
    int window_size = 40,
    int min_elements = 3
);

void scale_signal(torch::Tensor &signal, float scaling, float offset, SignalNormalisationParams& scale_params);
std::vector<Chunk *> chunks_from_tensor(torch::Tensor &tensor, int chunk_size, int overlap);
std::vector<torch::Tensor> tensor_as_chunks(torch::Tensor &signal, std::vector<Chunk *> &chunks, size_t chunk_size);

#endif
