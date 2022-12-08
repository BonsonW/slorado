#ifndef SIGNAL_PREP_H
#define SIGNAL_PREP_H

#include <slow5/slow5.h>
#include <torch/torch.h>

#include "Chunk.h"

slow5_rec_t *read_file_to_record(char *file_path);
torch::Tensor tensor_from_record(slow5_rec_t *rec);
int trim_signal(torch::Tensor signal, int window_size=40, float threshold_factor=2.4, int min_elements=3);
void scale_signal(torch::Tensor &signal);
std::vector<Chunk> chunks_from_tensor(torch::Tensor &tensor, int chunk_size, int overlap);

#endif