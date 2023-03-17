/* @file basecall.h
**
** methods for base calling step
** @@
******************************************************************************/

#ifndef BASECALL
#define BASECALL

#include "decode/Decoder.h"
#include "misc.h"
#include "nn/ModelRunner.h"
#include "Chunk.h"
#include "slorado.h"

#include <vector>

void basecall_chunks(std::vector<torch::Tensor> &tensors, std::vector<Chunk *> &chunks, int chunk_size, int batch_size, ModelRunnerBase &model_runner, ModelRunnerBase &decoder, timestamps_t *ts);

#endif