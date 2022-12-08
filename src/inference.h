/* @file inference.h
**
** methods for base calling step
** @@
******************************************************************************/

#ifndef INFERENCE
#define INFERENCE

#include "decode/Decoder.h"
#include "nn/ModelRunner.h"
#include "Chunk.h"

#include <vector>

std::vector<DecodedChunk> basecall_chunks(torch::Tensor &signal, std::vector<Chunk> &chunks, int chunk_size, ModelRunnerBase &model_runner);

#endif