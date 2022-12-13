/* @file basecall.h
**
** methods for base calling step
** @@
******************************************************************************/

#ifndef BASECALL
#define BASECALL

#include "decode/Decoder.h"
#include "nn/ModelRunner.h"
#include "Chunk.h"

#include <vector>

void basecall_chunks(torch::Tensor &signal, std::vector<Chunk> &chunks, int chunk_size, int batch_size, ModelRunnerBase &model_runner, double &time_basecall, double &time_decode);

#endif