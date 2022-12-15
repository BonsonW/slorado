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

void basecall_chunks(torch::Tensor &signal, std::vector<Chunk> &chunks, opt_t &opt, ModelRunnerBase &model_runner, timestamps_t &ts);

#endif