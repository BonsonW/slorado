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

void basecall_chunks(
    std::vector<torch::Tensor> tensors,
    std::vector<Chunk *> chunks,
    int chunk_size,
    int batch_size,
    ModelRunnerBase &model_runner,
    ModelRunnerBase &decoder,
    timestamps_t *ts
);

void basecall_thread(
    core_t* core,
    db_t* db,
    size_t runner_idx,
    size_t start,
    size_t end
);

#endif