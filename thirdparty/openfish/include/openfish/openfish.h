#ifndef OPENFISH_H
#define OPENFISH_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DECODER_INIT {32, 100.0, 2.0, 0.0, 1.0, 1.0, false}

typedef struct decoder_opts {
    size_t beam_width;
    float beam_cut;
    float blank_score;
    float q_shift;
    float q_scale;
    float temperature;
    bool move_pad;
} decoder_opts_t;

void decode_cpu(
    const int T,
    const int N,
    const int C,
    int nthreads,
    void *scores_TNC,
    const int state_len,
    const decoder_opts_t *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

void decode_gpu(
    const int T,
    const int N,
    const int C,
    void *scores_TNC,
    const int state_len,
    const decoder_opts_t *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

#ifdef __cplusplus
}
#endif

#endif // OPENFISH_H