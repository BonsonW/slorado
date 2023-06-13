/**
 * @file timestamps.h
 * @brief timestamp related implementations for slorado
 * @author Aminda Amarasinghe
 * */

#ifndef TIMESTAMPS_H
#define TIMESTAMPS_H

#include <stdlib.h>
#include <stdint.h>
#include <slow5/slow5.h>
#include <vector>
#include <memory>
<<<<<<< HEAD
#include <atomic>
=======

#include "misc.h"
>>>>>>> 07e37c48ff466d6879fee2cb110f23ee94bd85dd

/* time stamps */
typedef struct {
    double_t time_init_runners;
    double_t time_read;
    double_t time_tens;
    double_t time_trim;
    double_t time_scale;
    double_t time_chunk;
    double_t time_copy;
    double_t time_pad;
    double_t time_assign;
    double_t time_accept;
    double_t time_basecall;
    double_t time_decode;
    double_t time_stitch;
    double_t time_sync;
    double_t time_write;
    double_t time_total;
    double_t time_beam_search_emplace;
<<<<<<< HEAD
    double_t time_threads_emplace_back;
    double_t time_score;
=======
    double_t time_forward;
>>>>>>> 07e37c48ff466d6879fee2cb110f23ee94bd85dd
} timestamps_t;

typedef struct {
	std::atomic<double_t> linearCRFImpl {0};
    	std::atomic<double_t> convolutionImpl {0};
} timestamps_CRF;

extern timestamps_CRF * ts_CRF;
#endif
