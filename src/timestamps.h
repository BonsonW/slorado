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
    double_t time_forward;
} timestamps_t;

#endif
