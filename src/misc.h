/* @file misc.h
**
** miscellaneous definitions and inline functions
** @@
******************************************************************************/

#ifndef MISC_H
#define MISC_H

#include <sys/resource.h>
#include <sys/time.h>
#include <stdint.h>
#include <math.h>

/* time stamps */
typedef struct {
    double_t time_read;
    double_t time_tens;
    double_t time_trim;
    double_t time_scale;
    double_t time_chunk;
    double_t time_copy;
    double_t time_pad;
    double_t time_accept;
    double_t time_basecall;
    double_t time_decode;
    double_t time_stitch;
    double_t time_write;
    double_t time_total;
    
} timestamps_t;

double realtime(void);

double cputime(void);

long peakrss(void);

// Prints to the provided buffer a nice number of bytes (KB, MB, GB, etc)
void print_size(const char* name, uint64_t bytes);

int64_t mm_parse_num(const char* str);

void yes_or_no(uint64_t* flag_a, uint64_t flag, const char* opt_name, const char* arg, int yes_to_set);

void init_timestamps(timestamps_t* time_stamps);

#endif
