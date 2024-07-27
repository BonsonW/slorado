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

#include <string>
#include <vector>

double realtime(void);

double cputime(void);

long peakrss(void);

// prints to the provided buffer a nice number of bytes (KB, MB, GB, etc)
void print_size(const char* name, uint64_t bytes);

int64_t mm_parse_num(const char* str);

void yes_or_no(uint64_t* flag_a, uint64_t flag, const char* opt_name, const char* arg, int yes_to_set);

std::vector<std::string> parse_cuda_device_string(std::string device_arg);

#endif
