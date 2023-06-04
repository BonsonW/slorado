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

//color definitions 
#define CNRM  "\x1B[0m"
#define CRED  "\x1B[31m"
#define CGRN  "\x1B[32m"
#define CYEL  "\x1B[33m"
#define CBLU  "\x1B[34m"
#define CMAG  "\x1B[35m"
#define CCYN  "\x1B[36m"
#define CWHT  "\x1B[37m"
#define CGRY_H  "\x1B[40m"
#define CRED_H  "\x1B[41m"
#define CGRN_H  "\x1B[42m"
#define CYEL_H  "\x1B[43m"
#define CBLU_H  "\x1B[44m"
#define CMAG_H  "\x1B[45m"
#define CCYN_H  "\x1B[46m"
#define CWHT_H  "\x1B[47m"
#define CRESET "\033[0m"

double realtime(void);

double cputime(void);

long peakrss(void);

// Prints to the provided buffer a nice number of bytes (KB, MB, GB, etc)
void print_size(const char* name, uint64_t bytes);

int64_t mm_parse_num(const char* str);

void yes_or_no(uint64_t* flag_a, uint64_t flag, const char* opt_name, const char* arg, int yes_to_set);


#endif
