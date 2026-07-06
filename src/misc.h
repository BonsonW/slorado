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

// Per-stage timing sync gate. The per-stage torch::cuda::synchronize calls bracket the time_*
// counters (realtime() above) but force full-device drains, so they run ONLY in the non-streaming
// (profiling) path. Set once at startup from !(opt.flag & SLORADO_STREAM); default true (sync).
extern bool g_stage_sync;

// Self-contained one-liner for those syncs: no #ifdef USE_GPU or device check needed at the call
// site. `on_gpu` guards the CPU path (no CUDA sync there). Correctness in the streaming (no-sync)
// path is guaranteed by CUDA stream ordering, not these syncs — they are timing instrumentation.
#ifdef USE_GPU
#define STAGE_SYNC(on_gpu, dev) do { if (g_stage_sync && (on_gpu)) torch::cuda::synchronize(dev); } while (0)
#else
#define STAGE_SYNC(on_gpu, dev) do { (void)(on_gpu); (void)(dev); } while (0)
#endif

long peakrss(void);

// prints to the provided buffer a nice number of bytes (KB, MB, GB, etc)
void print_size(const char* name, uint64_t bytes);

int64_t mm_parse_num(const char* str);

void yes_or_no(uint64_t* flag_a, uint64_t flag, const char* opt_name, const char* arg, int yes_to_set);



#endif
