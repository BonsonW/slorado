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

// int8 CRF emission scores. When true, the flstm CRF head emits int8 scores in [-127,127]
// (dorado-style tanh*127) and openfish decodes them with score_scale = 5/127 -- halving the [N,T,C]
// scores tensor. When false, scores stay fp16 (score_scale = 1.0). Decode picks the path from the
// tensor dtype, so this only gates the CRF output. Set by init_core from whether --quant is enabled
// (tx/sup always stays fp16 -- its unbounded CRF breaks under a fixed ±5 int8 clamp).
extern bool g_scores_i8;
// Dequant multiplier for int8 CRF scores (matches dorado ±5 clamp -> [-127,127]).
#define SCORES_I8_SCALE (5.0f / 127.0f)

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
