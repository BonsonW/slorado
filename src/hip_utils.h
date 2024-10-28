#ifndef HIP_UTILS_H
#define HIP_UTILS_H

#include <hip/hip_runtime.h>
#include <cstdint>

#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

/// \brief Checks if the provided error code is \p hipSuccess and if not,
/// prints an error message to the standard error output and terminates the program
/// with an error code.
#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        const hipError_t error = (condition);                                               \
        if (error != hipSuccess)                                                            \
        {                                                                                   \
            ERROR("%s", hipGetErrorString(error));                                          \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    }

static inline uint64_t hip_freemem(int32_t devicenum) {

    uint64_t freemem, total;
    HIP_CHECK(hipMemGetInfo(&freemem, &total));
    fprintf(stderr, "[%s] %lu free of total %lu GPU memory\n",__func__, freemem, total);

    return freemem;
}

#ifdef __cplusplus
}
#endif

#endif // HIP_UTILS_H