#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime_api.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define checkCudaError() { gpuAssert(__FILE__, __LINE__); }

static inline void gpuAssert(const char *file, int line){
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s \n in file : %s line number : %d", cudaGetErrorString(code), file, line);
        exit(1);
   }
}

static inline uint64_t cuda_freemem(int32_t devicenum) {

    uint64_t freemem, total;
    cudaMemGetInfo(&freemem, &total);
    checkCudaError();
    fprintf(stderr, "[%s] %lu free of total %lu GPU memory\n",__func__, freemem, total);

    return freemem;
}

#ifdef __cplusplus
}
#endif

#endif // CUDA_UTILS_H