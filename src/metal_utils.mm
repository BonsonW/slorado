/* Metal helpers for the Apple Silicon (MPS) build — see metal_utils.h.
 * Built with `xcrun clang++ -x objective-c++ -fobjc-arc` (see the Makefile metal branch).
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <string.h>
#include "metal_utils.h"

// Cached system default device. openfish's decoder initialises its own state from
// MTLCreateSystemDefaultDevice() too, so a buffer allocated here is on the same GPU and usable by
// openfish's command queue.
static id<MTLDevice> g_device = nil;

extern "C" void *slorado_metal_upload_scores_f16(int n_timesteps, int batch_size, int n_channels,
                                                 const void *scores_f16_NTC) {
    if (g_device == nil) {
        g_device = MTLCreateSystemDefaultDevice();
    }
    const size_t bytes = (size_t)n_timesteps * (size_t)batch_size * (size_t)n_channels * sizeof(__fp16);
    id<MTLBuffer> buf = [g_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memcpy([buf contents], scores_f16_NTC, bytes);
    return (void *)CFBridgingRetain(buf);
}

extern "C" void *slorado_metal_buffer_contents(const void *mtl_buffer_handle) {
    if (mtl_buffer_handle == NULL) {
        return NULL;
    }
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)mtl_buffer_handle;
    if (buf.storageMode == MTLStorageModePrivate) {
        return NULL;   // GPU-only memory; caller must copy
    }
    return [buf contents];
}

extern "C" size_t slorado_metal_buffer_length(const void *mtl_buffer_handle) {
    if (mtl_buffer_handle == NULL) {
        return 0;
    }
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)mtl_buffer_handle;
    return (size_t)[buf length];
}

extern "C" void slorado_metal_free_scores(void *handle) {
    if (handle) {
        CFBridgingRelease(handle);
    }
}
