/* lstm_model_metal.mm — host driver for dorado's Metal LSTM + reorder kernels (see the header and
 * lstm_model_metal.metal). Mirrors dorado's MetalCRFModel LSTM path: reorder conv output (fp32
 * [T,N,C] row-major) into the interleaved LSTM layout, run each layer with the tiled `lstm` kernel
 * (one dispatch per <=20 timestep piece, direction by kLstmReversedInTime), then reorder the final
 * reverse-layer output back to fp32 [T,N,C]. Built with `xcrun clang++ -x objective-c++ -fobjc-arc`.
 */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>

#include <vector>
#include <cstdio>
#include <cstring>

#include "lstm_model_metal.h"
#include "lstm_model_metal_src.h"   // KERNELS_METAL_SRC[] — generated from the .metal at build time

static constexpr int kTileSize = 8;
static constexpr int kSIMDWidth = 32;
static constexpr int kMaxTimeSteps = 512;   // fewer piece-boundary GPU syncs than dorado's 20, while
                                            // keeping each kernel launch bounded (dorado caps at 20 to
                                            // avoid command-buffer submission errors on long kernels)
static constexpr int kLstmGates = 4;

struct metal_lstm_ctx {
    int lstm_size = 0, num_layers = 0, reverse_first = 0;
    int simd_groups = 16, thread_groups = 8;
    bool ok = false;

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> lstm_cps[2] = {nil, nil};   // [kLstmReversedInTime]
    id<MTLComputePipelineState> reorder_in_fwd = nil, reorder_in_rev = nil, reorder_out = nil;
    std::vector<id<MTLComputePipelineState>> _unused;
    std::vector<id<MTLBuffer>> weights;                     // per-layer [3C+1,C,4] fp16

    // working buffers, (re)allocated when (N,T) changes
    int cur_N = 0, cur_T = 0;
    id<MTLBuffer> working = nil, state = nil, reorder_args = nil;   // scratch; in/out are caller-supplied
    std::vector<id<MTLBuffer>> args;                        // per <=20-step piece
};

static int simd_groups_for(int lstm_size) {
    switch (lstm_size) {
        case 128: return 16;
        case 192: return 12;
        case 256: return 32;
        case 384: return 24;
        case 512: return 32;
        case 768: return 32;
        case 1024: return 32;
        default: return 16;
    }
}

// GPU core count (used as the threadgroup count, like dorado). Falls back to 8.
static int gpu_core_count() {
    int cores = 0;
    CFMutableDictionaryRef match = IOServiceMatching("AGXAccelerator");
    io_iterator_t it = 0;
    if (IOServiceGetMatchingServices((mach_port_t)0, match, &it) == KERN_SUCCESS) {
        io_object_t obj;
        while ((obj = IOIteratorNext(it))) {
            CFTypeRef p = IORegistryEntrySearchCFProperty(
                obj, kIOServicePlane, CFSTR("gpu-core-count"), kCFAllocatorDefault,
                kIORegistryIterateRecursively);
            if (p) {
                if (CFGetTypeID(p) == CFNumberGetTypeID()) {
                    int v = 0;
                    CFNumberGetValue((CFNumberRef)p, kCFNumberIntType, &v);
                    if (v > 0) cores = v;
                }
                CFRelease(p);
            }
            IOObjectRelease(obj);
        }
        IOObjectRelease(it);
    }
    return cores > 0 ? cores : 8;
}

static int layer_reverse(const metal_lstm_ctx *c, int layer) {
    // reverse_first => layers alternate reverse,forward,reverse,... starting reverse.
    return c->reverse_first ? ((layer & 1) == 0) : ((layer & 1) == 1);
}

static id<MTLComputePipelineState> make_cps(id<MTLDevice> dev, id<MTLLibrary> lib, NSString *name,
                                            MTLFunctionConstantValues *fcv) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name constantValues:fcv error:&err];
    if (!fn) {
        fprintf(stderr, "[metal_lstm] function '%s' failed: %s\n", name.UTF8String,
                err ? err.localizedDescription.UTF8String : "?");
        return nil;
    }
    id<MTLComputePipelineState> cps = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!cps) {
        fprintf(stderr, "[metal_lstm] pipeline '%s' failed: %s\n", name.UTF8String,
                err ? err.localizedDescription.UTF8String : "?");
    }
    return cps;
}

extern "C" metal_lstm_ctx_t *metal_lstm_create(int lstm_size, int num_layers, int reverse_first) {
    metal_lstm_ctx *c = new metal_lstm_ctx();
    c->lstm_size = lstm_size;
    c->num_layers = num_layers;
    c->reverse_first = reverse_first;
    c->weights.resize(num_layers, nil);

    // dorado only provides reorder_rev_lstm_output_to_linear, so the last layer must be reverse.
    if (layer_reverse(c, num_layers - 1) != 1) {
        fprintf(stderr, "[metal_lstm] last layer not reverse (num_layers=%d, reverse_first=%d); "
                        "no output reorder kernel -> falling back to ATen\n", num_layers, reverse_first);
        return c;  // ok stays false
    }

    c->device = MTLCreateSystemDefaultDevice();
    if (!c->device) { fprintf(stderr, "[metal_lstm] no Metal device\n"); return c; }
    c->queue = [c->device newCommandQueue];

    NSError *err = nil;
    NSString *src = [NSString stringWithUTF8String:KERNELS_METAL_SRC];
    MTLCompileOptions *opts = [MTLCompileOptions new];
    id<MTLLibrary> lib = [c->device newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        fprintf(stderr, "[metal_lstm] kernel compile failed: %s\n",
                err ? err.localizedDescription.UTF8String : "?");
        return c;
    }

    MTLFunctionConstantValues *fcv = [MTLFunctionConstantValues new];
    int ls = lstm_size;
    [fcv setConstantValue:&ls type:MTLDataTypeInt atIndex:0];   // kLstmLayerSize
    bool bfalse = false, btrue = true;
    [fcv setConstantValue:&bfalse type:MTLDataTypeBool atIndex:1];  // kLstmReversedInTime
    c->lstm_cps[0] = make_cps(c->device, lib, @"lstm", fcv);
    [fcv setConstantValue:&btrue type:MTLDataTypeBool atIndex:1];
    c->lstm_cps[1] = make_cps(c->device, lib, @"lstm", fcv);

    // reorder kernels only reference kLstmLayerSize.
    MTLFunctionConstantValues *fcv2 = [MTLFunctionConstantValues new];
    [fcv2 setConstantValue:&ls type:MTLDataTypeInt atIndex:0];
    c->reorder_in_fwd = make_cps(c->device, lib, @"reorder_input_to_fwd_lstm_output", fcv2);
    c->reorder_in_rev = make_cps(c->device, lib, @"reorder_input_to_rev_lstm_output", fcv2);
    c->reorder_out = make_cps(c->device, lib, @"reorder_rev_lstm_output_to_linear", fcv2);

    c->simd_groups = simd_groups_for(lstm_size);
    c->thread_groups = gpu_core_count();
    c->ok = c->lstm_cps[0] && c->lstm_cps[1] && c->reorder_in_fwd && c->reorder_in_rev && c->reorder_out;
    if (!c->ok) fprintf(stderr, "[metal_lstm] pipeline creation incomplete -> ATen fallback\n");
    return c;
}

extern "C" int metal_lstm_ok(const metal_lstm_ctx_t *ctx) {
    return ctx && ctx->ok ? 1 : 0;
}

extern "C" int metal_lstm_layer_reverse(const metal_lstm_ctx_t *ctx, int layer) {
    return layer_reverse(ctx, layer);
}

extern "C" void metal_lstm_set_layer(metal_lstm_ctx_t *ctx, int layer, const void *w, size_t bytes) {
    if (!ctx || !ctx->device || layer < 0 || layer >= ctx->num_layers) return;
    ctx->weights[layer] = [ctx->device newBufferWithBytes:w length:bytes
                                                  options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> make_args(id<MTLDevice> dev, int batch_tiles, int T, int begin, int end) {
    int32_t a[4] = {batch_tiles, T, begin, end};
    return [dev newBufferWithBytes:a length:sizeof(a) options:MTLResourceStorageModeShared];
}

// Allocate the per-(N,T) scratch (recurrence ring + cell state + arg buffers). in/out are supplied
// by the caller (the conv-output and result MPS tensors' MTLBuffers) for a zero-copy run.
static void ensure_buffers(metal_lstm_ctx *c, int N, int T) {
    if (c->cur_N == N && c->cur_T == T && c->working) return;
    const int C = c->lstm_size;
    c->working = [c->device newBufferWithLength:(size_t)(T + 3) * N * C * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    c->state   = [c->device newBufferWithLength:(size_t)N * C * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    const int batch_tiles = N / kTileSize;
    c->reorder_args = make_args(c->device, batch_tiles, T, 0, T);
    c->args.clear();
    for (int b = 0; b < T; b += kMaxTimeSteps) {
        int e = b + kMaxTimeSteps < T ? b + kMaxTimeSteps : T;
        c->args.push_back(make_args(c->device, batch_tiles, T, b, e));
    }
    c->cur_N = N; c->cur_T = T;
}

static void encode_reorder(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> cps,
                           id<MTLBuffer> args, id<MTLBuffer> in, id<MTLBuffer> out,
                           int thread_groups, int threads) {
    [enc setComputePipelineState:cps];
    [enc setBuffer:args offset:0 atIndex:0];
    [enc setBuffer:in offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(thread_groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

// Zero-copy run: in_mtl/out_mtl are MTLBuffers (from the conv-output and result MPS tensors'
// storage().data()); the kernels read/write them in place, so no host round-trip. The caller must
// have synced torch's MPS stream first (openfish/LSTM use a separate command queue).
extern "C" int metal_lstm_run(metal_lstm_ctx_t *ctx, int N, int T, const void *in_mtl, const void *out_mtl) {
    if (!ctx || !ctx->ok) return 1;
    if (N % (kTileSize * 6) != 0) {   // SIMD_TILES_M(6) * TILE_SIZE(8) = 48
        fprintf(stderr, "[metal_lstm] N=%d not a multiple of 48\n", N);
        return 1;
    }
    ensure_buffers(ctx, N, T);
    id<MTLBuffer> in_buf  = (__bridge id<MTLBuffer>)in_mtl;
    id<MTLBuffer> out_buf = (__bridge id<MTLBuffer>)out_mtl;

    const int threads = ctx->simd_groups * kSIMDWidth;
    const int res_bytes = (int)sizeof(uint16_t) * ctx->simd_groups * 2 * kTileSize * kTileSize;
    const int out_bytes = (int)sizeof(uint16_t) * ctx->simd_groups * kTileSize * kTileSize;

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        // conv output (fp32 row-major [T,N,C]) -> interleaved LSTM layout (fp16), layer-0 direction.
        bool rev0 = layer_reverse(ctx, 0);
        encode_reorder(enc, rev0 ? ctx->reorder_in_rev : ctx->reorder_in_fwd, ctx->reorder_args,
                       in_buf, ctx->working, ctx->thread_groups, threads);

        for (int l = 0; l < ctx->num_layers; ++l) {
            id<MTLComputePipelineState> cps = ctx->lstm_cps[layer_reverse(ctx, l) ? 1 : 0];
            for (id<MTLBuffer> a : ctx->args) {
                [enc setComputePipelineState:cps];
                [enc setBuffer:a offset:0 atIndex:0];
                [enc setBuffer:ctx->working offset:0 atIndex:1];
                [enc setBuffer:ctx->weights[l] offset:0 atIndex:2];
                [enc setBuffer:ctx->state offset:0 atIndex:3];
                [enc setThreadgroupMemoryLength:res_bytes atIndex:0];
                [enc setThreadgroupMemoryLength:out_bytes atIndex:1];
                [enc dispatchThreadgroups:MTLSizeMake(ctx->thread_groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            }
        }

        // interleaved reverse-layer output -> fp16 row-major [T,N,C] (in the caller's out MTLBuffer).
        encode_reorder(enc, ctx->reorder_out, ctx->reorder_args, ctx->working, out_buf,
                       ctx->thread_groups, threads);

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "[metal_lstm] command buffer error: %s\n",
                    cb.error ? cb.error.localizedDescription.UTF8String : "?");
            return 1;
        }
    }
    return 0;   // results are already in the caller's out MTLBuffer (zero-copy)
}

extern "C" void metal_lstm_free(metal_lstm_ctx_t *ctx) {
    delete ctx;   // ARC releases the id<> members (device/queue/pipelines/buffers/weights)
}
