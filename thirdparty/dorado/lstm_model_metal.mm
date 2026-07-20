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
static constexpr int kMaxTimeSteps = 512;   // timesteps per LSTM kernel dispatch (state kept resident
                                            // across the piece). Swept 20..1024: no measurable effect
                                            // (kernel is gate-matmul bound), so left at 512.
static constexpr int kLstmGates = 4;

struct metal_lstm_ctx {
    int lstm_size = 0, num_layers = 0, reverse_first = 0;
    int simd_groups = 16, thread_groups = 8;
    bool ok = false;

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> lib = nil;                               // kept for lazy conv pipeline creation
    id<MTLComputePipelineState> lstm_cps[2] = {nil, nil};   // [kLstmReversedInTime]
    id<MTLComputePipelineState> reorder_in_fwd = nil, reorder_in_rev = nil, reorder_out = nil;
    std::vector<id<MTLComputePipelineState>> _unused;
    std::vector<id<MTLBuffer>> weights;                     // per-layer [3C+1,C,4] fp16

    // working buffers, (re)allocated when (N,T) changes
    int cur_N = 0, cur_T = 0;
    id<MTLBuffer> working = nil, state = nil, reorder_args = nil;   // scratch; in/out are caller-supplied
    std::vector<id<MTLBuffer>> args;                        // per <=20-step piece

    // Optional fused conv stack (conv1->conv2->conv3 writing the LSTM layout directly, dropping the
    // ATen conv + reorder_input). Populated by metal_lstm_set_conv; when set, the run takes the raw
    // signal MTLBuffer instead of the conv-output.
    bool has_conv = false;
    struct conv_layer_m {
        id<MTLComputePipelineState> cps = nil;
        id<MTLBuffer> w = nil;                              // dorado padded weight+bias layout
        int in_size = 0, out_size = 0, win = 0, stride = 0, pad = 0, chunk_in = 0, simd_groups = 16;
        std::vector<id<MTLBuffer>> args;                    // conv3 is split into <=kMaxTimeSteps pieces
    } conv[3];
    id<MTLBuffer> conv_bufA = nil, conv_bufB = nil;        // conv1->A, conv2->B; conv3->working
    int conv_cur_N = 0, conv_cur_chunk = 0;
};

// Host mirror of the shader's ConvArgs (see lstm_model_metal.metal).
struct ConvArgsHost {
    int32_t in_size, win_size, out_size, stride, pad, chunk_size_in, num_chunks, ts_begin, ts_end;
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
    c->lib = lib;

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

extern "C" int metal_lstm_has_conv(const metal_lstm_ctx_t *ctx) {
    return ctx && ctx->ok && ctx->has_conv ? 1 : 0;
}

extern "C" int metal_lstm_layer_reverse(const metal_lstm_ctx_t *ctx, int layer) {
    return layer_reverse(ctx, layer);
}

extern "C" void metal_lstm_set_layer(metal_lstm_ctx_t *ctx, int layer, const void *w, size_t bytes) {
    if (!ctx || !ctx->device || layer < 0 || layer >= ctx->num_layers) return;
    ctx->weights[layer] = [ctx->device newBufferWithBytes:w length:bytes
                                                  options:MTLResourceStorageModeShared];
}

static id<MTLComputePipelineState> make_conv_cps(id<MTLDevice> dev, id<MTLLibrary> lib,
                                                 const char *name, bool clamp, bool tanh_act) {
    MTLFunctionConstantValues *fcv = [MTLFunctionConstantValues new];
    bool c = clamp, t = tanh_act;
    [fcv setConstantValue:&c type:MTLDataTypeBool atIndex:4];   // kConvOutputClamp
    [fcv setConstantValue:&t type:MTLDataTypeBool atIndex:9];   // kConvTanhActivation
    return make_cps(dev, lib, [NSString stringWithUTF8String:name], fcv);
}

// Register one conv layer (1,2,3). w is the dorado padded weight+bias layout [rows, new_out_size]
// (built ATen-side in lstm_model.cpp). Args (which need N/chunk) are built lazily in ensure_conv_bufs.
extern "C" void metal_lstm_set_conv(metal_lstm_ctx_t *ctx, int layer, int in_size, int out_size,
                                    int win, int stride, int clamp, const void *w, size_t bytes) {
    if (!ctx || !ctx->device || layer < 1 || layer > 3) return;
    auto &cl = ctx->conv[layer - 1];
    cl.in_size = in_size; cl.out_size = out_size; cl.win = win; cl.stride = stride; cl.pad = win / 2;
    cl.w = [ctx->device newBufferWithBytes:w length:bytes options:MTLResourceStorageModeShared];
    cl.simd_groups = (layer == 3 || (layer == 2 && in_size == 16)) ? 4 : 16;

    char name[64];
    if (layer == 1)      snprintf(name, sizeof name, "conv1_in%d_out%d_simd", in_size, out_size);
    else if (layer == 2) snprintf(name, sizeof name, "conv2_in%d_out%d_simd", in_size, out_size);
    else                 snprintf(name, sizeof name, "conv3_simd");
    cl.cps = make_conv_cps(ctx->device, ctx->lib, name, clamp != 0, /*tanh=*/false);
    ctx->has_conv = ctx->conv[0].cps && ctx->conv[1].cps && ctx->conv[2].cps;
}

// conv intermediate buffers ([N, chunk, 16] fp16) + per-layer ConvArgs (need N + signal length).
static void ensure_conv_bufs(metal_lstm_ctx *c, int N, int chunk) {
    if (c->conv_cur_N == N && c->conv_cur_chunk == chunk && c->conv_bufA) return;
    const int inter = 16;   // conv1/conv2 out_size
    c->conv_bufA = [c->device newBufferWithLength:(size_t)N * chunk * inter * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    c->conv_bufB = [c->device newBufferWithLength:(size_t)N * chunk * inter * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    for (int l = 0; l < 3; ++l) {
        auto &cl = c->conv[l];
        cl.args.clear();
        if (l != 2) {   // conv1/conv2: single launch (begin/end unused)
            ConvArgsHost a{cl.in_size, cl.win, cl.out_size, cl.stride, cl.pad, chunk, N, 0, 0};
            cl.args.push_back([c->device newBufferWithBytes:&a length:sizeof a options:MTLResourceStorageModeShared]);
        } else {        // conv3: split output time range into pieces (heaviest kernel)
            const int out_ts = chunk / cl.stride;
            const int piece = 128;
            for (int b = 0; b < out_ts; b += piece) {
                int e = b + piece < out_ts ? b + piece : out_ts;
                ConvArgsHost a{cl.in_size, cl.win, cl.out_size, cl.stride, cl.pad, chunk, N, b, e};
                cl.args.push_back([c->device newBufferWithBytes:&a length:sizeof a options:MTLResourceStorageModeShared]);
            }
        }
    }
    c->conv_cur_N = N; c->conv_cur_chunk = chunk;
}

// --- temp instrumentation: separate pure-GPU kernel time from ATen glue ---
static double g_gpu_secs = 0.0;   // sum of (GPUEndTime-GPUStartTime) across command buffers
static double g_wall_secs = 0.0;  // sum of commit->waitUntilCompleted wall time
static long   g_calls = 0;

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

// Dispatch one conv layer (all its time-piece args). Buffers: args(0), in(1), weights(2), out(3).
static void encode_conv(id<MTLComputeCommandEncoder> enc, const metal_lstm_ctx::conv_layer_m &cl,
                        id<MTLBuffer> in, id<MTLBuffer> out, int thread_groups) {
    const int threads = cl.simd_groups * kSIMDWidth;
    [enc setComputePipelineState:cl.cps];
    [enc setBuffer:in offset:0 atIndex:1];
    [enc setBuffer:cl.w offset:0 atIndex:2];
    [enc setBuffer:out offset:0 atIndex:3];
    for (id<MTLBuffer> a : cl.args) {
        [enc setBuffer:a offset:0 atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(thread_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }
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

        if (ctx->has_conv) {
            // Fused conv stack on GPU: in_buf is the raw scaled signal [N, chunk] (fp16). conv1->A,
            // conv2->B, conv3 writes the interleaved LSTM layout (FORWARD) directly into `working`,
            // zeroing the initial state -- so this replaces both the ATen conv and reorder_input.
            const int chunk_in = T * ctx->conv[2].stride;   // signal length (conv3 divides by stride)
            const int conv_tg = ctx->thread_groups * 4;     // dorado uses core_count*4 for conv
            ensure_conv_bufs(ctx, N, chunk_in);
            encode_conv(enc, ctx->conv[0], in_buf, ctx->conv_bufA, conv_tg);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_conv(enc, ctx->conv[1], ctx->conv_bufA, ctx->conv_bufB, conv_tg);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_conv(enc, ctx->conv[2], ctx->conv_bufB, ctx->working, conv_tg);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        } else {
            // conv output (row-major [T,N,C]) -> interleaved LSTM layout (fp16), FORWARD (dorado
            // convention; per-layer directions match ours so it decodes correctly).
            encode_reorder(enc, ctx->reorder_in_fwd, ctx->reorder_args,
                           in_buf, ctx->working, ctx->thread_groups, threads);
        }

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
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        [cb commit];
        [cb waitUntilCompleted];
        clock_gettime(CLOCK_MONOTONIC, &t1);
        g_wall_secs += (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        g_gpu_secs  += (cb.GPUEndTime - cb.GPUStartTime);
        g_calls++;
        if (cb.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "[metal_lstm] command buffer error: %s\n",
                    cb.error ? cb.error.localizedDescription.UTF8String : "?");
            return 1;
        }
    }
    return 0;   // results are already in the caller's out MTLBuffer (zero-copy)
}

extern "C" void metal_lstm_free(metal_lstm_ctx_t *ctx) {
    if (g_calls) {
        fprintf(stderr, "[metal_lstm] calls=%ld  pure-GPU=%.3fs  commit->wait wall=%.3fs  "
                "(glue=wall-gpu=%.3fs)\n", g_calls, g_gpu_secs, g_wall_secs, g_wall_secs - g_gpu_secs);
    }
    delete ctx;   // ARC releases the id<> members (device/queue/pipelines/buffers/weights)
}
