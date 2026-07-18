#include "fluke_wrapper.h"

// tensor_quant_t lives in the dorado tree; resolved via -I thirdparty.
#include "dorado/tensor_chunk_utils.h"

enum fluke_format_t fluke_parse_format(const std::string &method) {
    if (method.rfind("int8", 0) == 0) return FLUKE_FORMAT_INT8;
    // Future real-kernel formats plug in here (fp8, mxfp4). Everything else — including "fp16",
    // "dummy", empty, and the fake-quant-only methods — stays on the fp16 path.
    return FLUKE_FORMAT_NONE;
}

// The fused-int8 kernels are GPU-only. On a CPU build (neither HAVE_CUDA nor HAVE_ROCM) the ops
// are never selected (fluke_select_backend returns NULL) and compile to stubs.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)

#ifdef HAVE_CUDA
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream()
#elif defined(HAVE_ROCM)
#include <c10/hip/HIPStream.h>       // c10::hip::getCurrentHIPStream() — lightweight; the ATen/hip
                                     // context header drags in cusolver, and ATen/cuda pulls cuda_runtime_api.h
#endif

// The stream the fluke kernels launch on: torch's current stream for this device. During CUDA-graph
// capture torch sets the current stream to the capture stream, so the flstm/down-proj launches are
// captured automatically (matching how ATen's own ops behave).
static inline void *fluke_current_stream() {
#ifdef HAVE_CUDA
    return (void *)at::cuda::getCurrentCUDAStream().stream();
#else
    return (void *)c10::hip::getCurrentHIPStream().stream();
#endif
}

fluke_int8_backend_t *fluke_select_backend(int device_index, enum fluke_format_t desired, fluke_dims_t dims) {
    if (desired != FLUKE_FORMAT_INT8) return nullptr;      // only int8 kernels exist today
    return fluke_int8_select(device_index, dims);          // fluke's own handle; NULL on arch/dims mismatch
}

at::Tensor fluke_qkv_rotary_i8(const fluke_int8_backend_t *b, const tensor_quant_t &x, const tensor_quant_t &wqkv,
                               const at::Tensor &sin, const at::Tensor &cos) {
    const fluke_dims_t d = fluke_int8_dims(b);
    const int64_t N = x.tensor.size(0);
    const int64_t T = x.tensor.size(1);
    const int64_t M = N * T;
    auto a2d = x.tensor.reshape({M, d.d_model});
    auto out = torch::empty({M, 3 * d.d_model}, x.tensor.options().dtype(at::kHalf));

    // Runtime seqlen = T (rotary indexes seq = row % seqlen). All memrefs are contiguous.
    fluke_qkv_rotary_i8_gpu(
        b, out.data_ptr(), a2d.data_ptr(), wqkv.tensor.data_ptr(),
        x.scale.data_ptr(), wqkv.scale.data_ptr(), (const void *)sin.data_ptr(), (const void *)cos.data_ptr(),
        (int)M, (int)T);

    return out.view({N, T, 3, d.nhead, d.head_dim});
}

at::Tensor fluke_gated_mlp_i8(const fluke_int8_backend_t *b, const tensor_quant_t &x, const tensor_quant_t &gate,
                              const tensor_quant_t &up) {
    const fluke_dims_t d = fluke_int8_dims(b);
    const int64_t N = x.tensor.size(0);
    const int64_t T = x.tensor.size(1);
    const int64_t M = N * T;
    auto a2d = x.tensor.reshape({M, d.d_model});
    auto out = torch::empty({M, d.dim_feedforward}, x.tensor.options().dtype(at::kHalf));

    fluke_gated_mlp_i8_gpu(
        b, out.data_ptr(), a2d.data_ptr(), gate.tensor.data_ptr(), up.tensor.data_ptr(),
        x.scale.data_ptr(), gate.scale.data_ptr(), up.scale.data_ptr(), (int)M);

    return out.view({N, T, d.dim_feedforward});
}

#else // no GPU backend — ops never selected, so these are stubs.

fluke_int8_backend_t *fluke_select_backend(int, enum fluke_format_t, fluke_dims_t) { return nullptr; }

at::Tensor fluke_qkv_rotary_i8(const fluke_int8_backend_t *, const tensor_quant_t &, const tensor_quant_t &,
                               const at::Tensor &, const at::Tensor &) { return at::Tensor(); }

at::Tensor fluke_gated_mlp_i8(const fluke_int8_backend_t *, const tensor_quant_t &, const tensor_quant_t &,
                              const tensor_quant_t &) { return at::Tensor(); }

#endif // HAVE_CUDA || HAVE_ROCM
