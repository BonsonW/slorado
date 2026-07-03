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

#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream()

// The stream the fluke kernels launch on: torch's current stream for this device. During CUDA-graph
// capture torch sets the current stream to the capture stream, so the flstm/down-proj launches are
// captured automatically (matching how ATen's own ops behave).
static inline void *fluke_current_stream() {
    return (void *)at::cuda::getCurrentCUDAStream().stream();
}

// slorado-side backend: the dims we selected for + fluke's opaque kernel handle. Process-lifetime.
struct fluke_backend {
    fluke_dims_t dims;
    const fluke_int8_backend_t *h;
};

fluke_backend_t *fluke_select_backend(int device_index, enum fluke_format_t desired, fluke_dims_t dims) {
    if (desired != FLUKE_FORMAT_INT8) return nullptr;  // only int8 kernels exist today
    const fluke_int8_backend_t *h = fluke_int8_select(device_index, dims);  // NULL if arch/dims mismatch
    if (!h) return nullptr;
    static fluke_backend b;  // process-lifetime; all layers share it
    b.dims = dims;
    b.h = h;
    return &b;
}

at::Tensor fluke_qkv_rotary_i8(const fluke_backend_t *b, const tensor_quant_t &x, const tensor_quant_t &wqkv,
                               const at::Tensor &sin, const at::Tensor &cos) {
    const int64_t N = x.tensor.size(0);
    const int64_t T = x.tensor.size(1);
    const int64_t M = N * T;
    auto a2d = x.tensor.reshape({M, b->dims.d_model});
    auto out = torch::empty({M, 3 * b->dims.d_model}, x.tensor.options().dtype(at::kHalf));

    // Runtime seqlen = T (rotary indexes seq = row % seqlen). All memrefs are contiguous.
    fluke_qkv_rotary_i8_gpu(
        b->h, out.data_ptr(), a2d.data_ptr(), wqkv.tensor.data_ptr(),
        x.scale.data_ptr(), wqkv.scale.data_ptr(), (const void *)sin.data_ptr(), (const void *)cos.data_ptr(),
        (int)M, (int)T);

    return out.view({N, T, 3, b->dims.nhead, b->dims.head_dim});
}

at::Tensor fluke_gated_mlp_i8(const fluke_backend_t *b, const tensor_quant_t &x, const tensor_quant_t &gate,
                              const tensor_quant_t &up) {
    const int64_t N = x.tensor.size(0);
    const int64_t T = x.tensor.size(1);
    const int64_t M = N * T;
    auto a2d = x.tensor.reshape({M, b->dims.d_model});
    auto out = torch::empty({M, b->dims.dim_feedforward}, x.tensor.options().dtype(at::kHalf));

    fluke_gated_mlp_i8_gpu(
        b->h, out.data_ptr(), a2d.data_ptr(), gate.tensor.data_ptr(), up.tensor.data_ptr(),
        x.scale.data_ptr(), gate.scale.data_ptr(), up.scale.data_ptr(), (int)M);

    return out.view({N, T, b->dims.dim_feedforward});
}

// slorado-side FLSTM backend: fluke's opaque kernel handle. Process-lifetime.
struct fluke_flstm_wrap {
    fluke_flstm_backend_t *h;
};

fluke_flstm_wrap_t *fluke_select_flstm(int device_index, enum fluke_format_t desired, int H, int K_hh, int R) {
    if (desired != FLUKE_FORMAT_INT8) return nullptr;  // only int8 kernels exist today
    fluke_flstm_backend_t *h = fluke_flstm_select(device_index, H, K_hh, R);  // NULL if arch/shape mismatch
    if (!h) return nullptr;
    static fluke_flstm_wrap b;  // process-lifetime; all layers share it (same shape)
    b.h = h;
    return &b;
}

at::Tensor fluke_flstm_down_proj_i8(const fluke_flstm_wrap_t *b, const at::Tensor &a_i8,
                                    const at::Tensor &scale_a, const tensor_quant_t &w) {
    const int64_t M = a_i8.size(0);
    const int64_t R = w.tensor.size(0);
    auto out = torch::empty({M, R}, a_i8.options().dtype(at::kHalf));
    fluke_down_proj_i8_gpu(b->h, out.data_ptr(), a_i8.data_ptr(), w.tensor.data_ptr(),
                           scale_a.data_ptr(), w.scale.data_ptr(), (int)M, fluke_current_stream());
    return out;
}

void fluke_flstm_down_proj_i8_into(const fluke_flstm_wrap_t *b, at::Tensor &out, const at::Tensor &a_i8,
                                   const at::Tensor &scale_a, const tensor_quant_t &w) {
    const int64_t M = a_i8.size(0);
    fluke_down_proj_i8_gpu(b->h, out.data_ptr(), a_i8.data_ptr(), w.tensor.data_ptr(),
                           scale_a.data_ptr(), w.scale.data_ptr(), (int)M, fluke_current_stream());
}

at::Tensor fluke_dequant_int8_transpose(const at::Tensor &in_tnc, float scale) {
    const int64_t T = in_tnc.size(0), N = in_tnc.size(1), C = in_tnc.size(2);
    auto in = in_tnc.contiguous();
    auto out = torch::empty({N, T, C}, in.options().dtype(at::kHalf));
    fluke_dequant_int8_transpose_gpu(in.data_ptr(), out.data_ptr(), (int)T, (int)N, (int)C, scale);
    return out;
}

void fluke_flstm_step_i8(const fluke_flstm_wrap_t *b, at::Tensor &h_i8, at::Tensor &c_f32,
                         const at::Tensor &a_f16, const at::Tensor gate_w[4], const at::Tensor gate_b[4]) {
    const int64_t B = a_f16.size(0);
    fluke_flstm_step_i8_gpu(
        b->h, h_i8.data_ptr(), a_f16.data_ptr(),
        gate_w[0].data_ptr(), gate_w[1].data_ptr(), gate_w[2].data_ptr(), gate_w[3].data_ptr(),
        gate_b[0].data_ptr(), gate_b[1].data_ptr(), gate_b[2].data_ptr(), gate_b[3].data_ptr(),
        c_f32.data_ptr(), (int)B, fluke_current_stream());
}

#else // no GPU backend — ops never selected, so these are stubs.

fluke_backend_t *fluke_select_backend(int, enum fluke_format_t, fluke_dims_t) { return nullptr; }

at::Tensor fluke_qkv_rotary_i8(const fluke_backend_t *, const tensor_quant_t &, const tensor_quant_t &,
                               const at::Tensor &, const at::Tensor &) { return at::Tensor(); }

at::Tensor fluke_gated_mlp_i8(const fluke_backend_t *, const tensor_quant_t &, const tensor_quant_t &,
                              const tensor_quant_t &) { return at::Tensor(); }

fluke_flstm_wrap_t *fluke_select_flstm(int, enum fluke_format_t, int, int, int) { return nullptr; }

at::Tensor fluke_flstm_down_proj_i8(const fluke_flstm_wrap_t *, const at::Tensor &, const at::Tensor &,
                                    const tensor_quant_t &) { return at::Tensor(); }

void fluke_flstm_down_proj_i8_into(const fluke_flstm_wrap_t *, at::Tensor &, const at::Tensor &,
                                   const at::Tensor &, const tensor_quant_t &) {}

at::Tensor fluke_dequant_int8_transpose(const at::Tensor &, float) { return at::Tensor(); }

void fluke_flstm_step_i8(const fluke_flstm_wrap_t *, at::Tensor &, at::Tensor &, const at::Tensor &,
                         const at::Tensor[4], const at::Tensor[4]) {}

#endif // HAVE_CUDA || HAVE_ROCM
