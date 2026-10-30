#pragma once

// Fake (simulated) quantisation of a tensor along its last (reduction) dim. Formats are listed in
// TxModel.cpp; also used standalone by scripts/fq_check.cpp (driven by scripts/fq_check.py).
//
// The libraries below are Python-only, so this is a libtorch port of the same arithmetic. scripts/fq_check.py
// verifies it matches them bit-for-bit (round half to even) at the commits linked here.
//
// MX formats follow the OCP MX v1.0 scheme as implemented in Microsoft microxcaling (commit 7bc4195):
//   shared scale 2^(floor(log2(block absmax)) - emax_elem), clamped to the e8m0 range:
//     https://github.com/microsoft/microxcaling/blob/7bc41952de394f5cc5e782baf132e7c7542eb4e4/mx/mx_ops.py#L49  (_shared_exponents)
//     https://github.com/microsoft/microxcaling/blob/7bc41952de394f5cc5e782baf132e7c7542eb4e4/mx/mx_ops.py#L173 (_quantize_mx)
//   per-element exponent clipped to emin (subnormals), mantissa round, saturating clamp:
//     https://github.com/microsoft/microxcaling/blob/7bc41952de394f5cc5e782baf132e7c7542eb4e4/mx/elemwise_ops.py#L82 (_quantize_elemwise_core)
//   element emax / max values; int8 is sign-magnitude 1.6 fixed point, so mxint8 is symmetric (+-127/64):
//     https://github.com/microsoft/microxcaling/blob/7bc41952de394f5cc5e782baf132e7c7542eb4e4/mx/formats.py#L61  (_get_format_params)
//   Note microxcaling's default rounding is "nearest" (ties away from zero); we use round half to even.
//
// AMD TensorCast (commit c4d9b1d) implements the same MX scheme and also the per-row float-scaled formats
// (scale = row absmax / maxint or maxfloat), matching our int8/int4/fp8 path:
//     https://github.com/ROCm/tensorcast/blob/c4d9b1d2f0d3590f236635ae6756430a17df4820/tcast/cast.py#L63  (_get_scales)
//     https://github.com/ROCm/tensorcast/blob/c4d9b1d2f0d3590f236635ae6756430a17df4820/tcast/cast.py#L105 (_apply_scales)
//     https://github.com/ROCm/tensorcast/blob/c4d9b1d2f0d3590f236635ae6756430a17df4820/tcast/number.py#L320 (ints are sign-magnitude, minint = -maxint)
//
// Where libtorch has the format natively we use it instead: at::fake_quantize_per_channel_affine for
// int8/int4, and round trips through kFloat8_e4m3fn / kFloat8_e5m2 for fp8 elements.

#include <ATen/ATen.h>

#include <string>

// Round to a sign-magnitude minifloat grid (RNE, saturating, with subnormals).
static at::Tensor round_minifloat(const at::Tensor &x, int emin, int mbits, float max) {
    auto step = at::exp2(at::floor(at::log2(x.abs())).clamp_min_(emin).sub_(mbits));
    return at::round(x / step).mul_(step).clamp_(-max, max);
}

static at::Tensor fake_quant(const at::Tensor &x, const std::string &f) {
    const int64_t K = x.size(-1);
    auto x2 = x.reshape({-1, K}).to(at::kFloat);
    at::Tensor y;
    if (f == "int8" || f == "int4") {
        const int qmax = f == "int8" ? 127 : 7;
        auto scale = std::get<0>(x2.abs().max(1)).div_(qmax).clamp_min_(1e-12f);
        auto zp = at::zeros(scale.sizes(), scale.options().dtype(at::kInt));
        y = at::fake_quantize_per_channel_affine(x2, scale, zp, 0, -qmax, qmax);
    } else if (f == "fp8_e4m3" || f == "fp8_e5m2") {
        const bool e4m3 = f == "fp8_e4m3";
        auto scale = std::get<0>(x2.abs().max(1, true)).div_(e4m3 ? 448.f : 57344.f).clamp_min_(1e-12f);
        y = (x2 / scale).to(e4m3 ? at::kFloat8_e4m3fn : at::kFloat8_e5m2).to(at::kFloat).mul_(scale);
    } else {
        TORCH_CHECK(K % 32 == 0, "MX fake quant needs reduction dim divisible by 32, got ", K);
        // emax of the element format; shared scale = 2^(floor(log2(block absmax)) - emax)
        const int emax = f == "mxint8" ? 0 : f == "mxfp8_e4m3" ? 8 : f == "mxfp8_e5m2" ? 15 : f == "mxfp6_e3m2" ? 4 : 2;
        auto xb = x2.view({-1, K / 32, 32});
        auto amax = std::get<0>(xb.abs().max(-1, true));
        auto scale = at::exp2(at::floor(at::log2(amax)).sub_(emax).clamp_(-127, 127));
        auto v = xb / scale;
        if (f == "mxint8")          v = v.mul_(64).round_().clamp_(-127, 127).div_(64);  // 1.6 fixed point, symmetric
        else if (f == "mxfp8_e4m3") v = v.clamp_(-448, 448).to(at::kFloat8_e4m3fn).to(at::kFloat);
        else if (f == "mxfp8_e5m2") v = v.clamp_(-57344, 57344).to(at::kFloat8_e5m2).to(at::kFloat);
        else if (f == "mxfp6_e2m3") v = round_minifloat(v, 0, 3, 7.5f);
        else if (f == "mxfp6_e3m2") v = round_minifloat(v, -2, 2, 28.f);
        else                        v = round_minifloat(v, 0, 1, 6.f);  // mxfp4 (e2m1)
        y = v.mul_(scale).view({-1, K});
    }
    return y.view(x.sizes()).to(x.dtype());
}
