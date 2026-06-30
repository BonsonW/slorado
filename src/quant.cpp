#include "quant.h"

#include <cstdio>
#include <torch/version.h>
#include <cstring>
#include <fstream>
#include <sstream>

thread_local bool g_quant_active = true;

void build_quant_methods(std::unordered_map<std::string, layer_quant_t> &out,
                         const std::unordered_map<std::string, std::string> &cfg) {
    out.reserve(cfg.size());
    for (const auto &kv : cfg) {
        const auto &key = kv.first;
        if (key.size() > 4 && key.compare(key.size() - 4, 4, ".act") == 0) {
            out[key.substr(0, key.size() - 4)].act = kv.second;
        } else {
            out[key].weight = kv.second;
        }
    }
    for (auto &kv : out) {
        if (kv.second.act.empty()) kv.second.act = kv.second.weight;
    }
}

// ---------------------------------------------------------------------------
// Minimal JSON parser — extract top-level string-to-string key/value pairs.
// Handles the simple format produced by Python's json.dump():
//   { "key1": "val1", "key2": "val2", ... }
// ---------------------------------------------------------------------------
static std::unordered_map<std::string, std::string> parse_simple_json(const std::string &text) {
    std::unordered_map<std::string, std::string> out;
    const char *p = text.c_str();
    auto skip_ws = [&]() { while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; };
    auto read_str = [&]() -> std::string {
        if (*p != '"') return "";
        ++p;
        std::string s;
        while (*p && *p != '"') {
            if (*p == '\\') ++p;
            s += *p++;
        }
        if (*p == '"') ++p;
        return s;
    };

    skip_ws();
    if (*p == '{') ++p;
    while (*p) {
        skip_ws();
        if (*p == '}' || *p == '\0') break;
        if (*p == ',') { ++p; continue; }
        auto key = read_str();
        skip_ws();
        if (*p == ':') ++p;
        skip_ws();
        auto val = read_str();
        if (!key.empty()) out[key] = val;
    }
    return out;
}

std::unordered_map<std::string, std::string> load_quant_config(const std::string &path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[quant] error: cannot open quant config %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    auto cfg = parse_simple_json(ss.str());
    fprintf(stderr, "[quant] loaded quant config from %s (%zu layers)\n", path.c_str(), cfg.size());
    for (const auto &kv : cfg) {
        fprintf(stderr, "[quant]   %s: %s\n", kv.first.c_str(), kv.second.c_str());
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Grid rounding helpers — operate on float32 tensors already divided by their scale.
// ---------------------------------------------------------------------------

// Apply FP4 E2M1 rounding. Positive representable values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
static at::Tensor apply_fp4_grid(const at::Tensor &x_scaled) {
    auto bp   = torch::tensor({0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f}, x_scaled.options());
    auto grid = torch::tensor({0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f},    x_scaled.options());
    auto sign = x_scaled.sign();
    auto abs_ = x_scaled.abs();
    auto idx  = torch::bucketize(abs_, bp).clamp_(0, 7);
    return grid.index({idx.flatten()}).reshape_as(abs_) * sign;
}

// Apply INT8 rounding.
static at::Tensor apply_int8_grid(const at::Tensor &x_scaled) {
    return x_scaled.round().clamp_(-128.f, 127.f);
}

// Apply MXFP6 E2M3 rounding. 1 sign + 2 exp (bias=1) + 3 mantissa bits; max = 7.5.
static at::Tensor apply_fp6e2m3_grid(const at::Tensor &x_scaled) {
    static constexpr float fp6_max = 7.5f;
    auto sign = x_scaled.sign();
    auto abs_ = x_scaled.abs().clamp_max_(fp6_max);
    auto q = torch::where(abs_ < 2.f,
                          (abs_ * 8.f).round() / 8.f,
             torch::where(abs_ < 4.f,
                          (abs_ * 4.f).round() / 4.f,
                          (abs_ * 2.f).round() / 2.f)).clamp_max_(fp6_max);
    return q * sign;
}

// Apply FP8 E4M3FN rounding.
static at::Tensor apply_fp8e4m3_grid(const at::Tensor &x_scaled) {
    static constexpr float fp8_max = 448.f;
    auto x = x_scaled.clamp(-fp8_max, fp8_max);
#if (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 1) || TORCH_VERSION_MAJOR >= 3
    return x.to(torch::kFloat8_e4m3fn).to(torch::kFloat32);
#else
    // torch < 2.1: kFloat8_e4m3fn dtype unavailable — simulate via ATen ops.
    static constexpr float norm_min = 1.f / 64.f;   // 2^-6, smallest normal
    static constexpr float sub_step = 1.f / 512.f;  // 2^-9, subnormal step
    auto a    = x.abs();
    auto step = torch::where(
        a.ge(norm_min),
        (a.clamp_min(1e-38f).log2().floor() - 3.f).exp2(),
        torch::full_like(a, sub_step));
    return x.sign() * ((a / step).round() * step).clamp_max_(fp8_max);
#endif
}

// ---------------------------------------------------------------------------
// 2D quantization helpers — operate on [rows, cols] float32 tensors.
// per_row=true: one scale per row (per-channel for weights, per-token for activations).
// per_row=false: one global scale.
// ---------------------------------------------------------------------------

static at::Tensor quant_int8_2d(const at::Tensor &x, bool per_row) {
    at::Tensor scale;
    if (per_row) {
        scale = std::get<0>(x.abs().max(1, /*keepdim=*/true)) / 127.f;
        scale.clamp_min_(1e-6f);
    } else {
        float amax = x.abs().max().item<float>();
        if (amax == 0.f) return x;
        scale = torch::full({1}, amax / 127.f, x.options());
    }
    return (x / scale).round().clamp_(-128.f, 127.f).mul_(scale);
}

static at::Tensor quant_int4_2d(const at::Tensor &x, bool per_row) {
    at::Tensor scale;
    if (per_row) {
        scale = std::get<0>(x.abs().max(1, /*keepdim=*/true)) / 7.f;
        scale.clamp_min_(1e-6f);
    } else {
        float amax = x.abs().max().item<float>();
        if (amax == 0.f) return x;
        scale = torch::full({1}, amax / 7.f, x.options());
    }
    return (x / scale).round().clamp_(-8.f, 7.f).mul_(scale);
}

static at::Tensor quant_fp8e4m3_2d(const at::Tensor &x, bool per_row) {
    // FP8 E4M3FN: 4 exponent bits, 3 mantissa bits, max = 448.
    static constexpr float fp8_max = 448.f;
    at::Tensor scale;
    if (per_row) {
        scale = std::get<0>(x.abs().max(1, /*keepdim=*/true)) / fp8_max;
        scale.clamp_min_(1e-6f);
    } else {
        float amax = x.abs().max().item<float>();
        if (amax == 0.f) return x;
        scale = torch::full({1}, amax / fp8_max, x.options());
    }
    return apply_fp8e4m3_grid(x / scale).mul_(scale);
}

static at::Tensor quant_fp4_2d(const at::Tensor &x, bool per_row) {
    // FP4 E2M1: max representable = 6.0.
    static constexpr float fp4_max = 6.f;
    at::Tensor scale;
    if (per_row) {
        scale = std::get<0>(x.abs().max(1, /*keepdim=*/true)) / fp4_max;
        scale.clamp_min_(1e-6f);
    } else {
        float amax = x.abs().max().item<float>();
        if (amax == 0.f) return x;
        scale = torch::full({1}, amax / fp4_max, x.options());
    }
    return apply_fp4_grid(x / scale).mul_(scale);
}

// ---------------------------------------------------------------------------
// Group-32 microscaling (MX) helpers
// ---------------------------------------------------------------------------

typedef at::Tensor(*GridFn)(const at::Tensor&);

// Fake-quantize a 2D weight [out, in] (or [in, out] when transposed) with one scale per 32
// input-channel elements.  po2_scale=true snaps scales to the nearest power-of-2 (E8M0), as
// required by the OCP MXINT8 spec.
static at::Tensor group32_weight(const at::Tensor &W, bool transposed, float qmax, GridFn grid_fn,
                                  bool po2_scale = false) {
    static constexpr int64_t G = 32;
    auto W_f = W.to(torch::kFloat32);
    int64_t n_quant = transposed ? W_f.size(0) : W_f.size(1);  // input channels
    int64_t n_other = transposed ? W_f.size(1) : W_f.size(0);  // output channels

    auto W_2d = (transposed ? W_f.t() : W_f).contiguous();  // [n_other, n_quant]

    int64_t n_groups = (n_quant + G - 1) / G;
    int64_t padded   = n_groups * G;
    if (padded != n_quant) {
        W_2d = at::constant_pad_nd(W_2d, {0, padded - n_quant});
    }
    auto W_g   = W_2d.reshape({n_other, n_groups, G});
    auto scale = std::get<0>(W_g.abs().max(-1, true)) / qmax;
    if (po2_scale) {
        scale = torch::exp2(torch::ceil(torch::log2(scale.clamp_min(1e-38f))));
    }
    scale.clamp_min_(1e-6f);
    auto W_q = grid_fn(W_g / scale).mul_(scale).reshape({n_other, padded});
    if (padded != n_quant) {
        W_q = W_q.slice(1, 0, n_quant);
    }
    if (transposed) {
        W_q = W_q.t().contiguous();
    }
    return W_q.to(W.dtype());
}

// Fake-quantize activation tensor [..., features] with one scale per 32 features.
// po2_scale=true uses E8M0 power-of-2 scales (MXINT8).
static at::Tensor group32_act(const at::Tensor &x, float qmax, GridFn grid_fn,
                               bool po2_scale = false) {
    static constexpr int64_t G = 32;
    auto orig_shape = x.sizes().vec();
    auto x_f = x.reshape({-1, x.size(-1)}).to(torch::kFloat32).contiguous();
    int64_t rows  = x_f.size(0);
    int64_t feats = x_f.size(1);

    int64_t n_groups = (feats + G - 1) / G;
    int64_t padded   = n_groups * G;
    if (padded != feats) {
        x_f = at::constant_pad_nd(x_f, {0, padded - feats});
    }
    auto x_g   = x_f.reshape({rows, n_groups, G});
    auto scale = std::get<0>(x_g.abs().max(-1, true)) / qmax;
    if (po2_scale) {
        scale = torch::exp2(torch::ceil(torch::log2(scale.clamp_min(1e-38f))));
    }
    scale.clamp_min_(1e-6f);
    auto x_q = grid_fn(x_g / scale).mul_(scale).reshape({rows, padded});
    if (padded != feats) {
        x_q = x_q.slice(1, 0, feats);
    }
    return x_q.reshape(orig_shape).to(x.dtype());
}

at::Tensor fake_quant(const at::Tensor &x, const std::string &method, bool transposed) {
    if (!g_quant_active || method.empty() || method == "dummy" || method == "fp16")
        return x;

    const bool is_weight = (x.dim() == 2);

    // Fixed-scale methods: element-wise, no per-row scaling needed.
    if (method.rfind("int8_s", 0) == 0) {
        // Calibrated fixed scale embedded in method string.
        float scale = std::stof(method.c_str() + 6);
        auto orig = x.sizes().vec();
        auto xf = x.reshape({-1, x.size(-1)}).to(torch::kFloat32);
        return (xf / scale).round().clamp_(-128.f, 127.f).mul_(scale).reshape(orig).to(x.dtype());
    }
    if (method == "int8_fixed") {
        constexpr float scale = 1.f / 127.f;
        auto xf = x.to(torch::kFloat32);
        return (xf / scale).round().clamp_(-128.f, 127.f).mul_(scale).to(x.dtype());
    }
    if (method == "int4_fixed") {
        constexpr float scale = 1.f / 7.f;
        auto xf = x.to(torch::kFloat32);
        return (xf / scale).round().clamp_(-8.f, 7.f).mul_(scale).to(x.dtype());
    }
    if (method == "fp8_fixed") {
        // Assumes input in [-1, 1]; maps to the full e4m3 grid.
        constexpr float scale = 1.f / 448.f;
        auto xf = x.to(torch::kFloat32);
        return apply_fp8e4m3_grid(xf / scale).mul_(scale).to(x.dtype());
    }
    if (method == "fp4_fixed") {
        // Assumes input in [-1, 1]; maps to the FP4 E2M1 grid.
        constexpr float scale = 1.f / 6.f;
        auto xf = x.to(torch::kFloat32);
        return apply_fp4_grid(xf / scale).mul_(scale).to(x.dtype());
    }
    if (method == "int8_fixed_4") {
        // ±4σ fixed scale for unit-variance (post-RMSNorm) activations.
        constexpr float scale = 4.f / 127.f;
        auto xf = x.to(torch::kFloat32);
        return (xf / scale).round().clamp_(-128.f, 127.f).mul_(scale).to(x.dtype());
    }

    // MX group-32 — check before substring matches ("mxfp4" contains "fp4", etc.)
    if (method == "mxint8") return is_weight ? group32_weight(x, transposed, 127.f, apply_int8_grid, /*po2=*/true)
                                             : group32_act(x, 127.f, apply_int8_grid, /*po2=*/true);
    if (method == "mxfp4")  return is_weight ? group32_weight(x, transposed, 6.f,   apply_fp4_grid)
                                             : group32_act(x, 6.f,   apply_fp4_grid);
    if (method == "mxfp6")  return is_weight ? group32_weight(x, transposed, 7.5f,  apply_fp6e2m3_grid)
                                             : group32_act(x, 7.5f,  apply_fp6e2m3_grid);
    if (method == "mxfp8")  return is_weight ? group32_weight(x, transposed, 448.f, apply_fp8e4m3_grid)
                                             : group32_act(x, 448.f, apply_fp8e4m3_grid);

    // Dynamic-scale quantization: normalise to [rows, cols], quantize, restore shape.
    // Weights: [out, in] or [in, out] when transposed — transpose so output channels are rows.
    // Activations: [..., T, C] — flatten leading dims so tokens are rows.
    auto orig_shape = x.sizes().vec();
    at::Tensor x2d;
    if (is_weight && transposed) {
        x2d = x.t().contiguous().to(torch::kFloat32);
    } else if (is_weight) {
        x2d = x.to(torch::kFloat32);
    } else {
        x2d = x.reshape({-1, x.size(-1)}).to(torch::kFloat32);
    }

    // "per_channel" in method → per-row scaling (per output-channel for weights, per-token for acts).
    bool per_row = method.find("per_channel") != std::string::npos;
    at::Tensor result;
    if      (method.find("fp8")  != std::string::npos) result = quant_fp8e4m3_2d(x2d, per_row);
    else if (method.find("fp4")  != std::string::npos) result = quant_fp4_2d(x2d, per_row);
    else if (method.find("int4") != std::string::npos) result = quant_int4_2d(x2d, per_row);
    else                                               result = quant_int8_2d(x2d, per_row);

    if (is_weight && transposed) result = result.t().contiguous();
    return result.reshape(orig_shape).to(x.dtype());
}
