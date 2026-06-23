#include "quant.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

thread_local bool g_quant_active = true;

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
        return {};
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
// Fake quantization — weights
// ---------------------------------------------------------------------------

static at::Tensor fake_quant_int8(const at::Tensor &W, bool per_channel, bool transposed) {
    auto W_f = W.to(torch::kFloat32);
    at::Tensor scale;
    if (per_channel) {
        int reduce_dim = transposed ? 0 : 1;
        scale = std::get<0>(W_f.abs().max(reduce_dim, /*keepdim=*/true)) / 127.f;
        scale.clamp_min_(1e-6f);
    } else {
        float amax = W_f.abs().max().item<float>();
        if (amax == 0.f) return W;
        scale = torch::full({1}, amax / 127.f, W_f.options());
    }
    return (W_f / scale).round().clamp_(-128.f, 127.f).mul_(scale).to(W.dtype());
}

static at::Tensor fake_quant_int4(const at::Tensor &W, bool per_channel, bool transposed) {
    auto W_f = W.to(torch::kFloat32);
    at::Tensor scale;
    if (per_channel) {
        int reduce_dim = transposed ? 0 : 1;
        scale = std::get<0>(W_f.abs().max(reduce_dim, /*keepdim=*/true)) / 7.f;
        scale.clamp_min_(1e-6f);
    } else {
        float amax = W_f.abs().max().item<float>();
        if (amax == 0.f) return W;
        scale = torch::full({1}, amax / 7.f, W_f.options());
    }
    return (W_f / scale).round().clamp_(-8.f, 7.f).mul_(scale).to(W.dtype());
}

static at::Tensor fake_quant_fp8e4m3(const at::Tensor &W, bool per_channel, bool transposed) {
    // FP8 E4M3FN: 4 exponent bits, 3 mantissa bits, max = 448.
    // Scales W into the E4M3FN range, casts through the native fp8 dtype to round to the
    // correct non-uniform grid, then scales back.
    static constexpr float fp8_max = 448.f;
    auto W_f = W.to(torch::kFloat32);
    at::Tensor scale;
    if (per_channel) {
        int reduce_dim = transposed ? 0 : 1;
        scale = std::get<0>(W_f.abs().max(reduce_dim, /*keepdim=*/true)) / fp8_max;
        scale.clamp_min_(1e-6f);
    } else {
        float amax = W_f.abs().max().item<float>();
        if (amax == 0.f) return W;
        scale = torch::full({1}, amax / fp8_max, W_f.options());
    }
    auto W_scaled = (W_f / scale).clamp_(-fp8_max, fp8_max);
    return W_scaled.to(torch::kFloat8_e4m3fn).to(torch::kFloat32).mul_(scale).to(W.dtype());
}

// Apply FP4 E2M1 rounding to a float32 tensor that has already been divided by its scale.
// Positive representable values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
// Breakpoints are midpoints between adjacent values.
static at::Tensor apply_fp4_grid(const at::Tensor &x_scaled) {
    auto bp  = torch::tensor({0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f}, x_scaled.options());
    auto grid = torch::tensor({0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f},   x_scaled.options());
    auto sign = x_scaled.sign();
    auto abs_ = x_scaled.abs();
    // bucketize: for each element returns index i s.t. bp[i-1] <= val < bp[i], clamped to [0,7]
    auto idx = torch::bucketize(abs_, bp).clamp_(0, 7);
    return grid.index({idx.flatten()}).reshape_as(abs_) * sign;
}

static at::Tensor fake_quant_fp4(const at::Tensor &W, bool per_channel, bool transposed) {
    // FP4 E2M1: max representable = 6.0, 8 distinct absolute values.
    static constexpr float fp4_max = 6.f;
    auto W_f = W.to(torch::kFloat32);
    at::Tensor scale;
    if (per_channel) {
        int reduce_dim = transposed ? 0 : 1;
        scale = std::get<0>(W_f.abs().max(reduce_dim, /*keepdim=*/true)) / fp4_max;
        scale.clamp_min_(1e-6f);
    } else {
        float amax = W_f.abs().max().item<float>();
        if (amax == 0.f) return W;
        scale = torch::full({1}, amax / fp4_max, W_f.options());
    }
    return apply_fp4_grid(W_f / scale).mul_(scale).to(W.dtype());
}

at::Tensor maybe_fake_quant(const at::Tensor &W, const std::string &method, bool transposed) {
    if (!g_quant_active || method.empty() || method == "dummy" || method == "fp16") {
        return W;
    }
    // Calibrated fixed scale: "int8_s<scale>" where <scale> is a pre-computed float value.
    if (method.rfind("int8_s", 0) == 0) {
        float scale = std::stof(method.c_str() + 6);
        auto W_f = W.to(torch::kFloat32);
        return (W_f / scale).round().clamp_(-128.f, 127.f).mul_(scale).to(W.dtype());
    }
    bool per_channel = method.find("per_channel") != std::string::npos;
    if (method.find("fp8")  != std::string::npos) return fake_quant_fp8e4m3(W, per_channel, transposed);
    if (method.find("fp4")  != std::string::npos) return fake_quant_fp4(W, per_channel, transposed);
    if (method.find("int4") != std::string::npos) return fake_quant_int4(W, per_channel, transposed);
    return fake_quant_int8(W, per_channel, transposed);
}

// ---------------------------------------------------------------------------
// Fake quantization — activations
// ---------------------------------------------------------------------------

static at::Tensor fake_quant_int8_act(const at::Tensor &x, bool per_token) {
    auto orig_shape = x.sizes().vec();
    auto x_f = x.reshape({-1, x.size(-1)}).to(torch::kFloat32);
    at::Tensor scale;
    if (per_token) {
        scale = std::get<0>(x_f.abs().max(1, /*keepdim=*/true)) / 127.f;
        scale.clamp_min_(1e-6f);
    } else {
        float amax = x_f.abs().max().item<float>();
        if (amax == 0.f) return x;
        scale = torch::full({1}, amax / 127.f, x_f.options());
    }
    return (x_f / scale).round().clamp_(-128.f, 127.f).mul_(scale).reshape(orig_shape).to(x.dtype());
}

static at::Tensor fake_quant_int4_act(const at::Tensor &x, bool per_token) {
    auto orig_shape = x.sizes().vec();
    auto x_f = x.reshape({-1, x.size(-1)}).to(torch::kFloat32);
    at::Tensor scale;
    if (per_token) {
        scale = std::get<0>(x_f.abs().max(1, /*keepdim=*/true)) / 7.f;
        scale.clamp_min_(1e-6f);
    } else {
        float amax = x_f.abs().max().item<float>();
        if (amax == 0.f) return x;
        scale = torch::full({1}, amax / 7.f, x_f.options());
    }
    return (x_f / scale).round().clamp_(-8.f, 7.f).mul_(scale).reshape(orig_shape).to(x.dtype());
}

static at::Tensor fake_quant_fp8e4m3_act(const at::Tensor &x, bool per_token) {
    static constexpr float fp8_max = 448.f;
    auto orig_shape = x.sizes().vec();
    auto x_f = x.reshape({-1, x.size(-1)}).to(torch::kFloat32);
    at::Tensor scale;
    if (per_token) {
        scale = std::get<0>(x_f.abs().max(1, /*keepdim=*/true)) / fp8_max;
        scale.clamp_min_(1e-6f);
    } else {
        float amax = x_f.abs().max().item<float>();
        if (amax == 0.f) return x;
        scale = torch::full({1}, amax / fp8_max, x_f.options());
    }
    auto x_scaled = (x_f / scale).clamp_(-fp8_max, fp8_max);
    return x_scaled.to(torch::kFloat8_e4m3fn).to(torch::kFloat32).mul_(scale).reshape(orig_shape).to(x.dtype());
}

static at::Tensor fake_quant_fp4_act(const at::Tensor &x, bool per_token) {
    static constexpr float fp4_max = 6.f;
    auto orig_shape = x.sizes().vec();
    auto x_f = x.reshape({-1, x.size(-1)}).to(torch::kFloat32);
    at::Tensor scale;
    if (per_token) {
        scale = std::get<0>(x_f.abs().max(1, /*keepdim=*/true)) / fp4_max;
        scale.clamp_min_(1e-6f);
    } else {
        float amax = x_f.abs().max().item<float>();
        if (amax == 0.f) return x;
        scale = torch::full({1}, amax / fp4_max, x_f.options());
    }
    return apply_fp4_grid(x_f / scale).mul_(scale).reshape(orig_shape).to(x.dtype());
}

at::Tensor maybe_fake_quant_act(const at::Tensor &x, const std::string &method) {
    if (!g_quant_active || method.empty() || method == "dummy" || method == "fp16") {
        return x;
    }
    // Calibrated fixed scale: "int8_s<scale>" where <scale> is a pre-computed float value.
    if (method.rfind("int8_s", 0) == 0) {
        float scale = std::stof(method.c_str() + 6);
        auto orig_shape = x.sizes().vec();
        auto x_f = x.reshape({-1, x.size(-1)}).to(torch::kFloat32);
        return (x_f / scale).round().clamp_(-128.f, 127.f).mul_(scale).reshape(orig_shape).to(x.dtype());
    }
    // Fixed-scale variants for activations with known bounded range (e.g. FLSTM hh in [-1,1]).
    if (method == "int8_fixed") {
        constexpr float scale = 1.f / 127.f;
        auto x_f = x.to(torch::kFloat32);
        return (x_f / scale).round().clamp_(-128.f, 127.f).mul_(scale).to(x.dtype());
    }
    if (method == "int4_fixed") {
        constexpr float scale = 1.f / 7.f;
        auto x_f = x.to(torch::kFloat32);
        return (x_f / scale).round().clamp_(-8.f, 7.f).mul_(scale).to(x.dtype());
    }
    if (method == "fp4_fixed") {
        // FP4 E2M1 fixed scale: assumes input in [-1, 1], maps to [-6, 6] grid.
        constexpr float scale = 1.f / 6.f;
        auto x_f = x.to(torch::kFloat32);
        return apply_fp4_grid(x_f / scale).mul_(scale).to(x.dtype());
    }
    if (method == "int8_fixed_4") {
        // Fixed scale covering ±4σ for approximately unit-variance (post-RMSNorm) activations.
        constexpr float scale = 4.f / 127.f;
        auto x_f = x.to(torch::kFloat32);
        return (x_f / scale).round().clamp_(-128.f, 127.f).mul_(scale).to(x.dtype());
    }
    bool per_token = method.find("per_channel") != std::string::npos;
    if (method.find("fp8")  != std::string::npos) return fake_quant_fp8e4m3_act(x, per_token);
    if (method.find("fp4")  != std::string::npos) return fake_quant_fp4_act(x, per_token);
    if (method.find("int4") != std::string::npos) return fake_quant_int4_act(x, per_token);
    return fake_quant_int8_act(x, per_token);
}
