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
// Fake quantization
// ---------------------------------------------------------------------------

static at::Tensor fake_quant_int8(const at::Tensor &W, bool per_channel, bool transposed) {
    auto W_f = W.to(torch::kFloat32);
    at::Tensor scale;
    if (per_channel) {
        // Reduce over the in-features dimension to get one scale per output channel.
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

static at::Tensor fake_quant_fp8(const at::Tensor &W, bool per_channel, bool transposed) {
    // Software simulation of FP8 E4M3FN: max representable = 448, 3 mantissa bits.
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
    // Round mantissa to 3 bits: scale to [0, 8], round, scale back.
    auto W_s = W_f / scale;
    auto W_q = (W_s.abs() * 8.f).round().div_(8.f) * W_s.sign();
    return W_q.clamp_(-fp8_max, fp8_max).mul_(scale).to(W.dtype());
}

at::Tensor maybe_fake_quant(const at::Tensor &W, const std::string &method, bool transposed) {
    if (!g_quant_active || method.empty() || method == "dummy" || method == "fp16") {
        return W;
    }
    bool per_channel = method.find("per_channel") != std::string::npos;
    bool fp8 = method.find("fp8") != std::string::npos;
    if (fp8) return fake_quant_fp8(W, per_channel, transposed);
    return fake_quant_int8(W, per_channel, transposed);
}
