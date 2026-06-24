#pragma once

#include <string>
#include <unordered_map>

#include <torch/torch.h>

// Quantization methods for a single linear layer.
// Both fields are empty strings by default — fake_quant treats empty as fp16 pass-through.
struct layer_quant_t {
    std::string weight;
    std::string act;
};

// Thread-local flag: set to false to bypass quant paths (used for the fp16 baseline pass
// in sensitivity mode). True by default — quant layers run normally.
extern thread_local bool g_quant_active;

// Load a per-layer quantization config from a JSON file.
// Returns a map of layer_name → quant_method ("int8_per_channel", "int8_per_tensor",
// "fp8_per_channel", "fp8_per_tensor", "dummy", "fp16").
std::unordered_map<std::string, std::string> load_quant_config(const std::string &path);

// Build a per-layer layer_quant_t map from a raw quant_config (layer_name → method string).
// Keys ending in ".act" set the activation method; all others set the weight method.
// If a layer has no ".act" override, act defaults to the weight method.
void build_quant_methods(std::unordered_map<std::string, layer_quant_t> &out,
                         const std::unordered_map<std::string, std::string> &cfg);

// Apply fake quantization to a tensor and return the result.
// No-op if g_quant_active is false, or method is empty / "dummy" / "fp16".
//
// For 2D weight tensors [out, in]: pass transposed=true when stored as [in, out] so that
// per-channel scaling reduces over the correct dimension.
// For activation tensors [..., T, C]: transposed is unused — leading dims are flattened.
//
// "per_channel" in the method string selects per-row scaling (one scale per output channel
// for weights, or per token for activations). "per_tensor" uses a single global scale.
at::Tensor fake_quant(const at::Tensor &x, const std::string &method, bool transposed = false);
