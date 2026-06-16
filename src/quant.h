#pragma once

#include <string>
#include <unordered_map>

#include <torch/torch.h>

// Thread-local flag: set to false to bypass quant paths (used for the fp16 baseline pass
// in sensitivity mode). True by default — quant layers run normally.
extern thread_local bool g_quant_active;

// Load a per-layer quantization config from a JSON file.
// Returns a map of layer_name → quant_method ("int8_per_channel", "int8_per_tensor",
// "fp8_per_channel", "fp8_per_tensor", "dummy", "fp16").
std::unordered_map<std::string, std::string> load_quant_config(const std::string &path);

// Apply fake quantization to a weight tensor and return the result.
// No-op (returns W) if g_quant_active is false, or method is empty/dummy/fp16.
// weight_transposed: true when W has shape (in, out) instead of standard (out, in).
//   For transposed weights, per-channel scales along dim 1 (output channels).
at::Tensor maybe_fake_quant(const at::Tensor &W, const std::string &method,
                             bool weight_transposed = false);
