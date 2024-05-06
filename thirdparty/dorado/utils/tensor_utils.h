#pragma once

#include <torch/torch.h>

#include <string>
#include <vector>

// Serialise Torch tensor to disk.
void serialise_tensor(torch::Tensor t, const std::string& path);
// Load serialised tensor from disk.
std::vector<torch::Tensor> load_tensors(const std::string& dir,
                                        const std::vector<std::string>& tensors);

// Computes the q-th quantiles of each row of the input tensor `t`
// using a partial sort as opposed a full sort per torch::quantiles
// Only `interpolation='lower'` is currently implemented.
torch::Tensor quantile(const torch::Tensor t, const torch::Tensor q);

// Computes the q-th quantiles of each row of the input tensor `t`
// using a counting sort which is extremely fast for low range integers.
// Only `interpolation='lower'` is currently implemented.
torch::Tensor quantile_counting(const torch::Tensor t, const torch::Tensor q);

// Quantize a tensor to int8, returning a pair of tensors `{scales, quantized_tensor}`, where:
// `scales` is the same size as `tensor` with dimension 0 dropped, dtype float
// `quantized_tensor` is the same size as `tensor`, dtype int8
// such that `quantized_tensor / scales ~= tensor`
std::pair<at::Tensor, at::Tensor> quantize_tensor(const at::Tensor& tensor);