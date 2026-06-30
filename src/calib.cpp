#include "calib.h"

#include <algorithm>
#include <cstdio>
#include <vector>

static void compute_weight_stats(calib_layer_t *layer, const at::Tensor &weight) {
    auto W = weight.detach().to(torch::kFloat32).cpu();
    layer->out_features = W.size(0);
    layer->in_features  = W.size(1);
    layer->w_min = W.min().item<float>();
    layer->w_max = W.max().item<float>();
    layer->w_per_out_ch_max = std::get<0>(W.max(/*dim=*/1));  // (out_features,)
    layer->w_per_out_ch_min = std::get<0>(W.min(/*dim=*/1));  // (out_features,)
}

calib_layer_t* calib_stats_t::register_layer(const std::string &name, const at::Tensor &weight) {
    // If already registered (e.g. by another runner), skip to avoid duplicate entries.
    auto it = by_name.find(name);
    if (it != by_name.end()) {
        return nullptr;
    }

    entries.emplace_back();
    calib_layer_t *layer = &entries.back();
    layer->name = name;
    compute_weight_stats(layer, weight);
    by_name[name] = layer;
    return layer;
}

void calib_stats_t::update_weight(calib_layer_t *layer, const at::Tensor &weight) {
    if (!layer) return;
    compute_weight_stats(layer, weight);
}

void calib_stats_t::accumulate(calib_layer_t *layer, const at::Tensor &input) {
    // Caller must pass input in (..., T, C) layout: last dim = features, second-to-last = seq pos.
    // All reductions run on the input's device (GPU) — only small scalar/vector results come to CPU.
    auto x = input.detach().to(torch::kFloat32);

    layer->x_min = std::min(layer->x_min, x.min().item<float>());
    layer->x_max = std::max(layer->x_max, x.max().item<float>());

    // Per-token amax: reduce over feature dim → (..., T), then collapse batch/leading dims.
    // We track the mean amax per position (sum / n_batches) rather than the running max,
    // which would inflate quiet positions toward the global worst-case over many batches.
    auto tok_max = std::get<0>(x.max(-1));                                       // (..., T) on device
    auto tok_min = std::get<0>(x.min(-1));                                       // (..., T) on device
    int64_t T = tok_max.size(-1);
    auto pos_max  = std::get<0>(tok_max.reshape({-1, T}).max(0));                // (T,) on device
    auto pos_min  = std::get<0>(tok_min.reshape({-1, T}).min(0));                // (T,) on device
    auto pos_amax = torch::maximum(pos_max.abs(), pos_min.abs()).cpu();          // (T,) on CPU

    if (!layer->x_per_token_amax_sum.defined()) {
        layer->x_per_token_amax_sum = pos_amax;
    } else if (pos_amax.size(0) == layer->x_per_token_amax_sum.size(0)) {
        layer->x_per_token_amax_sum += pos_amax;
    }

    layer->n_batches++;
}

void calib_stats_t::save_json(const std::string &path) const {
    FILE *fp = fopen(path.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "[calib] error: cannot open output file %s\n", path.c_str());
        return;
    }

    // Sort entries by name for deterministic output.
    std::vector<const calib_layer_t*> sorted;
    for (const auto &e : entries) {
        sorted.push_back(&e);
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const calib_layer_t *a, const calib_layer_t *b) { return a->name < b->name; });

    fprintf(fp, "{\n  \"layers\": {\n");

    for (size_t li = 0; li < sorted.size(); li++) {
        const auto *L = sorted[li];
        if (li > 0) fprintf(fp, ",\n");

        fprintf(fp, "    \"%s\": {\n", L->name.c_str());
        fprintf(fp, "      \"n_batches\": %ld,\n", (long)L->n_batches);

        int64_t out_f = L->out_features;
        int64_t in_f  = L->in_features;
        int64_t seq_len = L->x_per_token_amax_sum.defined() ? L->x_per_token_amax_sum.size(0) : 0;
        fprintf(fp, "      \"out_features\": %ld,\n", (long)out_f);
        fprintf(fp, "      \"in_features\": %ld,\n", (long)in_f);
        fprintf(fp, "      \"seq_len\": %ld,\n", (long)seq_len);

        // Weight stats — per-channel and per-tensor ranges (max - min) plus amax for scale computation.
        float w_amax = std::max(std::abs(L->w_max), std::abs(L->w_min));
        fprintf(fp, "      \"weight\": {\n");
        fprintf(fp, "        \"per_tensor_range\": %.6g,\n", L->w_max - L->w_min);
        fprintf(fp, "        \"per_tensor_amax\": %.6g", w_amax);
        if (L->w_per_out_ch_max.defined()) {
            // per-channel amax = max(|ch_max|, |ch_min|) — directly comparable to per_tensor_amax.
            // Percentiles show the distribution shape: p50/max ratio tells you how uniform channels
            // are (close to 1 = per-tensor is efficient, low = per-channel quantization helps).
            auto ch_amax = torch::maximum(L->w_per_out_ch_max.abs(), L->w_per_out_ch_min.abs());
            auto pcts = torch::quantile(ch_amax.to(torch::kFloat32),
                                        torch::tensor({0.25f, 0.50f, 0.75f, 0.99f}));
            fprintf(fp, ",\n        \"per_out_channel_amax\": {"
                        "\"p25\": %.6g, \"p50\": %.6g, \"p75\": %.6g, \"p99\": %.6g, \"max\": %.6g}",
                    pcts[0].item<float>(), pcts[1].item<float>(),
                    pcts[2].item<float>(), pcts[3].item<float>(),
                    ch_amax.max().item<float>());
        }
        fprintf(fp, "\n      },\n");

        // Input activation stats — omitted entirely for weights-only calibration runs.
        fprintf(fp, "      \"input\": {");
        if (L->n_batches > 0) {
            float x_amax = std::max(std::abs(L->x_max), std::abs(L->x_min));
            fprintf(fp, "\n        \"per_tensor_range\": %.6g,\n", L->x_max - L->x_min);
            fprintf(fp, "        \"per_tensor_amax\": %.6g", x_amax);
            if (L->x_per_token_amax_sum.defined()) {
                // Divide cumulative sum by batch count to get mean amax per token position.
                auto tok_mean_amax = L->x_per_token_amax_sum / (float)L->n_batches;
                auto tok_pcts = torch::quantile(tok_mean_amax.to(torch::kFloat32),
                                                torch::tensor({0.25f, 0.50f, 0.75f, 0.99f}));
                fprintf(fp, ",\n        \"per_token_amax\": {"
                            "\"p25\": %.6g, \"p50\": %.6g, \"p75\": %.6g, \"p99\": %.6g, \"max\": %.6g}",
                        tok_pcts[0].item<float>(), tok_pcts[1].item<float>(),
                        tok_pcts[2].item<float>(), tok_pcts[3].item<float>(),
                        tok_mean_amax.max().item<float>());
            }
            fprintf(fp, "\n      ");
        }
        fprintf(fp, "}\n");

        fprintf(fp, "    }");
    }

    fprintf(fp, "\n  }\n}\n");
    fclose(fp);
    fprintf(stderr, "[calib] saved %zu layer stats to %s\n", sorted.size(), path.c_str());
}
