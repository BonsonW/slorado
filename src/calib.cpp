#include "calib.h"

#include <algorithm>
#include <cstdio>
#include <vector>

static void compute_weight_stats(calib_layer_t *layer, const at::Tensor &weight) {
    auto W = weight.detach().to(torch::kFloat32).cpu();
    layer->out_features = W.size(0);
    layer->in_features  = W.size(1);
    layer->w_min  = W.min().item<float>();
    layer->w_max  = W.max().item<float>();
    layer->w_amax = W.abs().max().item<float>();
    layer->w_per_out_ch_amax = std::get<0>(W.abs().max(/*dim=*/1));  // (out_features,)
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
    // Move to CPU as float32 before any reduction.
    auto x = input.detach().to(torch::kFloat32).cpu();

    layer->x_min  = std::min(layer->x_min,  x.min().item<float>());
    layer->x_max  = std::max(layer->x_max,  x.max().item<float>());
    layer->x_amax = std::max(layer->x_amax, x.abs().max().item<float>());

    // Per-token amax: max |x| over the feature dim (last) → (..., T).
    // Then max over all batch/leading dims → (T,), one value per sequence position.
    auto tok_amax = std::get<0>(x.abs().max(-1));          // (..., T)
    int64_t T = tok_amax.size(-1);
    auto pos_amax = std::get<0>(tok_amax.reshape({-1, T}).max(0));  // (T,)

    if (!layer->x_per_token_amax.defined()) {
        layer->x_per_token_amax = pos_amax;
    } else if (pos_amax.size(0) == layer->x_per_token_amax.size(0)) {
        layer->x_per_token_amax = torch::maximum(layer->x_per_token_amax, pos_amax);
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
        int64_t seq_len = L->x_per_token_amax.defined() ? L->x_per_token_amax.size(0) : 0;
        fprintf(fp, "      \"out_features\": %ld,\n", (long)out_f);
        fprintf(fp, "      \"in_features\": %ld,\n", (long)in_f);
        fprintf(fp, "      \"seq_len\": %ld,\n", (long)seq_len);

        // Weight stats
        fprintf(fp, "      \"weight\": {\n");
        fprintf(fp, "        \"per_tensor\": {\"min\": %.6g, \"max\": %.6g, \"amax\": %.6g},\n",
                L->w_min, L->w_max, L->w_amax);
        fprintf(fp, "        \"per_out_channel_amax\": [");
        if (L->w_per_out_ch_amax.defined()) {
            const float *d = L->w_per_out_ch_amax.data_ptr<float>();
            for (int64_t i = 0; i < out_f; i++) {
                if (i) fprintf(fp, ", ");
                fprintf(fp, "%.6g", d[i]);
            }
        }
        fprintf(fp, "]\n");
        fprintf(fp, "      },\n");

        // Input activation stats
        fprintf(fp, "      \"input\": {\n");
        fprintf(fp, "        \"per_tensor\": {\"min\": %.6g, \"max\": %.6g, \"amax\": %.6g},\n",
                L->x_min, L->x_max, L->x_amax);
        fprintf(fp, "        \"per_token_amax\": [");
        if (L->x_per_token_amax.defined()) {
            const float *d = L->x_per_token_amax.data_ptr<float>();
            for (int64_t i = 0; i < seq_len; i++) {
                if (i) fprintf(fp, ", ");
                fprintf(fp, "%.6g", d[i]);
            }
        }
        fprintf(fp, "]\n");
        fprintf(fp, "      }\n");

        fprintf(fp, "    }");
    }

    fprintf(fp, "\n  }\n}\n");
    fclose(fp);
    fprintf(stderr, "[calib] saved %zu layer stats to %s\n", sorted.size(), path.c_str());
}
