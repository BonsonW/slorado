#include "sensitivity.h"

#include <cstdio>
#include <cmath>

static float kl_divergence(const at::Tensor &p_logits, const at::Tensor &q_logits) {
    // KL(P || Q), averaged over all tokens and batch items.
    // P = baseline (fp16), Q = quantized.
    auto p     = torch::softmax(p_logits.to(torch::kFloat32), -1);
    auto log_q = torch::log_softmax(q_logits.to(torch::kFloat32), -1);
    auto log_p = torch::log(p.clamp_min(1e-9f));
    return (p * (log_p - log_q)).sum(-1).mean().item<float>();
}

void sensitivity_stats_t::accumulate(const at::Tensor &fp16_logits, const at::Tensor &quant_logits) {
    auto p = fp16_logits.detach().cpu();
    auto q = quant_logits.detach().cpu();
    float kl = kl_divergence(p, q);
    kl_sum += kl;
    if (kl > kl_max) kl_max = kl;
    n_batches++;
}

void sensitivity_stats_t::save_json(const std::string &path, const char *quant_config_path) const {
    FILE *fp = fopen(path.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "[sensitivity] error: cannot open output file %s\n", path.c_str());
        return;
    }
    float kl_mean = n_batches > 0 ? (float)(kl_sum / n_batches) : 0.f;
    fprintf(fp, "{\n");
    if (quant_config_path) {
        fprintf(fp, "  \"quant_config\": \"%s\",\n", quant_config_path);
    }
    fprintf(fp, "  \"n_batches\": %ld,\n", (long)n_batches);
    fprintf(fp, "  \"kl_mean\": %.6g,\n", kl_mean);
    fprintf(fp, "  \"kl_max\": %.6g\n", kl_max);
    fprintf(fp, "}\n");
    fclose(fp);
    fprintf(stderr, "[sensitivity] saved sensitivity stats to %s (n_batches=%ld, kl_mean=%.4g, kl_max=%.4g)\n",
            path.c_str(), (long)n_batches, kl_mean, kl_max);
}
