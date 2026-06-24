#!/usr/bin/env python3
"""Compile per-config TSV sensitivity results into a single summary TSV."""

import os
import statistics

RESULTS = os.path.join(os.path.dirname(__file__), "results")
OUT     = os.path.join(os.path.dirname(__file__), "sensitivity_results.tsv")

# (tag, model, phase, scope, weight_scale, act_scale)
RUNS = [
    # ── FLSTM baselines ───────────────────────────────────────────────────────
    ("lstm_fp16",           "HAC v6", 0, "all", "fp16", "fp16"),
    # ── FLSTM Phase 1: weights only ───────────────────────────────────────────
    ("lstm_hh_w_pc",        "HAC v6", 1, "hh", "int8 chnl", "fp16"),
    ("lstm_hh_w_pt",        "HAC v6", 1, "hh", "int8 tens",  "fp16"),
    ("lstm_hh_w_fp8pc",     "HAC v6", 1, "hh", "fp8 chnl",  "fp16"),
    ("lstm_hh_w_fp8pt",     "HAC v6", 1, "hh", "fp8 tens",   "fp16"),
    ("lstm_hh_w_mxint8",    "HAC v6", 1, "hh", "mxint8 g32",       "fp16"),
    ("lstm_hh_w_mxfp4",     "HAC v6", 1, "hh", "mxfp4 g32",        "fp16"),
    ("lstm_hh_w_mxfp6",     "HAC v6", 1, "hh", "mxfp6 g32",        "fp16"),
    ("lstm_hh_w_mxfp8",     "HAC v6", 1, "hh", "mxfp8 g32",        "fp16"),
    ("lstm_ih_w_pc",        "HAC v6", 1, "ih", "int8 chnl", "fp16"),
    ("lstm_ih_w_pt",        "HAC v6", 1, "ih", "int8 tens",  "fp16"),
    ("lstm_ih_w_fp8pc",     "HAC v6", 1, "ih", "fp8 chnl",  "fp16"),
    ("lstm_ih_w_fp8pt",     "HAC v6", 1, "ih", "fp8 tens",   "fp16"),
    ("lstm_ih_w_mxint8",    "HAC v6", 1, "ih", "mxint8 g32",       "fp16"),
    ("lstm_ih_w_mxfp4",     "HAC v6", 1, "ih", "mxfp4 g32",        "fp16"),
    ("lstm_ih_w_mxfp6",     "HAC v6", 1, "ih", "mxfp6 g32",        "fp16"),
    ("lstm_ih_w_mxfp8",     "HAC v6", 1, "ih", "mxfp8 g32",        "fp16"),
    # ── FLSTM Phase 2: activations only ───────────────────────────────────────
    ("lstm_hh_a_ptoken",    "HAC v6", 2, "hh", "fp16", "int8 tok"),
    ("lstm_hh_a_fixed",     "HAC v6", 2, "hh", "fp16", "int8 fixed (1/127)"),
    ("lstm_hh_a_fp8fixed",  "HAC v6", 2, "hh", "fp16", "fp8 fixed (1/448)"),
    ("lstm_hh_a_fp8ptoken", "HAC v6", 2, "hh", "fp16", "fp8 tok"),
    ("lstm_hh_a_mxint8",    "HAC v6", 2, "hh", "fp16", "mxint8 g32"),
    ("lstm_hh_a_mxfp4",     "HAC v6", 2, "hh", "fp16", "mxfp4 g32"),
    ("lstm_hh_a_mxfp6",     "HAC v6", 2, "hh", "fp16", "mxfp6 g32"),
    ("lstm_hh_a_mxfp8",     "HAC v6", 2, "hh", "fp16", "mxfp8 g32"),
    ("lstm_ih_a_ptoken",    "HAC v6", 2, "ih", "fp16", "int8 tok"),
    ("lstm_ih_a_fp8ptoken", "HAC v6", 2, "ih", "fp16", "fp8 tok"),
    ("lstm_ih_a_mxint8",    "HAC v6", 2, "ih", "fp16", "mxint8 g32"),
    ("lstm_ih_a_mxfp4",     "HAC v6", 2, "ih", "fp16", "mxfp4 g32"),
    ("lstm_ih_a_mxfp6",     "HAC v6", 2, "ih", "fp16", "mxfp6 g32"),
    ("lstm_ih_a_mxfp8",     "HAC v6", 2, "ih", "fp16", "mxfp8 g32"),
    # ── Transformer baselines ─────────────────────────────────────────────────
    ("tx_fp16",             "SUP v5", 0, "all", "fp16", "fp16"),
    # ── Transformer Phase 1: weights only ────────────────────────────────────
    ("tx_w_pc",             "SUP v5", 1, "wqkv+op+fc1+fc2", "int8 chnl", "fp16"),
    ("tx_w_pt",             "SUP v5", 1, "wqkv+op+fc1+fc2", "int8 tens",  "fp16"),
    ("tx_w_fp8pc",          "SUP v5", 1, "wqkv+op+fc1+fc2", "fp8 chnl",  "fp16"),
    ("tx_w_fp8pt",          "SUP v5", 1, "wqkv+op+fc1+fc2", "fp8 tens",   "fp16"),
    ("tx_w_mxint8",         "SUP v5", 1, "wqkv+op+fc1+fc2", "mxint8 g32",       "fp16"),
    ("tx_w_mxfp4",          "SUP v5", 1, "wqkv+op+fc1+fc2", "mxfp4 g32",        "fp16"),
    ("tx_w_mxfp6",          "SUP v5", 1, "wqkv+op+fc1+fc2", "mxfp6 g32",        "fp16"),
    ("tx_w_mxfp8",          "SUP v5", 1, "wqkv+op+fc1+fc2", "mxfp8 g32",        "fp16"),
    # ── Transformer Phase 2: activations only ────────────────────────────────
    ("tx_a_ptoken",         "SUP v5", 2, "wqkv+op+fc1+fc2",       "fp16", "int8 tok"),
    ("tx_a_fixed",          "SUP v5", 2, "wqkv+op+fc1 (fc2 dyn)", "fp16", "int8 fixed_4 (4/127)"),
    ("tx_a_fp8ptoken",      "SUP v5", 2, "wqkv+op+fc1+fc2",       "fp16", "fp8 tok"),
    ("tx_a_mxint8",         "SUP v5", 2, "wqkv+op+fc1+fc2",       "fp16", "mxint8 g32"),
    ("tx_a_mxfp4",          "SUP v5", 2, "wqkv+op+fc1+fc2",       "fp16", "mxfp4 g32"),
    ("tx_a_mxfp6",          "SUP v5", 2, "wqkv+op+fc1+fc2",       "fp16", "mxfp6 g32"),
    ("tx_a_mxfp8",          "SUP v5", 2, "wqkv+op+fc1+fc2",       "fp16", "mxfp8 g32"),
]


def load_kl(tag):
    path = os.path.join(RESULTS, f"{tag}.tsv")
    try:
        with open(path) as f:
            f.readline()  # header
            parts = f.readline().split("\t")
        if parts[2].strip() == "NA":
            return "N/A", "N/A"         # baseline — intentionally not quantized
        return float(parts[2]), float(parts[3].rstrip())
    except FileNotFoundError:
        return None, None               # not yet run
    except Exception:
        return None, None


def load_id(tag):
    path = os.path.join(RESULTS, f"{tag}_id.tsv")
    try:
        with open(path) as f:
            scores = [float(l) for l in f if l.strip()]
        if not scores:
            return None, None, None
        return statistics.mean(scores), statistics.median(scores), len(scores)
    except Exception:
        return None, None, None


fields = ["model", "scope", "wt_scale", "act_scale",
          "kl_mean", "kl_max",
          "mean_id", "median_id", "n_aligned"]

with open(OUT, "w") as f:
    f.write("\t".join(fields) + "\n")
    for tag, model, phase, scope, w_scale, a_scale in RUNS:
        kl_mean, kl_max = load_kl(tag)
        mean_id, median_id, n_aligned = load_id(tag)
        row = [
            model,
            scope,
            w_scale,
            a_scale,
            kl_mean if isinstance(kl_mean, str) else (f"{kl_mean:.3e}" if kl_mean is not None else "pending"),
            kl_max  if isinstance(kl_max,  str) else (f"{kl_max:.3e}"  if kl_max  is not None else "pending"),
            f"{mean_id:.4f}"   if mean_id   is not None else "pending",
            f"{median_id:.4f}" if median_id is not None else "pending",
            str(n_aligned)     if n_aligned  is not None else "",
        ]
        f.write("\t".join(row) + "\n")

print(f"Written to {OUT}")
with open(OUT) as f:
    print(f.read())
