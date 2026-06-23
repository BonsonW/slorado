#!/usr/bin/env python3
"""Compile per-config TSV sensitivity results into a single summary TSV."""

import os

RESULTS = os.path.join(os.path.dirname(__file__), "results")
OUT     = os.path.join(os.path.dirname(__file__), "sensitivity_results.tsv")

# (tag, model, phase, scope, weight_scale, act_scale)
RUNS = [
    # ── FLSTM baselines ───────────────────────────────────────────────────────
    ("lstm_fp16", "HAC v6", 0, "all", "fp16", "fp16"),
    # ── FLSTM Phase 1: weights only ───────────────────────────────────────────
    ("lstm_hh_w_pc",    "HAC v6", 1, "hh", "int8 per_channel", "fp16"),
    ("lstm_hh_w_pt",    "HAC v6", 1, "hh", "int8 per_tensor",  "fp16"),
    ("lstm_hh_w_fp8pc", "HAC v6", 1, "hh", "fp8 per_channel",  "fp16"),
    ("lstm_hh_w_fp8pt", "HAC v6", 1, "hh", "fp8 per_tensor",   "fp16"),
    ("lstm_hh_w_mxint8","HAC v6", 1, "hh", "mxint8 g32",       "fp16"),
    ("lstm_hh_w_mxfp4", "HAC v6", 1, "hh", "mxfp4 g32",        "fp16"),
    ("lstm_hh_w_mxfp6", "HAC v6", 1, "hh", "mxfp6 g32",        "fp16"),
    ("lstm_hh_w_mxfp8", "HAC v6", 1, "hh", "mxfp8 g32",        "fp16"),
    ("lstm_ih_w_pc",    "HAC v6", 1, "ih", "int8 per_channel", "fp16"),
    ("lstm_ih_w_pt",    "HAC v6", 1, "ih", "int8 per_tensor",  "fp16"),
    ("lstm_ih_w_fp8pc", "HAC v6", 1, "ih", "fp8 per_channel",  "fp16"),
    ("lstm_ih_w_fp8pt", "HAC v6", 1, "ih", "fp8 per_tensor",   "fp16"),
    ("lstm_ih_w_mxint8","HAC v6", 1, "ih", "mxint8 g32",       "fp16"),
    ("lstm_ih_w_mxfp4", "HAC v6", 1, "ih", "mxfp4 g32",        "fp16"),
    ("lstm_ih_w_mxfp6", "HAC v6", 1, "ih", "mxfp6 g32",        "fp16"),
    ("lstm_ih_w_mxfp8", "HAC v6", 1, "ih", "mxfp8 g32",        "fp16"),
    # ── FLSTM Phase 2: activations only ───────────────────────────────────────
    ("lstm_hh_a_ptoken",    "HAC v6", 2, "hh", "fp16", "int8 per_token"),
    ("lstm_hh_a_ptensor",   "HAC v6", 2, "hh", "fp16", "int8 per_tensor"),
    ("lstm_hh_a_fixed",     "HAC v6", 2, "hh", "fp16", "int8 fixed (1/127)"),
    ("lstm_hh_a_fp8ptoken", "HAC v6", 2, "hh", "fp16", "fp8 per_token"),
    ("lstm_hh_a_fp8ptensor","HAC v6", 2, "hh", "fp16", "fp8 per_tensor"),
    ("lstm_hh_a_mxint8",    "HAC v6", 2, "hh", "fp16", "mxint8 g32"),
    ("lstm_hh_a_mxfp4",     "HAC v6", 2, "hh", "fp16", "mxfp4 g32"),
    ("lstm_hh_a_mxfp6",     "HAC v6", 2, "hh", "fp16", "mxfp6 g32"),
    ("lstm_hh_a_mxfp8",     "HAC v6", 2, "hh", "fp16", "mxfp8 g32"),
    ("lstm_ih_a_ptoken",    "HAC v6", 2, "ih", "fp16", "int8 per_token"),
    ("lstm_ih_a_ptensor",   "HAC v6", 2, "ih", "fp16", "int8 per_tensor"),
    ("lstm_ih_a_fp8ptoken", "HAC v6", 2, "ih", "fp16", "fp8 per_token"),
    ("lstm_ih_a_fp8ptensor","HAC v6", 2, "ih", "fp16", "fp8 per_tensor"),
    ("lstm_ih_a_mxint8",    "HAC v6", 2, "ih", "fp16", "mxint8 g32"),
    ("lstm_ih_a_mxfp4",     "HAC v6", 2, "ih", "fp16", "mxfp4 g32"),
    ("lstm_ih_a_mxfp6",     "HAC v6", 2, "ih", "fp16", "mxfp6 g32"),
    ("lstm_ih_a_mxfp8",     "HAC v6", 2, "ih", "fp16", "mxfp8 g32"),
    # ── Transformer baselines ─────────────────────────────────────────────────
    ("tx_fp16", "SUP v5", 0, "all", "fp16", "fp16"),
    # ── Transformer Phase 1: weights only ────────────────────────────────────
    ("tx_w_pc",    "SUP v5", 1, "wqkv+fc1+fc2", "int8 per_channel", "fp16"),
    ("tx_w_pt",    "SUP v5", 1, "wqkv+fc1+fc2", "int8 per_tensor",  "fp16"),
    ("tx_w_fp8pc", "SUP v5", 1, "wqkv+fc1+fc2", "fp8 per_channel",  "fp16"),
    ("tx_w_fp8pt", "SUP v5", 1, "wqkv+fc1+fc2", "fp8 per_tensor",   "fp16"),
    ("tx_w_mxint8","SUP v5", 1, "wqkv+fc1+fc2", "mxint8 g32",       "fp16"),
    ("tx_w_mxfp4", "SUP v5", 1, "wqkv+fc1+fc2", "mxfp4 g32",        "fp16"),
    ("tx_w_mxfp6", "SUP v5", 1, "wqkv+fc1+fc2", "mxfp6 g32",        "fp16"),
    ("tx_w_mxfp8", "SUP v5", 1, "wqkv+fc1+fc2", "mxfp8 g32",        "fp16"),
    # ── Transformer Phase 2: activations only ────────────────────────────────
    ("tx_a_ptoken",    "SUP v5", 2, "wqkv+fc1+fc2",       "fp16", "int8 per_token"),
    ("tx_a_ptensor",   "SUP v5", 2, "wqkv+fc1+fc2",       "fp16", "int8 per_tensor"),
    ("tx_a_fixed",     "SUP v5", 2, "wqkv+fc1 (fc2 dyn)", "fp16", "int8 fixed_4 (4/127)"),
    ("tx_a_fp8ptoken", "SUP v5", 2, "wqkv+fc1+fc2",       "fp16", "fp8 per_token"),
    ("tx_a_fp8ptensor","SUP v5", 2, "wqkv+fc1+fc2",       "fp16", "fp8 per_tensor"),
    ("tx_a_mxint8",    "SUP v5", 2, "wqkv+fc1+fc2",       "fp16", "mxint8 g32"),
    ("tx_a_mxfp4",     "SUP v5", 2, "wqkv+fc1+fc2",       "fp16", "mxfp4 g32"),
    ("tx_a_mxfp6",     "SUP v5", 2, "wqkv+fc1+fc2",       "fp16", "mxfp6 g32"),
    ("tx_a_mxfp8",     "SUP v5", 2, "wqkv+fc1+fc2",       "fp16", "mxfp8 g32"),
]

def load(tag):
    # Combined TSV: tag  n_batches  kl_mean  kl_max  identity (one row per read)
    path = os.path.join(RESULTS, f"{tag}.tsv")
    try:
        with open(path) as f:
            f.readline()  # header
            parts = f.readline().split("\t")
        n_batches = parts[1]
        kl_mean   = parts[2]
        kl_max    = parts[3].rstrip()
        if kl_mean == "NA":
            return None, None, None
        return float(kl_mean), float(kl_max), int(n_batches)
    except Exception:
        return None, None, None

fields = ["model", "phase", "scope", "weight_scale", "act_scale",
          "kl_mean", "kl_max", "n_batches"]

with open(OUT, "w") as f:
    f.write("\t".join(fields) + "\n")
    for tag, model, phase, scope, w_scale, a_scale in RUNS:
        kl_mean, kl_max, n_batches = load(tag)
        row = [
            model,
            str(phase),
            scope,
            w_scale,
            a_scale,
            f"{kl_mean:.6g}" if kl_mean is not None else "pending",
            f"{kl_max:.6g}"  if kl_max  is not None else "pending",
            str(n_batches) if n_batches is not None else "",
        ]
        f.write("\t".join(row) + "\n")

print(f"Written to {OUT}")
with open(OUT) as f:
    print(f.read())
