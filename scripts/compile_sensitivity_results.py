#!/usr/bin/env python3
"""Compile sensitivity JSON results into sensitivity_results.csv."""

import json
import csv
import os

RESULTS = os.path.join(os.path.dirname(__file__), "results")
OUT     = os.path.join(os.path.dirname(__file__), "sensitivity_results.csv")

# (tag, model, phase, scope, weight_scale, act_scale)
RUNS = [
    # ── FLSTM Phase 1: weights only ──────────────────────────────────────────
    ("lstm_hh_w_pc", "HAC v6", 1, "hh", "per_channel", "fp16"),
    ("lstm_hh_w_pt", "HAC v6", 1, "hh", "per_tensor",  "fp16"),
    ("lstm_ih_w_pc", "HAC v6", 1, "ih", "per_channel", "fp16"),
    ("lstm_ih_w_pt", "HAC v6", 1, "ih", "per_tensor",  "fp16"),
    # ── FLSTM Phase 2: activations only ──────────────────────────────────────
    ("lstm_hh_a_ptoken",  "HAC v6", 2, "hh", "fp16", "per_token"),
    ("lstm_hh_a_ptensor", "HAC v6", 2, "hh", "fp16", "per_tensor"),
    ("lstm_hh_a_fixed",   "HAC v6", 2, "hh", "fp16", "fixed (1/127)"),
    ("lstm_ih_a_ptoken",  "HAC v6", 2, "ih", "fp16", "per_token"),
    ("lstm_ih_a_ptensor", "HAC v6", 2, "ih", "fp16", "per_tensor"),
    # ── Transformer Phase 1: weights only ────────────────────────────────────
    ("tx_w_pc", "SUP v5", 1, "wqkv+fc1+fc2", "per_channel", "fp16"),
    ("tx_w_pt", "SUP v5", 1, "wqkv+fc1+fc2", "per_tensor",  "fp16"),
    # ── Transformer Phase 2: activations only ────────────────────────────────
    ("tx_a_ptoken",  "SUP v5", 2, "wqkv+fc1+fc2",      "fp16", "per_token"),
    ("tx_a_ptensor", "SUP v5", 2, "wqkv+fc1+fc2",      "fp16", "per_tensor"),
    ("tx_a_fixed",   "SUP v5", 2, "wqkv+fc1 (fc2 dyn)", "fp16", "fixed_4 (4/127)"),
]

def load(tag):
    path = os.path.join(RESULTS, f"sens_{tag}.json")
    try:
        d = json.load(open(path))
        return d.get("kl_mean"), d.get("kl_max"), d.get("n_batches")
    except Exception:
        return None, None, None

fields = ["model", "phase", "scope", "weight_scale", "act_scale",
          "kl_mean", "kl_max", "n_batches"]

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for tag, model, phase, scope, w_scale, a_scale in RUNS:
        kl_mean, kl_max, n_batches = load(tag)
        w.writerow({
            "model":        model,
            "phase":        phase,
            "scope":        scope,
            "weight_scale": w_scale,
            "act_scale":    a_scale,
            "kl_mean":      f"{kl_mean:.6g}" if kl_mean is not None else "pending",
            "kl_max":       f"{kl_max:.6g}"  if kl_max  is not None else "pending",
            "n_batches":    n_batches or "",
        })

print(f"Written to {OUT}")
with open(OUT) as f:
    print(f.read())
