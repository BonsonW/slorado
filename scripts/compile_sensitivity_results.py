#!/usr/bin/env python3
"""Compile per-config TSV sensitivity results into a single summary TSV."""

import os
import statistics

RESULTS = os.path.join(os.path.dirname(__file__), "results")
OUT     = os.path.join(os.path.dirname(__file__), "sensitivity_results.tsv")

# (tag, model, phase, scope, weight_scale, act_scale)
_LSTM_W = [
    ("pc",    "int8 chnl"), ("pt",    "int8 tens"),
    ("fp8pc", "fp8 chnl"),  ("fp8pt", "fp8 tens"),
    ("mxint8","mxint8 g32"),("mxfp4", "mxfp4 g32"),
    ("mxfp6", "mxfp6 g32"), ("mxfp8", "mxfp8 g32"),
]
_LSTM_A = [
    ("ptoken",    "int8 tok"),   ("fp8ptoken", "fp8 tok"),
    ("mxint8",    "mxint8 g32"), ("mxfp4",     "mxfp4 g32"),
    ("mxfp6",     "mxfp6 g32"),  ("mxfp8",     "mxfp8 g32"),
]

RUNS = [
    # ── FLSTM baselines ───────────────────────────────────────────────────────
    ("lstm_fp16", "HAC v6", 0, "all", "fp16", "fp16"),
]
for _lname in ["dn_ih", "up_ih", "dn_hh", "up_hh"]:
    for _mtag, _mdesc in _LSTM_W:
        RUNS.append((f"lstm_{_lname}_w_{_mtag}", "HAC v6", 1, _lname, _mdesc, "fp16"))
    for _mtag, _mdesc in _LSTM_A:
        RUNS.append((f"lstm_{_lname}_a_{_mtag}", "HAC v6", 2, _lname, "fp16", _mdesc))
    if _lname == "dn_hh":
        RUNS.append((f"lstm_{_lname}_a_fixed",    "HAC v6", 2, _lname, "fp16", "int8 fixed (1/127)"))
        RUNS.append((f"lstm_{_lname}_a_fp8fixed", "HAC v6", 2, _lname, "fp16", "fp8 fixed (1/448)"))

RUNS.append(
    # ── Transformer baselines ─────────────────────────────────────────────────
    ("tx_fp16", "SUP v5", 0, "all", "fp16", "fp16"),
)

_TX_W = [
    ("pc",    "int8 chnl"),  ("pt",    "int8 tens"),
    ("fp8pc", "fp8 chnl"),   ("fp8pt", "fp8 tens"),
    ("mxint8","mxint8 g32"), ("mxfp4", "mxfp4 g32"),
    ("mxfp6", "mxfp6 g32"),  ("mxfp8", "mxfp8 g32"),
]
_TX_A = [
    ("ptoken",    "int8 tok"),   ("fp8ptoken", "fp8 tok"),
    ("mxint8",    "mxint8 g32"), ("mxfp4",     "mxfp4 g32"),
    ("mxfp6",     "mxfp6 g32"),  ("mxfp8",     "mxfp8 g32"),
]
for _lname in ["wqkv", "op", "fc1", "fc2"]:
    for _mtag, _mdesc in _TX_W:
        RUNS.append((f"tx_{_lname}_w_{_mtag}", "SUP v5", 1, _lname, _mdesc, "fp16"))
    for _mtag, _mdesc in _TX_A:
        RUNS.append((f"tx_{_lname}_a_{_mtag}", "SUP v5", 2, _lname, "fp16", _mdesc))
    if _lname in ("wqkv", "fc1"):
        RUNS.append((f"tx_{_lname}_a_fixed", "SUP v5", 2, _lname, "fp16", "int8 fixed_4 (4/127)"))


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
