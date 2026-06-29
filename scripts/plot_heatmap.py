#!/usr/bin/env python3
"""Heatmap per scope × quantization method, one figure per model.

Usage:
  python plot_heatmap.py [--metric kl|identity]   (default: kl)

  kl:       color = log10(KL mean), annotate with KL value
  identity: color = mean identity (or drop vs fp16), annotate with identity value
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=["kl", "identity"], default="kl")
args = parser.parse_args()

TSV = os.path.join(os.path.dirname(__file__), "sensitivity_results.tsv")
OUT = os.path.dirname(__file__)

# ── Load ───────────────────────────────────────────────────────────────────────

df_raw = pd.read_csv(TSV, sep="\t")
df_raw = df_raw[df_raw["kl_mean"] != "pending"].copy()

for col in ("mean_id", "median_id"):
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

baselines = (df_raw[df_raw["kl_mean"] == "N/A"]
             .groupby("model")["mean_id"].first())

df_raw["kl_mean"] = pd.to_numeric(df_raw["kl_mean"], errors="coerce")
df = df_raw.dropna(subset=["kl_mean"]).copy()

def get_phase(row):
    if row["act_scale"] == "fp16":
        return "weights"
    elif row["wt_scale"] == "fp16":
        return "activations"
    else:
        return "mx"

df["phase"] = df.apply(get_phase, axis=1)
df["id_drop_pp"] = df.apply(
    lambda r: (baselines.get(r["model"], np.nan) - r["mean_id"]) * 100, axis=1
)

# ── Layout config ──────────────────────────────────────────────────────────────

W_ORDER  = ["int4 tens", "int4 chnl", "int8 tens", "int8 chnl", "fp8 tens", "fp8 chnl"]
A_ORDER  = ["int4 tok", "int8 tok", "fp8 tok",
            "int8 fixed (1/127)", "fp8 fixed (1/448)", "int8 fixed_4 (4/127)"]
MX_ORDER = ["mxint8 g32", "mxfp4 g32", "mxfp6 g32", "mxfp8 g32"]

W_LABELS  = {"int4 tens": "int4\ntens", "int4 chnl": "int4\nchnl",
             "int8 tens": "int8\ntens", "int8 chnl": "int8\nchnl",
             "fp8 tens":  "fp8\ntens",  "fp8 chnl":  "fp8\nchnl"}
A_LABELS  = {"int4 tok": "int4\ntok", "int8 tok": "int8\ntok", "fp8 tok": "fp8\ntok",
             "int8 fixed (1/127)": "int8\nfixed", "fp8 fixed (1/448)": "fp8\nfixed",
             "int8 fixed_4 (4/127)": "int8\nfixed4"}
MX_LABELS = {"mxint8 g32": "mxint8", "mxfp4 g32": "mxfp4",
             "mxfp6 g32": "mxfp6",   "mxfp8 g32": "mxfp8"}

LSTM_SCOPES = ["dn_ih", "up_ih", "dn_hh", "up_hh"]
TX_SCOPES   = ["wqkv", "op", "fc1", "fc2"]

# ── Draw panel ────────────────────────────────────────────────────────────────

def pivot_color_data(sub, phase, scopes, col_order):
    """Return the color-encoded data for a panel (used for range computation and drawing)."""
    col_key  = "wt_scale" if phase in ("weights", "mx") else "act_scale"
    present  = [c for c in col_order if c in sub[col_key].values]
    phase_df = sub[sub["phase"] == phase]
    if not present or phase_df.empty:
        return None, None, None

    if args.metric == "kl":
        pivot = (phase_df
                 .pivot_table(index="scope", columns=col_key,
                              values="kl_mean", aggfunc="mean")
                 .reindex(index=scopes, columns=present))
        return np.log10(pivot.replace(0, np.nan)), present, col_key
    else:
        use_drop = sub["id_drop_pp"].notna().any()
        val      = "id_drop_pp" if use_drop else "mean_id"
        pivot    = (phase_df
                    .pivot_table(index="scope", columns=col_key,
                                 values=val, aggfunc="mean")
                    .reindex(index=scopes, columns=present))
        return pivot, present, col_key


def draw_panel(ax, sub, phase, scopes, col_order, col_labels, title, vmin, vmax):
    color_data, present, col_key = pivot_color_data(sub, phase, scopes, col_order)
    if color_data is None:
        ax.set_visible(False)
        return

    phase_df = sub[sub["phase"] == phase]

    if args.metric == "kl":
        kl_pivot = (phase_df
                    .pivot_table(index="scope", columns=col_key,
                                 values="kl_mean", aggfunc="mean")
                    .reindex(index=scopes, columns=present))
        cmap       = "RdYlGn_r"
        cbar_label = "log₁₀(KL mean)"
        annot      = kl_pivot.map(lambda v: f"{v:.1e}" if pd.notna(v) else "")
    else:
        use_drop   = sub["id_drop_pp"].notna().any()
        cmap       = "RdYlGn_r" if use_drop else "RdYlGn"
        cbar_label = "Identity drop vs fp16 (pp)" if use_drop else "Mean identity"
        fmt_fn     = (lambda v: f"{v:+.2f}pp") if use_drop else (lambda v: f"{v:.4f}")
        annot      = color_data.map(lambda v: fmt_fn(v) if pd.notna(v) else "")

    annot_size = 8 if args.metric == "identity" else 7
    sns.heatmap(color_data, ax=ax, cmap=cmap, annot=annot, fmt="",
                vmin=vmin, vmax=vmax,
                annot_kws={"size": annot_size}, linewidths=0.4, linecolor="white",
                cbar_kws={"label": cbar_label, "shrink": 0.8})

    ax.set_xticklabels([col_labels.get(c, c) for c in present],
                       fontsize=7, rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=0)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("")

# ── Per-model figures ─────────────────────────────────────────────────────────

for model_name, scopes in [("HAC v6", LSTM_SCOPES), ("SUP v5", TX_SCOPES)]:
    mdf = df[df["model"] == model_name]

    fig_h = 5 if args.metric == "identity" else 4
    fig, axes = plt.subplots(1, 3, figsize=(16, fig_h),
                             gridspec_kw={"width_ratios": [6, 6, 4]})
    metric_desc = ("Color: log₁₀(KL mean)" if args.metric == "kl"
                   else "Color: identity drop vs fp16"
                        if df["id_drop_pp"].notna().any()
                        else "Color: mean identity")
    fig.suptitle(f"{model_name} — Quantization Sensitivity  ·  {metric_desc}",
                 fontsize=11, fontweight="bold", y=1.03)

    # Compute shared color scale across all three panels
    all_vals = []
    for ph, order in [("weights", W_ORDER), ("activations", A_ORDER), ("mx", MX_ORDER)]:
        cd, _, _ = pivot_color_data(mdf, ph, scopes, order)
        if cd is not None:
            all_vals.append(cd.values.flatten())
    all_vals = np.concatenate(all_vals)
    # For identity, clip to 5th–95th percentile so outliers don't wash out
    # the colour differences among well-behaved configs.
    # For KL the log scale already handles range well, so use full range.
    if args.metric == "identity":
        vmin = np.nanpercentile(all_vals, 5)
        vmax = np.nanpercentile(all_vals, 95)
    else:
        vmin = np.nanmin(all_vals)
        vmax = np.nanmax(all_vals)

    draw_panel(axes[0], mdf, "weights",     scopes, W_ORDER,  W_LABELS,
               "Phase 1 — weights only (act = fp16)", vmin, vmax)
    draw_panel(axes[1], mdf, "activations", scopes, A_ORDER,  A_LABELS,
               "Phase 2 — activations only (wt = fp16)", vmin, vmax)
    draw_panel(axes[2], mdf, "mx",          scopes, MX_ORDER, MX_LABELS,
               "MX — W + A matched pairs", vmin, vmax)

    plt.tight_layout()
    safe = model_name.replace(" ", "_").lower()
    path = os.path.join(OUT, f"heatmap_{safe}_{args.metric}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

print("Done.")
