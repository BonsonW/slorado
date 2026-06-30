#!/usr/bin/env python3
"""Per-scope grouped bar chart: KL divergence or identity drop.

Usage:
  python scripts/plot_per_layer.py [--metric kl|identity]   (default: kl)

One figure per model, one subplot per scope. Bars grouped by phase
(weights-only | activations-only | MX W+A), colored by format.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=["kl", "identity"], default="kl")
args = parser.parse_args()

TSV = os.path.join(os.path.dirname(__file__), "sensitivity_results.tsv")
OUT = os.path.dirname(__file__)

# ── Data loading ───────────────────────────────────────────────────────────────
df_raw = pd.read_csv(TSV, sep="\t")
df_raw = df_raw[df_raw["kl_mean"] != "pending"].copy()
df_raw["mean_id"] = pd.to_numeric(df_raw["mean_id"], errors="coerce")
baselines = df_raw[df_raw["kl_mean"] == "N/A"].groupby("model")["mean_id"].first()
df_raw["kl_mean"] = pd.to_numeric(df_raw["kl_mean"], errors="coerce")
df = df_raw.dropna(subset=["kl_mean"]).copy()

df["phase"] = df.apply(
    lambda r: "weights"     if r["act_scale"] == "fp16" else
              "activations" if r["wt_scale"]  == "fp16" else
              "mx",
    axis=1,
)
df["id_drop_pp"] = df.apply(
    lambda r: (baselines.get(r["model"], np.nan) - r["mean_id"]) * 100, axis=1
)

# ── Style ──────────────────────────────────────────────────────────────────────
FORMAT_COLORS = {
    "int4":   "#dc143c",
    "int8":   "#3498db",
    "fp8":    "#b8d400",
    "mxint8": "#9b59b6",
    "mxfp4":  "#ffb3d9",
    "mxfp6":  "#1abc9c",
    "mxfp8":  "#f39c12",
}
COLOR_ORDER = ["mxint8", "mxfp4", "mxfp6", "mxfp8", "int4", "int8", "fp8"]

PHASE_BG    = {"weights": "#eef2ff", "activations": "#fff4ee", "mx": "#eefff4"}
PHASE_TITLE = {"weights": "weights only", "activations": "activations only", "mx": "MX W+A"}

def scale_color(s):
    for k in COLOR_ORDER:
        if k in s:
            return FORMAT_COLORS[k]
    return "#888888"

# Method draw order per phase (most → least aggressive bit-width)
W_SEQ  = ["int4 tens", "int4 chnl", "int8 tens", "int8 chnl", "fp8 tens", "fp8 chnl"]
A_SEQ  = ["int4 tok", "int8 tok", "fp8 tok",
           "int8 fixed (1/127)", "fp8 fixed (1/448)", "int8 fixed_4 (4/127)"]
MX_SEQ = ["mxfp4 g32", "mxfp6 g32", "mxint8 g32", "mxfp8 g32"]

SHORT = {
    "int4 tens": "int4\ntens", "int4 chnl": "int4\nchnl",
    "int8 tens": "int8\ntens", "int8 chnl": "int8\nchnl",
    "fp8 tens":  "fp8\ntens",  "fp8 chnl":  "fp8\nchnl",
    "int4 tok":  "int4\ntok",  "int8 tok":  "int8\ntok",  "fp8 tok":  "fp8\ntok",
    "int8 fixed (1/127)":   "int8\nfixed",
    "fp8 fixed (1/448)":    "fp8\nfixed",
    "int8 fixed_4 (4/127)": "int8\nfix4",
    "mxfp4 g32": "mxfp4", "mxfp6 g32": "mxfp6",
    "mxint8 g32": "mxint8", "mxfp8 g32": "mxfp8",
}

GAP = 1.2   # extra x-gap between phase groups
BAR_W = 0.7

# ── Bar collection ─────────────────────────────────────────────────────────────
def collect_bars(sdf):
    """Return bars list + phase spans for one scope dataframe."""
    bars  = []  # (x, short_label, color, y_value)
    spans = []  # (x_left, x_right, phase_name)
    x = 0
    for phase, seq, key in [
        ("weights",     W_SEQ,  "wt_scale"),
        ("activations", A_SEQ,  "act_scale"),
        ("mx",          MX_SEQ, "wt_scale"),
    ]:
        sub     = sdf[sdf["phase"] == phase]
        present = [m for m in seq if m in sub[key].values]
        if not present:
            continue
        x_start = x - 0.5
        for m in present:
            row = sub[sub[key] == m]
            if row.empty:
                x += 1
                continue
            if args.metric == "kl":
                v = row["kl_mean"].values[0]
                y = v if (pd.notna(v) and v > 0) else np.nan
            else:
                v = row["id_drop_pp"].values[0]
                y = v if pd.notna(v) else np.nan
            bars.append((x, SHORT.get(m, m), scale_color(m), y))
            x += 1
        spans.append((x_start, x - 0.5, phase))
        x += GAP
    return bars, spans


# ── Draw one subplot ───────────────────────────────────────────────────────────
def draw_scope(ax, sdf, show_ylabel=False):
    bars, spans = collect_bars(sdf)
    if not bars:
        ax.set_visible(False)
        return

    # Phase background bands
    for x_s, x_e, phase in spans:
        ax.axvspan(x_s, x_e, color=PHASE_BG[phase], alpha=0.65, zorder=0)

    # Bars
    for x, label, color, y in bars:
        if pd.notna(y):
            ax.bar(x, y, color=color, width=BAR_W, alpha=0.88,
                   edgecolor="white", linewidth=0.5, zorder=2)

    # Zero / reference lines
    if args.metric == "identity":
        ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    if args.metric == "kl":
        ax.set_yscale("log")

    # Phase labels at top (x in data coords, y in axes fraction via xaxis_transform)
    for x_s, x_e, phase in spans:
        ax.text((x_s + x_e) / 2, 0.985, PHASE_TITLE[phase],
                ha="center", va="top", fontsize=6.5, color="#444444",
                style="italic", transform=ax.get_xaxis_transform())

    ax.set_xticks([b[0] for b in bars])
    ax.set_xticklabels([b[1] for b in bars], fontsize=7, rotation=0, ha="center")
    ax.set_xlim(bars[0][0] - 0.8, bars[-1][0] + 0.8)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5, which="both", zorder=0)
    ax.tick_params(axis="x", length=2)

    if show_ylabel:
        y_label = "KL divergence (log scale)" if args.metric == "kl" \
                  else "Identity drop vs fp16 (pp)"
        ax.set_ylabel(y_label, fontsize=10)


# ── Per-model figures ──────────────────────────────────────────────────────────
for model_name, scopes in [("HAC v6", ["dn_ih", "up_ih", "dn_hh", "up_hh"]),
                            ("SUP v5", ["wqkv", "op", "fc1", "fc2"])]:
    mdf = df[df["model"] == model_name]
    if mdf.empty:
        print(f"No data for {model_name}, skipping")
        continue

    fig, axes = plt.subplots(
        1, len(scopes),
        figsize=(4.8 * len(scopes), 5.2),
        sharey=True,
        layout="constrained",
        gridspec_kw={"wspace": 0.04},
    )

    for i, (ax, scope) in enumerate(zip(axes, scopes)):
        draw_scope(ax, mdf[mdf["scope"] == scope], show_ylabel=(i == 0))
        ax.set_title(scope, fontsize=11, fontweight="bold", pad=20)

    # Color legend (format)
    color_patches = [mpatches.Patch(color=FORMAT_COLORS[k], label=k)
                     for k in COLOR_ORDER]
    fig.legend(handles=color_patches, title="format",
               loc="upper right", bbox_to_anchor=(1.0, 1.0),
               fontsize=8, title_fontsize=8, framealpha=0.9, ncol=1)

    metric_str = "KL divergence (log scale, lower = better)" \
                 if args.metric == "kl" \
                 else "Identity drop vs fp16 (pp, lower = better)"
    fig.suptitle(f"{model_name} — Per-scope sensitivity  ·  {metric_str}",
                 fontsize=12, fontweight="bold", y=1.01)

    safe = model_name.replace(" ", "_").lower()
    path = os.path.join(OUT, f"per_layer_{safe}_{args.metric}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

print("Done.")
