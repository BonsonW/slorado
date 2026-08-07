#!/usr/bin/env python3
"""Scatter plot of KL divergence vs identity drop, one figure per model.

Usage:
  python plot_scatter.py [--metric kl|identity]   (default: kl)

  kl:       x=KL mean (log), y=identity drop vs fp16
  identity: x=scope/method label, y=mean identity (bar-style scatter)
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
parser.add_argument("--metric", choices=["kl", "identity"], default="kl")
args = parser.parse_args()

TSV = os.path.join(os.path.dirname(__file__), "sensitivity_results.tsv")
OUT = os.path.dirname(__file__)

# ── Load ───────────────────────────────────────────────────────────────────────

df_raw = pd.read_csv(TSV, sep="\t")
df_raw = df_raw[df_raw["kl_mean"] != "pending"].copy()
df_raw["mean_id"] = pd.to_numeric(df_raw["mean_id"], errors="coerce")

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

# ── Style ──────────────────────────────────────────────────────────────────────

FORMAT_COLORS = {
    # standard formats
    "int4":   "#dc143c",
    "int8":   "#3498db",
    "fp8":    "#b8d400",
    # MX formats — check these before generic substrings
    "mxint8": "#9b59b6",
    "mxfp4":  "#ffb3d9",
    "mxfp6":  "#1abc9c",
    "mxfp8":  "#f39c12",
}
# Lookup order: specific MX keys before generic int/fp keys
COLOR_LOOKUP_ORDER = ["mxint8", "mxfp4", "mxfp6", "mxfp8", "int4", "int8", "fp8"]

PHASE_MARKERS = {"weights": "o", "activations": "s", "mx": "D"}
PHASE_LABELS  = {"weights": "weights only", "activations": "act only", "mx": "MX W+A"}

def fmt_color(row):
    val = row["wt_scale"] if row["phase"] in ("weights", "mx") else row["act_scale"]
    for k in COLOR_LOOKUP_ORDER:
        if k in val:
            return FORMAT_COLORS[k]
    return "#888888"

df["fmt_color"] = df.apply(fmt_color, axis=1)

# ── Per-model figures ─────────────────────────────────────────────────────────

for model_name in ["HAC v6", "SUP v5"]:
    mdf = df[df["model"] == model_name].copy()
    if mdf.empty:
        print(f"No data for {model_name}, skipping")
        continue

    safe = model_name.replace(" ", "_").lower()

    if args.metric == "kl":
        plot_df = mdf.dropna(subset=["kl_mean"])
        if plot_df.empty:
            print(f"No KL data for {model_name}, skipping")
            continue

        has_id  = plot_df["id_drop_pp"].notna().any()
        y_col   = "id_drop_pp" if has_id else "mean_id"
        y_label = ("Identity drop vs fp16 (pp)" if has_id else "Mean identity")
        plot_df = plot_df.dropna(subset=[y_col])

        fig, ax = plt.subplots(figsize=(9, 6))
        for phase, marker in PHASE_MARKERS.items():
            sub = plot_df[plot_df["phase"] == phase]
            if sub.empty:
                continue
            ax.scatter(sub["kl_mean"], sub[y_col],
                       c=sub["fmt_color"], marker=marker,
                       s=28, alpha=0.85, edgecolors="white", linewidths=0.3,
                       zorder=3)

        for _, r in plot_df.nlargest(5, "kl_mean").iterrows():
            label = (f"{r['scope']}\n"
                     f"{r['wt_scale'] if r['phase'] != 'activations' else r['act_scale']}")
            ax.annotate(label, (r["kl_mean"], r[y_col]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=6.5, color="#333333")

        if has_id:
            ax.axhline(0, color="#aaaaaa", linestyle="--", linewidth=0.8)

        ax.set_xscale("log")
        ax.set_xlabel("KL divergence (mean, log scale)", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(f"{model_name} — KL divergence vs identity",
                     fontsize=12, fontweight="bold")

    else:  # identity
        plot_df = mdf.dropna(subset=["mean_id"]).copy()
        if plot_df.empty:
            print(f"No identity data for {model_name}, skipping")
            continue

        # Sort by phase then method for a consistent x ordering
        plot_df = plot_df.sort_values(["phase", "scope", "wt_scale", "act_scale"])
        plot_df["x"] = range(len(plot_df))
        baseline_id = baselines.get(model_name, np.nan)

        fig, ax = plt.subplots(figsize=(12, 5))
        for phase, marker in PHASE_MARKERS.items():
            sub = plot_df[plot_df["phase"] == phase]
            if sub.empty:
                continue
            ax.scatter(sub["x"], sub["mean_id"],
                       c=sub["fmt_color"], marker=marker,
                       s=28, alpha=0.85, edgecolors="white", linewidths=0.3,
                       zorder=3)

        if pd.notna(baseline_id):
            ax.axhline(baseline_id, color="#aaaaaa", linestyle="--",
                       linewidth=0.8, label=f"fp16 baseline ({baseline_id:.4f})")

        ax.set_xticks([])
        ax.set_xlabel("configs (grouped by phase)", fontsize=11)
        ax.set_ylabel("Mean identity", fontsize=11)
        ax.set_title(f"{model_name} — Mean identity per config",
                     fontsize=12, fontweight="bold")
    ax.grid(True, which="both" if args.metric == "kl" else "major",
            alpha=0.2, linewidth=0.5)

    # Legends outside the plot to the right
    color_patches = [mpatches.Patch(color=FORMAT_COLORS[k], label=k)
                     for k in COLOR_LOOKUP_ORDER]
    shape_handles = [ax.scatter([], [], marker=m, color="#666666", s=40,
                                label=PHASE_LABELS[p])
                     for p, m in PHASE_MARKERS.items()]
    leg1 = ax.legend(handles=color_patches, title="format",
                     loc="upper left", bbox_to_anchor=(1.01, 1),
                     borderaxespad=0, fontsize=8, framealpha=0.85)
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, title="phase",
              loc="upper left", bbox_to_anchor=(1.01, 0.42),
              borderaxespad=0, fontsize=8, framealpha=0.85)

    plt.tight_layout()
    safe = model_name.replace(" ", "_").lower()
    path = os.path.join(OUT, f"scatter_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

print("Done.")
