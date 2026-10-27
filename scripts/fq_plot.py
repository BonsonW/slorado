#!/usr/bin/env python3
"""Per-read identity density for each fake-quant run in a fq_sweep.sh output dir.

usage: fq_plot.py <sweep outdir> [out.png]
One panel per run (format x layers), fp16 baseline ("none") in grey on every panel.
Also writes <outdir>/summary.tsv (mapped reads, mean/median identity, median delta vs fp16).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

FORMAT_ORDER = ["int8", "mxint8", "fp8_e4m3", "mxfp8_e4m3", "fp8_e5m2", "mxfp8_e5m2",
                "mxfp6_e2m3", "mxfp6_e3m2", "int4", "mxfp4"]
INK, MUTED, GRID, SERIES, BASE = "#1a1a19", "#6b6a63", "#e6e5e0", "#2a78d6", "#a3a29a"


def load(d):
    ident = np.loadtxt(os.path.join(d, "identity.tsv"), skiprows=1, usecols=1, ndmin=1)
    info = dict(zip(*[l.rstrip("\n").split("\t") for l in open(os.path.join(d, "info.tsv"))]))
    return ident, info


def main():
    outdir = sys.argv[1]
    png = sys.argv[2] if len(sys.argv) > 2 else os.path.join(outdir, "identity_density.png")
    runs = {d: load(os.path.join(outdir, d)) for d in sorted(os.listdir(outdir))
            if os.path.isfile(os.path.join(outdir, d, "identity.tsv"))}
    base, base_info = runs.pop("none")
    order = lambda d: (runs[d][1]["layers"], FORMAT_ORDER.index(runs[d][1]["format"])
                       if runs[d][1]["format"] in FORMAT_ORDER else 99)
    names = sorted(runs, key=order)

    with open(os.path.join(outdir, "summary.tsv"), "w") as f:
        f.write("run\tformat\tlayers\ttarget\tn_reads\tn_mapped\tmean\tmedian\tdelta_median_vs_fp16\n")
        for d, (x, info) in [("none", (base, base_info))] + [(d, runs[d]) for d in names]:
            f.write(f"{d}\t{info['format']}\t{info['layers']}\t{info['target']}\t{info['n_reads']}\t{len(x)}\t"
                    f"{x.mean():.5f}\t{np.median(x):.5f}\t{np.median(x) - np.median(base):+.5f}\n")

    lo = np.percentile(np.concatenate([base] + [runs[d][0] for d in names]), 1)
    grid = np.linspace(lo, 1.0, 400)
    base_kde = gaussian_kde(base)(grid)

    ncol = min(5, max(1, len(names)))
    nrow = max(1, -(-len(names) // ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), sharex=True, sharey=True,
                             squeeze=False, facecolor="#fcfcfb")
    for ax, d in zip(axes.flat, names):
        x, info = runs[d]
        ax.set_facecolor("#fcfcfb")
        ax.plot(grid, base_kde, color=BASE, lw=1.5, label="fp16")
        ax.plot(grid, gaussian_kde(x)(grid), color=SERIES, lw=2, label=info["format"])
        title = info["format"] if info["layers"] == "*" else f"{info['format']}  [{info['layers']}]"
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.text(0.03, 0.95, f"median {np.median(x):.4f}\nΔ {np.median(x) - np.median(base):+.4f}\nmapped {len(x)}",
                transform=ax.transAxes, va="top", fontsize=7, color=MUTED)
        ax.grid(color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=7, labelbottom=True)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("per-read identity", fontsize=8, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("density", fontsize=8, color=INK)
    handles = [plt.Line2D([], [], color=SERIES, lw=2), plt.Line2D([], [], color=BASE, lw=1.5)]
    fig.legend(handles, ["fake-quant format (panel title)", f"fp16 baseline (median {np.median(base):.4f}, mapped {len(base)})"],
               loc="upper right", fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle(f"Per-read identity, target={runs[names[0]][1]['target']}", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(png, dpi=150)
    print(f"wrote {png} and {os.path.join(outdir, 'summary.tsv')}")


if __name__ == "__main__":
    main()
