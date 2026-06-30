#!/usr/bin/env python3
"""Plot per-layer calibration stats from weight and activation JSON files.

Usage:
  python scripts/plot_calibration.py

Reads:
  scripts/results/calib_weights_{lstm,tx}.json  (from calib_weights.py)
  scripts/results/calib_acts_{lstm,tx}.json      (from slorado --calibrate)

Outputs (4 figures):
  scripts/calib_{lstm,tx}_acts.png     activation amax + token uniformity ratio
  scripts/calib_{lstm,tx}_weights.png  weight amax + channel uniformity ratio
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

RESULTS = os.path.join(os.path.dirname(__file__), "results")
OUT     = os.path.dirname(__file__)

ACT_COLOR  = "#2980b9"   # blue   — activations
WT_COLOR   = "#e67e22"   # orange — weights
TOK_COLOR  = "#27ae60"   # green  — token uniformity
CH_COLOR   = "#8e44ad"   # purple — channel uniformity

# ── Data loading ──────────────────────────────────────────────────────────────

def load_json(path):
    if os.path.exists(path):
        return json.load(open(path))["layers"]
    return {}

def merge_layers(wt_path, act_path):
    wt  = load_json(wt_path)
    act = load_json(act_path)
    merged = {}
    for name in set(wt) | set(act):
        merged[name] = {
            "weight": wt.get(name,  {}).get("weight", {}),
            "input":  act.get(name, {}).get("input",  {}),
        }
    return merged

# ── Parse helpers ─────────────────────────────────────────────────────────────

def parse_lstm(wt_path, act_path):
    """scope → {rnn_idx → stats}"""
    layers = merge_layers(wt_path, act_path)
    data = {}
    for key, v in layers.items():
        parts   = key.split(".")
        rnn_idx = int(parts[1].replace("rnn", ""))
        scope   = parts[2]
        tok = v["input"].get("per_token_amax", {})
        ch  = v["weight"].get("per_out_channel_amax", {})
        data.setdefault(scope, {})[rnn_idx] = {
            "amax":      v["input"].get("per_tensor_amax", 0),
            "tok_p25":   tok.get("p25", 0),
            "tok_p50":   tok.get("p50", 0),
            "tok_p75":   tok.get("p75", 0),
            "wt_amax":   v["weight"].get("per_tensor_amax", 0),
            "ch_p25":    ch.get("p25", 0),
            "ch_p50":    ch.get("p50", 0),
            "ch_p75":    ch.get("p75", 0),
        }
    return data

def parse_tx(wt_path, act_path):
    """scope → {layer_idx → stats}"""
    layers = merge_layers(wt_path, act_path)
    SCOPE_MAP = {
        "ff.fc1":             "fc1",
        "ff.fc2":             "fc2",
        "self_attn.out_proj": "op",
        "self_attn.wqkv":     "wqkv",
    }
    data = {}
    for key, v in layers.items():
        parts     = key.split(".", 2)
        layer_idx = int(parts[1])
        scope     = SCOPE_MAP.get(parts[2], parts[2])
        tok = v["input"].get("per_token_amax", {})
        ch  = v["weight"].get("per_out_channel_amax", {})
        data.setdefault(scope, {})[layer_idx] = {
            "amax":    v["input"].get("per_tensor_amax", 0),
            "tok_p25": tok.get("p25", 0),
            "tok_p50": tok.get("p50", 0),
            "tok_p75": tok.get("p75", 0),
            "wt_amax": v["weight"].get("per_tensor_amax", 0),
            "ch_p25":  ch.get("p25", 0),
            "ch_p50":  ch.get("p50", 0),
            "ch_p75":  ch.get("p75", 0),
        }
    return data

# ── Subplot drawers ───────────────────────────────────────────────────────────

def draw_acts(ax, scope_data, scope_name):
    idxs = sorted(scope_data.keys())
    xs   = np.arange(len(idxs))
    amax = np.array([scope_data[i]["amax"] for i in idxs])

    ax.scatter(xs, amax, color=ACT_COLOR, s=30, marker="D", zorder=4)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("act amax", fontsize=8, color=ACT_COLOR)
    ax.tick_params(axis="y", labelcolor=ACT_COLOR, labelsize=7)
    ax.set_title(scope_name, fontsize=10, fontweight="bold")
    ax.set_xlabel("layer", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(idxs, rotation=90, fontsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    ax2 = ax.twinx()
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("token uniformity\n(p50/amax)", fontsize=8, color=TOK_COLOR)
    ax2.tick_params(axis="y", labelcolor=TOK_COLOR, labelsize=7)

    has_tok = np.array([scope_data[i]["tok_p50"] for i in idxs]).max() > 0
    if has_tok:
        denom   = np.where(amax > 0, amax, 1)
        p25     = np.array([scope_data[i]["tok_p25"] for i in idxs]) / denom
        p50     = np.array([scope_data[i]["tok_p50"] for i in idxs]) / denom
        p75     = np.array([scope_data[i]["tok_p75"] for i in idxs]) / denom
        ax2.fill_between(xs, p25, p75, color=TOK_COLOR, alpha=0.18, zorder=2)
        ax2.scatter(xs, p50, color=TOK_COLOR, s=22, marker="o", zorder=3)


def draw_weights(ax, scope_data, scope_name):
    idxs   = sorted(scope_data.keys())
    xs     = np.arange(len(idxs))
    wt_max = np.array([scope_data[i]["wt_amax"] for i in idxs])

    ax.scatter(xs, wt_max, color=WT_COLOR, s=30, marker="s", zorder=4)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("wt amax", fontsize=8, color=WT_COLOR)
    ax.tick_params(axis="y", labelcolor=WT_COLOR, labelsize=7)
    ax.set_title(scope_name, fontsize=10, fontweight="bold")
    ax.set_xlabel("layer", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(idxs, rotation=90, fontsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    ax2 = ax.twinx()
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("channel uniformity\n(p50/amax)", fontsize=8, color=CH_COLOR)
    ax2.tick_params(axis="y", labelcolor=CH_COLOR, labelsize=7)

    has_ch = np.array([scope_data[i]["ch_p50"] for i in idxs]).max() > 0
    if has_ch:
        denom = np.where(wt_max > 0, wt_max, 1)
        p25   = np.array([scope_data[i]["ch_p25"] for i in idxs]) / denom
        p50   = np.array([scope_data[i]["ch_p50"] for i in idxs]) / denom
        p75   = np.array([scope_data[i]["ch_p75"] for i in idxs]) / denom
        ax2.fill_between(xs, p25, p75, color=CH_COLOR, alpha=0.18, zorder=2)
        ax2.scatter(xs, p50, color=CH_COLOR, s=22, marker="^", zorder=3)

# ── Figure builder ────────────────────────────────────────────────────────────

def make_figures(data, scopes, model_label, out_prefix, figsize_acts, figsize_wts):
    act_legend = [
        mlines.Line2D([], [], color=ACT_COLOR, lw=0, marker="D", ms=6, label="act amax (left)"),
        mlines.Line2D([], [], color=TOK_COLOR, lw=0, marker="o", ms=6, label="token uniformity p50/amax (right, band=p25-p75)"),
    ]
    wt_legend = [
        mlines.Line2D([], [], color=WT_COLOR,  lw=0, marker="s", ms=6, label="wt amax (left)"),
        mlines.Line2D([], [], color=CH_COLOR,  lw=0, marker="^", ms=6, label="channel uniformity p50/amax (right, band=p25-p75)"),
    ]

    # activation figure
    fig, axes = plt.subplots(1, len(scopes), figsize=figsize_acts, layout="constrained")
    fig.suptitle(f"{model_label} — Activations", fontsize=12, fontweight="bold")
    for ax, scope in zip(axes, scopes):
        if scope in data:
            draw_acts(ax, data[scope], scope)
    fig.legend(handles=act_legend, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.1), framealpha=0.9)
    path = os.path.join(OUT, f"{out_prefix}_acts.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    # weight figure
    fig, axes = plt.subplots(1, len(scopes), figsize=figsize_wts, layout="constrained")
    fig.suptitle(f"{model_label} — Weights", fontsize=12, fontweight="bold")
    for ax, scope in zip(axes, scopes):
        if scope in data:
            draw_weights(ax, data[scope], scope)
    fig.legend(handles=wt_legend, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.1), framealpha=0.9)
    path = os.path.join(OUT, f"{out_prefix}_weights.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

wt_lstm  = os.path.join(RESULTS, "calib_weights_lstm.json")
act_lstm = os.path.join(RESULTS, "calib_acts_lstm.json")
if os.path.exists(wt_lstm) or os.path.exists(act_lstm):
    data = parse_lstm(wt_lstm, act_lstm)
    make_figures(data, ["dn_ih", "up_ih", "dn_hh", "up_hh"],
                 "HAC v6 (FLSTM)", "calib_lstm",
                 figsize_acts=(14, 4), figsize_wts=(14, 4))
else:
    print(f"Skipping LSTM: neither {wt_lstm} nor {act_lstm} found")

wt_tx  = os.path.join(RESULTS, "calib_weights_tx.json")
act_tx = os.path.join(RESULTS, "calib_acts_tx.json")
if os.path.exists(wt_tx) or os.path.exists(act_tx):
    data = parse_tx(wt_tx, act_tx)
    make_figures(data, ["wqkv", "op", "fc1", "fc2"],
                 "SUP v5 (Transformer)", "calib_tx",
                 figsize_acts=(16, 4.5), figsize_wts=(16, 4.5))
else:
    print(f"Skipping TX: neither {wt_tx} nor {act_tx} found")

print("Done.")
