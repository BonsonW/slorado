#!/usr/bin/env python3
"""
Generate quant configs using calibrated fixed activation scales.

Usage:
    python3 scripts/gen_calib_configs.py \
        --lstm /tmp/calib_lstm.json \
        --tx   /tmp/calib_tx.json

Outputs /tmp/qc_<tag>.json for all phase 1 and phase 2 configs.
Phase 1 (weight-only) configs: identical to gen_configs.py — weights are
static so dynamic scale equals the calibrated scale anyway.
Phase 2 (activation per-tensor): uses "int8_s<float>" with the per-layer
amax from calibration instead of computing amax dynamically each batch.
Phase 2 (activation per-token and fixed): unchanged — per-token dynamic is
still the best approximation, and fixed scales are already hardcoded.
"""

import json
import os
import argparse

OUT = "/tmp"


def write(tag, cfg):
    path = os.path.join(OUT, f"qc_{tag}.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    print(f"  {path}  ({len(cfg)} keys)")


def calib_act_pt(layers, layer_name, fallback="int8_per_tensor"):
    """Return calibrated per-tensor method string, or fallback if not in calib data."""
    layer = layers.get(layer_name)
    if not layer:
        return fallback
    amax = layer.get("input", {}).get("per_tensor_amax")
    if not amax or amax <= 0:
        return fallback
    scale = amax / 127.0
    return f"int8_s{scale:.6g}"


# ── FLSTM helpers ──────────────────────────────────────────────────────────────

def lstm_hh(w_method, a_fn):
    cfg = {}
    for i in range(1, 6):
        k = f"rnns.rnn{i}.hh_fused"
        if w_method:
            cfg[k] = w_method
        a = a_fn(k)
        if a:
            cfg[k + ".act"] = a
    return cfg


def lstm_ih(w_method, a_fn):
    cfg = {}
    for i in range(1, 6):
        k = f"rnns.rnn{i}.ih_fused"
        if w_method:
            cfg[k] = w_method
        a = a_fn(k)
        if a:
            cfg[k + ".act"] = a
    return cfg


# ── Transformer helpers ────────────────────────────────────────────────────────

def tx_all(w_method, a_fn, fc2_a=None):
    cfg = {}
    for i in range(18):
        for key in [f"transformer_encoder.{i}.self_attn.wqkv",
                    f"transformer_encoder.{i}.ff.fc1",
                    f"transformer_encoder.{i}.ff.fc2"]:
            if w_method:
                cfg[key] = w_method
            a = fc2_a if (fc2_a and key.endswith(".fc2")) else a_fn(key)
            if a:
                cfg[key + ".act"] = a
    return cfg


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lstm", help="Calibration JSON for LSTM model")
    parser.add_argument("--tx",   help="Calibration JSON for Transformer model")
    args = parser.parse_args()

    lstm_layers: dict = {}
    tx_layers:   dict = {}

    if args.lstm:
        data = json.load(open(args.lstm))
        lstm_layers = data["layers"]
        print(f"LSTM calibration: {len(lstm_layers)} layers from {args.lstm}")
    else:
        print("No LSTM calibration — per-tensor activation configs will use dynamic scale")

    if args.tx:
        data = json.load(open(args.tx))
        tx_layers = data["layers"]
        print(f"TX calibration:   {len(tx_layers)} layers from {args.tx}")
    else:
        print("No TX calibration  — per-tensor activation configs will use dynamic scale")

    fp16  = lambda _: "fp16"
    dyn_pc = lambda _: "int8_per_channel"
    dyn_pt = lambda _: "int8_per_tensor"
    fixed  = lambda _: "int8_fixed"
    fixed4 = lambda _: "int8_fixed_4"

    def lstm_calib_pt(k):
        return calib_act_pt(lstm_layers, k)

    def tx_calib_pt(k):
        return calib_act_pt(tx_layers, k)

    print("\nFLSTM — Phase 1: weights only (act = fp16)")
    write("lstm_hh_w_pc", lstm_hh("int8_per_channel", fp16))
    write("lstm_hh_w_pt", lstm_hh("int8_per_tensor",  fp16))
    write("lstm_ih_w_pc", lstm_ih("int8_per_channel", fp16))
    write("lstm_ih_w_pt", lstm_ih("int8_per_tensor",  fp16))

    print("FLSTM — Phase 2: activations only (weight absent = fp16)")
    write("lstm_hh_a_ptoken",  lstm_hh(None, dyn_pc))
    write("lstm_hh_a_ptensor", lstm_hh(None, lstm_calib_pt))
    write("lstm_hh_a_fixed",   lstm_hh(None, fixed))
    write("lstm_ih_a_ptoken",  lstm_ih(None, dyn_pc))
    write("lstm_ih_a_ptensor", lstm_ih(None, lstm_calib_pt))

    print("Transformer — Phase 1: weights only (act = fp16)")
    write("tx_w_pc", tx_all("int8_per_channel", fp16))
    write("tx_w_pt", tx_all("int8_per_tensor",  fp16))

    print("Transformer — Phase 2: activations only (weight absent = fp16)")
    write("tx_a_ptoken",  tx_all(None, dyn_pc))
    write("tx_a_ptensor", tx_all(None, tx_calib_pt))
    # fc2 input is post-SiLU (not post-RMSNorm), keep dynamic
    write("tx_a_fixed",   tx_all(None, fixed4, fc2_a="int8_per_channel"))

    print("Done.")


if __name__ == "__main__":
    main()
