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

def lstm_layer(suffix, w_method, a_fn):
    cfg = {}
    for i in range(1, 6):
        k = f"rnns.rnn{i}.{suffix}"
        if w_method:
            cfg[k] = w_method
        a = a_fn(k)
        if a:
            cfg[k + ".act"] = a
    return cfg

def lstm_dn_ih(w_method, a_fn): return lstm_layer("dn_ih", w_method, a_fn)
def lstm_up_ih(w_method, a_fn): return lstm_layer("up_ih", w_method, a_fn)
def lstm_dn_hh(w_method, a_fn): return lstm_layer("dn_hh", w_method, a_fn)
def lstm_up_hh(w_method, a_fn): return lstm_layer("up_hh", w_method, a_fn)


# ── Transformer helpers ────────────────────────────────────────────────────────

def tx_layer(suffix, w_method, a_fn):
    """Quantize one layer type across all 18 encoder blocks."""
    cfg = {}
    for i in range(18):
        key = f"transformer_encoder.{i}.{suffix}"
        if w_method:
            cfg[key] = w_method
        a = a_fn(key)
        if a:
            cfg[key + ".act"] = a
    return cfg


TX_LAYERS = [
    ("wqkv", "self_attn.wqkv"),
    ("op",   "self_attn.out_proj"),
    ("fc1",  "ff.fc1"),
    ("fc2",  "ff.fc2"),
]


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

    fp16      = lambda _: "fp16"
    dyn_pc    = lambda _: "int8_per_channel"
    dyn_pt    = lambda _: "int8_per_tensor"
    fixed     = lambda _: "int8_fixed"
    fp8_fixed = lambda _: "fp8_fixed"
    fixed4    = lambda _: "int8_fixed_4"
    fp8_pc    = lambda _: "fp8_per_channel"
    fp8_pt    = lambda _: "fp8_per_tensor"
    mxint8    = lambda _: "mxint8"
    mxfp4     = lambda _: "mxfp4"
    mxfp6     = lambda _: "mxfp6"
    mxfp8     = lambda _: "mxfp8"

    def lstm_calib_pt(k):
        return calib_act_pt(lstm_layers, k)

    def tx_calib_pt(k):
        return calib_act_pt(tx_layers, k)

    LSTM_LAYERS = [
        ("dn_ih", lstm_dn_ih),
        ("up_ih", lstm_up_ih),
        ("dn_hh", lstm_dn_hh),
        ("up_hh", lstm_up_hh),
    ]

    print("\nFLSTM — Phase 1: weights only (act = fp16)")
    for lname, lfn in LSTM_LAYERS:
        write(f"lstm_{lname}_w_pc",    lfn("int8_per_channel", fp16))
        write(f"lstm_{lname}_w_pt",    lfn("int8_per_tensor",  fp16))
        write(f"lstm_{lname}_w_fp8pc", lfn("fp8_per_channel",  fp16))
        write(f"lstm_{lname}_w_fp8pt", lfn("fp8_per_tensor",   fp16))
        write(f"lstm_{lname}_w_mxint8",lfn("mxint8",           fp16))
        write(f"lstm_{lname}_w_mxfp4", lfn("mxfp4",            fp16))
        write(f"lstm_{lname}_w_mxfp6", lfn("mxfp6",            fp16))
        write(f"lstm_{lname}_w_mxfp8", lfn("mxfp8",            fp16))

    print("FLSTM — Phase 2: activations only (weight absent = fp16)")
    for lname, lfn in LSTM_LAYERS:
        write(f"lstm_{lname}_a_ptoken",    lfn(None, dyn_pc))
        write(f"lstm_{lname}_a_ptensor",   lfn(None, lstm_calib_pt))
        write(f"lstm_{lname}_a_fp8ptoken", lfn(None, fp8_pc))
        write(f"lstm_{lname}_a_mxint8",    lfn(None, mxint8))
        write(f"lstm_{lname}_a_mxfp4",     lfn(None, mxfp4))
        write(f"lstm_{lname}_a_mxfp6",     lfn(None, mxfp6))
        write(f"lstm_{lname}_a_mxfp8",     lfn(None, mxfp8))
        if lname in ("dn_hh", "dn_ih"):
            write(f"lstm_{lname}_a_fixed",    lfn(None, fixed))
            write(f"lstm_{lname}_a_fp8fixed", lfn(None, fp8_fixed))

    print("Transformer — Phase 1: weights only (one layer type at a time)")
    for lname, lsuffix in TX_LAYERS:
        write(f"tx_{lname}_w_pc",    tx_layer(lsuffix, "int8_per_channel", fp16))
        write(f"tx_{lname}_w_pt",    tx_layer(lsuffix, "int8_per_tensor",  fp16))
        write(f"tx_{lname}_w_fp8pc", tx_layer(lsuffix, "fp8_per_channel",  fp16))
        write(f"tx_{lname}_w_fp8pt", tx_layer(lsuffix, "fp8_per_tensor",   fp16))
        write(f"tx_{lname}_w_mxint8",tx_layer(lsuffix, "mxint8",           fp16))
        write(f"tx_{lname}_w_mxfp4", tx_layer(lsuffix, "mxfp4",            fp16))
        write(f"tx_{lname}_w_mxfp6", tx_layer(lsuffix, "mxfp6",            fp16))
        write(f"tx_{lname}_w_mxfp8", tx_layer(lsuffix, "mxfp8",            fp16))

    print("Transformer — Phase 2: activations only (one layer type at a time)")
    for lname, lsuffix in TX_LAYERS:
        write(f"tx_{lname}_a_ptoken",    tx_layer(lsuffix, None, dyn_pc))
        write(f"tx_{lname}_a_fp8ptoken", tx_layer(lsuffix, None, fp8_pc))
        write(f"tx_{lname}_a_mxint8",    tx_layer(lsuffix, None, mxint8))
        write(f"tx_{lname}_a_mxfp4",     tx_layer(lsuffix, None, mxfp4))
        write(f"tx_{lname}_a_mxfp6",     tx_layer(lsuffix, None, mxfp6))
        write(f"tx_{lname}_a_mxfp8",     tx_layer(lsuffix, None, mxfp8))
        if lname in ("wqkv", "fc1"):
            # post-RMSNorm activations can use a fixed scale
            write(f"tx_{lname}_a_fixed", tx_layer(lsuffix, None, fixed4))

    print("Done.")


if __name__ == "__main__":
    main()
