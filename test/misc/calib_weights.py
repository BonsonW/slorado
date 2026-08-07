#!/usr/bin/env python3
"""Compute per-layer weight calibration stats from a model directory.

Reads .tensor weight files directly — no inference needed.
Output JSON is compatible with the weight section of calib.cpp's save_json.

Usage:
  python scripts/calib_weights.py <model_dir> <output.json>

Supports:
  - FLSTM models (hac@v6 style): N.rnn.{dn,up}_weight_{ih,hh}.tensor
  - Transformer models (sup@v5 style): transformer_encoder.N.*.weight.tensor
"""

import argparse
import io
import json
import os
import pickle
import struct
import sys
import zipfile

import numpy as np


STORAGE_DTYPE = {
    "FloatStorage":    np.float32,
    "HalfStorage":     np.float16,
    "BFloat16Storage": np.float16,
    "DoubleStorage":   np.float64,
}


def load_tensor(path):
    """Load a single .tensor file and return a float32 numpy array."""
    name = os.path.basename(path).replace(".tensor", "")
    with zipfile.ZipFile(path) as z:
        raw       = z.read(f"{name}/data/0")
        byteorder = z.read(f"{name}/byteorder").strip()
        pkl_bytes = z.read(f"{name}/data.pkl")

    dtype_box = [np.float32]
    shape_box  = [None]

    class _FakeType:
        def __init__(self, *a, **kw): pass
        def __new__(cls, *a, **kw): return object.__new__(cls)

    class _Loader(pickle.Unpickler):
        def persistent_load(self, pid):
            if isinstance(pid, tuple) and pid[0] == "storage":
                storage_cls = pid[1]
                cls_name = getattr(storage_cls, "__name__", None) or str(storage_cls)
                dtype_box[0] = STORAGE_DTYPE.get(cls_name, np.float32)
            return None

        def find_class(self, module, name):
            if module == "torch" and name.endswith("Storage"):
                dtype_box[0] = STORAGE_DTYPE.get(name, np.float32)
                return _FakeType
            if module.startswith("__torch__"):
                return _FakeType
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                def rebuild(storage, offset, shape, strides, rg, *rest):
                    shape_box[0] = shape
                    return None
                return rebuild
            if module == "collections" and name == "OrderedDict":
                return dict
            return _FakeType

    _Loader(io.BytesIO(pkl_bytes)).load()

    order = "<" if byteorder == b"little" else ">"
    dt    = np.dtype(f"{order}{np.dtype(dtype_box[0]).str[1:]}")
    arr   = np.frombuffer(raw, dtype=dt).astype(np.float32)
    if shape_box[0]:
        arr = arr.reshape(shape_box[0])
    return arr


def weight_stats(W):
    """Return dict of weight stats matching calib.cpp JSON format."""
    W_abs  = np.abs(W)
    amax   = float(W_abs.max())
    # per output-channel amax: max |value| along each row
    if W.ndim >= 2:
        ch_amax = W_abs.max(axis=tuple(range(1, W.ndim)))  # (out_features,)
        ps = np.percentile(ch_amax, [25, 50, 75, 99]).tolist()
        ch_block = {
            "p25": ps[0], "p50": ps[1], "p75": ps[2], "p99": ps[3],
            "max": float(ch_amax.max()),
        }
    else:
        ch_block = None

    result = {
        "per_tensor_range": float(W.max() - W.min()),
        "per_tensor_amax":  amax,
    }
    if ch_block:
        result["per_out_channel_amax"] = ch_block
    return result


def discover_lstm(model_dir):
    """Return list of (layer_name, tensor_path) for FLSTM model dirs."""
    SCOPE = {
        "dn_weight_ih": "dn_ih",
        "dn_weight_hh": "dn_hh",
        "up_weight_ih": "up_ih",
        "up_weight_hh": "up_hh",
    }
    # collect rnn file indices
    rnn_indices = sorted({
        int(f.split(".")[0])
        for f in os.listdir(model_dir)
        if f.endswith(".rnn.dn_weight_ih.tensor")
    })
    layers = []
    for enum_i, file_idx in enumerate(rnn_indices, start=1):
        for file_scope, layer_scope in SCOPE.items():
            fname = f"{file_idx}.rnn.{file_scope}.tensor"
            fpath = os.path.join(model_dir, fname)
            if os.path.exists(fpath):
                layers.append((f"rnns.rnn{enum_i}.{layer_scope}", fpath))
    return layers


def discover_tx(model_dir):
    """Return list of (layer_name, tensor_path) for Transformer model dirs."""
    SCOPE = {
        "ff.fc1.weight":          "ff.fc1",
        "ff.fc2.weight":          "ff.fc2",
        "self_attn.out_proj.weight": "self_attn.out_proj",
        "self_attn.Wqkv.weight":  "self_attn.wqkv",
    }
    layers = []
    layer_indices = sorted({
        int(f.split(".")[1])
        for f in os.listdir(model_dir)
        if f.startswith("transformer_encoder.") and f.endswith(".weight.tensor")
    })
    for idx in layer_indices:
        for file_suffix, layer_suffix in SCOPE.items():
            fname = f"transformer_encoder.{idx}.{file_suffix}.tensor"
            fpath = os.path.join(model_dir, fname)
            if os.path.exists(fpath):
                layers.append((f"transformer_encoder.{idx}.{layer_suffix}", fpath))
    return layers


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", help="path to model directory containing .tensor files")
    ap.add_argument("output",    help="output JSON file path")
    args = ap.parse_args()

    model_dir = args.model_dir
    files = os.listdir(model_dir)

    # auto-detect model type
    is_lstm = any(f.endswith(".rnn.dn_weight_ih.tensor") for f in files)
    is_tx   = any(f.startswith("transformer_encoder.") and f.endswith(".weight.tensor") for f in files)

    if is_lstm:
        layers = discover_lstm(model_dir)
        print(f"[calib_weights] FLSTM model: {len(layers)} weight layers")
    elif is_tx:
        layers = discover_tx(model_dir)
        print(f"[calib_weights] Transformer model: {len(layers)} weight layers")
    else:
        print("Error: could not detect model type in", model_dir, file=sys.stderr)
        sys.exit(1)

    out = {"layers": {}}
    for layer_name, tensor_path in layers:
        W = load_tensor(tensor_path)
        stats = weight_stats(W)
        out["layers"][layer_name] = {
            "n_batches":   0,
            "out_features": W.shape[0],
            "in_features":  W.shape[1] if W.ndim >= 2 else 0,
            "seq_len":      0,
            "weight":       stats,
            "input":        {},
        }
        print(f"  {layer_name}: shape={W.shape}, amax={stats['per_tensor_amax']:.4f}")

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[calib_weights] saved {len(out['layers'])} layers to {args.output}")


if __name__ == "__main__":
    main()
