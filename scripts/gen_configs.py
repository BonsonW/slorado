#!/usr/bin/env python3
"""Generate quant config JSONs for isolated weight / activation sensitivity study."""

import json
import os

OUT = "/tmp"

def write(tag, cfg):
    path = os.path.join(OUT, f"qc_{tag}.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    print(f"  {path}  ({len(cfg)} keys)")

# ── FLSTM helpers ──────────────────────────────────────────────────────────────
# Each FLSTM layer has 4 quantizable weight matrices:
#   dn_ih: down-projection for input x  [K, C]
#   up_ih: up-projection for input x    [4*C, K]
#   dn_hh: down-projection for hidden h [K, C]   — activation is hh[t] ∈ [-1,1]
#   up_hh: up-projection for hidden h   [4*C, K]

def lstm_layer(suffix, w_method, a_method=None):
    cfg = {}
    for i in range(1, 6):
        k = f"rnns.rnn{i}.{suffix}"
        if w_method:
            cfg[k] = w_method
        if a_method:
            cfg[k + ".act"] = a_method
    return cfg

def lstm_dn_ih(w_method, a_method=None): return lstm_layer("dn_ih", w_method, a_method)
def lstm_up_ih(w_method, a_method=None): return lstm_layer("up_ih", w_method, a_method)
def lstm_dn_hh(w_method, a_method=None): return lstm_layer("dn_hh", w_method, a_method)
def lstm_up_hh(w_method, a_method=None): return lstm_layer("up_hh", w_method, a_method)

# ── Transformer helpers ────────────────────────────────────────────────────────
# Each TX encoder layer has 4 quantizable weight matrices:
#   wqkv:  self_attn.wqkv       — activation is post-RMSNorm, use int8_fixed_4
#   op:    self_attn.out_proj
#   fc1:   ff.fc1               — activation is post-RMSNorm, use int8_fixed_4
#   fc2:   ff.fc2               — activation is post-SiLU, not bounded

TX_SCOPE_KEYS = {
    "wqkv": "self_attn.wqkv",
    "op":   "self_attn.out_proj",
    "fc1":  "ff.fc1",
    "fc2":  "ff.fc2",
}

def tx_layer(scope, w_method, a_method=None):
    full = TX_SCOPE_KEYS[scope]
    cfg = {}
    for i in range(18):
        k = f"transformer_encoder.{i}.{full}"
        if w_method:
            cfg[k] = w_method
        if a_method:
            cfg[k + ".act"] = a_method
    return cfg

def tx_wqkv(w_method, a_method=None): return tx_layer("wqkv", w_method, a_method)
def tx_op  (w_method, a_method=None): return tx_layer("op",   w_method, a_method)
def tx_fc1 (w_method, a_method=None): return tx_layer("fc1",  w_method, a_method)
def tx_fc2 (w_method, a_method=None): return tx_layer("fc2",  w_method, a_method)

# ──────────────────────────────────────────────────────────────────────────────

print("FLSTM — Phase 1: weights only (act = fp16)")
for lname, lfn in [("dn_ih", lstm_dn_ih), ("up_ih", lstm_up_ih),
                    ("dn_hh", lstm_dn_hh), ("up_hh", lstm_up_hh)]:
    write(f"lstm_{lname}_w_pc",    lfn("int8_per_channel", "fp16"))
    write(f"lstm_{lname}_w_pt",    lfn("int8_per_tensor",  "fp16"))
    write(f"lstm_{lname}_w_fp8pc", lfn("fp8_per_channel",  "fp16"))
    write(f"lstm_{lname}_w_fp8pt", lfn("fp8_per_tensor",   "fp16"))
    write(f"lstm_{lname}_w_int4pc", lfn("int4_per_channel", "fp16"))
    write(f"lstm_{lname}_w_int4pt", lfn("int4_per_tensor",  "fp16"))

print("FLSTM — Phase 2: activations only (weight key absent = fp16)")
for lname, lfn in [("dn_ih", lstm_dn_ih), ("up_ih", lstm_up_ih),
                    ("dn_hh", lstm_dn_hh), ("up_hh", lstm_up_hh)]:
    write(f"lstm_{lname}_a_ptoken",     lfn(None, "int8_per_channel"))
    write(f"lstm_{lname}_a_fp8ptoken",  lfn(None, "fp8_per_channel"))
    write(f"lstm_{lname}_a_int4ptoken", lfn(None, "int4_per_channel"))
    if lname == "dn_hh":
        write("lstm_dn_hh_a_fixed",    lfn(None, "int8_fixed"))   # hh[t] ∈ [-1,1], scale=1/127
        write("lstm_dn_hh_a_fp8fixed", lfn(None, "fp8_fixed"))    # hh[t] ∈ [-1,1], scale=1/448

print("Transformer — Phase 1: weights only (act = fp16)")
for lname, lfn in [("wqkv", tx_wqkv), ("op", tx_op), ("fc1", tx_fc1), ("fc2", tx_fc2)]:
    write(f"tx_{lname}_w_pc",     lfn("int8_per_channel", "fp16"))
    write(f"tx_{lname}_w_pt",     lfn("int8_per_tensor",  "fp16"))
    write(f"tx_{lname}_w_fp8pc",  lfn("fp8_per_channel",  "fp16"))
    write(f"tx_{lname}_w_fp8pt",  lfn("fp8_per_tensor",   "fp16"))
    write(f"tx_{lname}_w_int4pc", lfn("int4_per_channel", "fp16"))
    write(f"tx_{lname}_w_int4pt", lfn("int4_per_tensor",  "fp16"))

print("Transformer — Phase 2: activations only")
for lname, lfn in [("wqkv", tx_wqkv), ("op", tx_op), ("fc1", tx_fc1), ("fc2", tx_fc2)]:
    write(f"tx_{lname}_a_ptoken",     lfn(None, "int8_per_channel"))
    write(f"tx_{lname}_a_fp8ptoken",  lfn(None, "fp8_per_channel"))
    write(f"tx_{lname}_a_int4ptoken", lfn(None, "int4_per_channel"))
    if lname in ("wqkv", "fc1"):
        write(f"tx_{lname}_a_fixed", lfn(None, "int8_fixed_4"))   # post-RMSNorm, scale=4/127

print("FLSTM — MX: weights + activations (matched pairs, OCP group-32 microscaling)")
for lname, lfn in [("dn_ih", lstm_dn_ih), ("up_ih", lstm_up_ih),
                    ("dn_hh", lstm_dn_hh), ("up_hh", lstm_up_hh)]:
    write(f"lstm_{lname}_mxint8", lfn("mxint8", "mxint8"))
    write(f"lstm_{lname}_mxfp4",  lfn("mxfp4",  "mxfp4"))
    write(f"lstm_{lname}_mxfp6",  lfn("mxfp6",  "mxfp6"))
    write(f"lstm_{lname}_mxfp8",  lfn("mxfp8",  "mxfp8"))

print("Transformer — MX: weights + activations (matched pairs)")
for lname, lfn in [("wqkv", tx_wqkv), ("op", tx_op), ("fc1", tx_fc1), ("fc2", tx_fc2)]:
    write(f"tx_{lname}_mxint8", lfn("mxint8", "mxint8"))
    write(f"tx_{lname}_mxfp4",  lfn("mxfp4",  "mxfp4"))
    write(f"tx_{lname}_mxfp6",  lfn("mxfp6",  "mxfp6"))
    write(f"tx_{lname}_mxfp8",  lfn("mxfp8",  "mxfp8"))

print("Done.")
