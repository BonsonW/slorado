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
#   dn_hh: down-projection for hidden h [K, C]
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

def tx_all(w_method, a_method=None, fc2_a_method=None):
    """wqkv + fc1 + fc2. fc2_a_method overrides fc2 activation independently."""
    cfg = {}
    for i in range(18):
        for key in [f"transformer_encoder.{i}.self_attn.wqkv",
                    f"transformer_encoder.{i}.ff.fc1",
                    f"transformer_encoder.{i}.ff.fc2"]:
            if w_method:
                cfg[key] = w_method
            act = a_method
            if fc2_a_method is not None and key.endswith(".fc2"):
                act = fc2_a_method
            if act:
                cfg[key + ".act"] = act
    return cfg

# ──────────────────────────────────────────────────────────────────────────────
print("FLSTM — Phase 1: weights only (act = fp16)")
for lname, lfn in [("dn_ih", lstm_dn_ih), ("up_ih", lstm_up_ih),
                    ("dn_hh", lstm_dn_hh), ("up_hh", lstm_up_hh)]:
    write(f"lstm_{lname}_w_pc",    lfn("int8_per_channel", "fp16"))
    write(f"lstm_{lname}_w_pt",    lfn("int8_per_tensor",  "fp16"))
    write(f"lstm_{lname}_w_fp8pc", lfn("fp8_per_channel",  "fp16"))
    write(f"lstm_{lname}_w_fp8pt", lfn("fp8_per_tensor",   "fp16"))

print("FLSTM — Phase 2: activations only (weight key absent = fp16)")
for lname, lfn in [("dn_ih", lstm_dn_ih), ("up_ih", lstm_up_ih),
                    ("dn_hh", lstm_dn_hh), ("up_hh", lstm_up_hh)]:
    write(f"lstm_{lname}_a_ptoken",    lfn(None, "int8_per_channel"))
    write(f"lstm_{lname}_a_fp8ptoken", lfn(None, "fp8_per_channel"))
    if lname == "dn_hh":
        write(f"lstm_{lname}_a_fixed", lfn(None, "int8_fixed"))  # hh[t] ∈ [-1,1]

print("Transformer — Phase 1: weights only (act = fp16)")
write("tx_w_pc",    tx_all("int8_per_channel", "fp16"))
write("tx_w_pt",    tx_all("int8_per_tensor",  "fp16"))
write("tx_w_fp8pc", tx_all("fp8_per_channel",  "fp16"))
write("tx_w_fp8pt", tx_all("fp8_per_tensor",   "fp16"))

print("Transformer — Phase 2: activations only (weight key absent = fp16)")
write("tx_a_ptoken",    tx_all(None, "int8_per_channel"))
write("tx_a_fp8ptoken", tx_all(None, "fp8_per_channel"))
# fc2 input is post-SiLU (not post-RMSNorm) so keep it dynamic
write("tx_a_fixed",     tx_all(None, "int8_fixed_4", fc2_a_method="int8_per_channel"))

print("FLSTM — Phase 1 MX: weights only (OCP group-32 microscaling)")
for lname, lfn in [("dn_ih", lstm_dn_ih), ("up_ih", lstm_up_ih),
                    ("dn_hh", lstm_dn_hh), ("up_hh", lstm_up_hh)]:
    write(f"lstm_{lname}_w_mxint8", lfn("mxint8", "fp16"))
    write(f"lstm_{lname}_w_mxfp4",  lfn("mxfp4",  "fp16"))
    write(f"lstm_{lname}_w_mxfp6",  lfn("mxfp6",  "fp16"))
    write(f"lstm_{lname}_w_mxfp8",  lfn("mxfp8",  "fp16"))

print("FLSTM — Phase 2 MX: activations only")
for lname, lfn in [("dn_ih", lstm_dn_ih), ("up_ih", lstm_up_ih),
                    ("dn_hh", lstm_dn_hh), ("up_hh", lstm_up_hh)]:
    write(f"lstm_{lname}_a_mxint8", lfn(None, "mxint8"))
    write(f"lstm_{lname}_a_mxfp4",  lfn(None, "mxfp4"))
    write(f"lstm_{lname}_a_mxfp6",  lfn(None, "mxfp6"))
    write(f"lstm_{lname}_a_mxfp8",  lfn(None, "mxfp8"))

print("Transformer — Phase 1 MX: weights only")
write("tx_w_mxint8", tx_all("mxint8", "fp16"))
write("tx_w_mxfp4",  tx_all("mxfp4",  "fp16"))
write("tx_w_mxfp6", tx_all("mxfp6", "fp16"))
write("tx_w_mxfp8", tx_all("mxfp8", "fp16"))

print("Transformer — Phase 2 MX: activations only")
write("tx_a_mxint8", tx_all(None, "mxint8"))
write("tx_a_mxfp4",  tx_all(None, "mxfp4"))
write("tx_a_mxfp6", tx_all(None, "mxfp6"))
write("tx_a_mxfp8", tx_all(None, "mxfp8"))

print("Done.")
