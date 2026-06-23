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

def lstm_hh(w_method, a_method=None):
    cfg = {}
    for i in range(1, 6):
        k = f"rnns.rnn{i}.hh_fused"
        if w_method:
            cfg[k] = w_method
        if a_method:
            cfg[k + ".act"] = a_method
    return cfg

def lstm_ih(w_method, a_method=None):
    cfg = {}
    for i in range(1, 6):
        k = f"rnns.rnn{i}.ih_fused"
        if w_method:
            cfg[k] = w_method
        if a_method:
            cfg[k + ".act"] = a_method
    return cfg

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
write("lstm_hh_w_pc",    lstm_hh("int8_per_channel",  "fp16"))
write("lstm_hh_w_pt",    lstm_hh("int8_per_tensor",   "fp16"))
write("lstm_hh_w_fp8pc", lstm_hh("fp8_per_channel",   "fp16"))
write("lstm_hh_w_fp8pt", lstm_hh("fp8_per_tensor",    "fp16"))
write("lstm_ih_w_pc",    lstm_ih("int8_per_channel",  "fp16"))
write("lstm_ih_w_pt",    lstm_ih("int8_per_tensor",   "fp16"))
write("lstm_ih_w_fp8pc", lstm_ih("fp8_per_channel",   "fp16"))
write("lstm_ih_w_fp8pt", lstm_ih("fp8_per_tensor",    "fp16"))

print("FLSTM — Phase 2: activations only (weight key absent = fp16)")
write("lstm_hh_a_ptoken",    lstm_hh(None, "int8_per_channel"))
write("lstm_hh_a_fixed",     lstm_hh(None, "int8_fixed"))       # hh[t] ∈ [-1,1]
write("lstm_hh_a_fp8ptoken", lstm_hh(None, "fp8_per_channel"))
write("lstm_ih_a_ptoken",    lstm_ih(None, "int8_per_channel"))
write("lstm_ih_a_fp8ptoken", lstm_ih(None, "fp8_per_channel"))

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
write("lstm_hh_w_mxint8", lstm_hh("mxint8", "fp16"))
write("lstm_hh_w_mxfp4",  lstm_hh("mxfp4",  "fp16"))
write("lstm_hh_w_mxfp6", lstm_hh("mxfp6", "fp16"))
write("lstm_hh_w_mxfp8", lstm_hh("mxfp8", "fp16"))
write("lstm_ih_w_mxint8", lstm_ih("mxint8", "fp16"))
write("lstm_ih_w_mxfp4",  lstm_ih("mxfp4",  "fp16"))
write("lstm_ih_w_mxfp6", lstm_ih("mxfp6", "fp16"))
write("lstm_ih_w_mxfp8", lstm_ih("mxfp8", "fp16"))

print("FLSTM — Phase 2 MX: activations only")
write("lstm_hh_a_mxint8", lstm_hh(None, "mxint8"))
write("lstm_hh_a_mxfp4",  lstm_hh(None, "mxfp4"))
write("lstm_hh_a_mxfp6", lstm_hh(None, "mxfp6"))
write("lstm_hh_a_mxfp8", lstm_hh(None, "mxfp8"))
write("lstm_ih_a_mxint8", lstm_ih(None, "mxint8"))
write("lstm_ih_a_mxfp4",  lstm_ih(None, "mxfp4"))
write("lstm_ih_a_mxfp6", lstm_ih(None, "mxfp6"))
write("lstm_ih_a_mxfp8", lstm_ih(None, "mxfp8"))

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
