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
write("lstm_hh_w_pc", lstm_hh("int8_per_channel", "fp16"))
write("lstm_hh_w_pt", lstm_hh("int8_per_tensor",  "fp16"))
write("lstm_ih_w_pc", lstm_ih("int8_per_channel", "fp16"))
write("lstm_ih_w_pt", lstm_ih("int8_per_tensor",  "fp16"))

print("FLSTM — Phase 2: activations only (weight key absent = fp16)")
write("lstm_hh_a_ptoken",  lstm_hh(None, "int8_per_channel"))
write("lstm_hh_a_ptensor", lstm_hh(None, "int8_per_tensor"))
write("lstm_hh_a_fixed",   lstm_hh(None, "int8_fixed"))       # hh[t] ∈ [-1,1]
write("lstm_ih_a_ptoken",  lstm_ih(None, "int8_per_channel"))
write("lstm_ih_a_ptensor", lstm_ih(None, "int8_per_tensor"))

print("Transformer — Phase 1: weights only (act = fp16)")
write("tx_w_pc", tx_all("int8_per_channel", "fp16"))
write("tx_w_pt", tx_all("int8_per_tensor",  "fp16"))

print("Transformer — Phase 2: activations only (weight key absent = fp16)")
write("tx_a_ptoken",  tx_all(None, "int8_per_channel"))
write("tx_a_ptensor", tx_all(None, "int8_per_tensor"))
# fc2 input is post-SiLU (not post-RMSNorm) so keep it dynamic
write("tx_a_fixed",   tx_all(None, "int8_fixed_4", fc2_a_method="int8_per_channel"))

print("Done.")
