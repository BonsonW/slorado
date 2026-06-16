#!/usr/bin/env python3
"""
Phase 3: automatically combine passing weight-only and act-only configs.

Reads scripts/results/sens_<tag>.json for all phase 1 and phase 2 tags.
For each (weight config, act config) pair from the same model+scope that both
fall below KL_THRESHOLD, generates a combined config and runs:
  - sensitivity  → scripts/results/sens_<combined_tag>.json
  - identity     → scripts/results/identity_<combined_tag>.txt

Usage:
  python3 scripts/run_phase3.py [--kl-threshold 0.05]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "scripts", "results")
REF     = os.path.join(ROOT, "test/slorado_test_ext_dat/genome/hg38noAlt.fa")
SLORADO = os.path.join(ROOT, "slorado")
MINIMAP2 = os.environ.get("MINIMAP2", "minimap2")
NTHREADS = os.environ.get("NTHREADS", "16")

MODEL_LSTM = os.path.join(ROOT, "models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0")
READS_LSTM = os.path.join(ROOT, "test/PGXXXX230339/reads_1k.blow5")
MODEL_TX   = os.path.join(ROOT, "models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0")
READS_TX   = "/data/slow5-testdata/hg2_prom_lsk114_subsubsample/reads.blow5"

# ── Phase metadata ─────────────────────────────────────────────────────────────
# (tag, model_key, scope, weight_method, act_method, fc2_act_override)
# weight_method=None means fp16 (no weight quant)
# act_method=None    means fp16 (no activation quant)

PHASE1 = [
    ("lstm_hh_w_pc", "lstm", "hh", "int8_per_channel", None,              None),
    ("lstm_hh_w_pt", "lstm", "hh", "int8_per_tensor",  None,              None),
    ("lstm_ih_w_pc", "lstm", "ih", "int8_per_channel", None,              None),
    ("lstm_ih_w_pt", "lstm", "ih", "int8_per_tensor",  None,              None),
    ("tx_w_pc",      "tx",   "all", "int8_per_channel", None,             None),
    ("tx_w_pt",      "tx",   "all", "int8_per_tensor",  None,             None),
]

PHASE2 = [
    ("lstm_hh_a_ptoken",  "lstm", "hh", None, "int8_per_channel", None),
    ("lstm_hh_a_ptensor", "lstm", "hh", None, "int8_per_tensor",  None),
    ("lstm_hh_a_fixed",   "lstm", "hh", None, "int8_fixed",       None),
    ("lstm_ih_a_ptoken",  "lstm", "ih", None, "int8_per_channel", None),
    ("lstm_ih_a_ptensor", "lstm", "ih", None, "int8_per_tensor",  None),
    ("tx_a_ptoken",  "tx", "all", None, "int8_per_channel", None),
    ("tx_a_ptensor", "tx", "all", None, "int8_per_tensor",  None),
    # fc2 input is post-SiLU, keep dynamic
    ("tx_a_fixed",   "tx", "all", None, "int8_fixed_4", "int8_per_channel"),
]

# ── Config generation ──────────────────────────────────────────────────────────

def make_lstm_hh(w, a):
    cfg = {}
    for i in range(1, 6):
        k = f"rnns.rnn{i}.hh_fused"
        if w: cfg[k]          = w
        if a: cfg[k + ".act"] = a
    return cfg

def make_lstm_ih(w, a):
    cfg = {}
    for i in range(1, 6):
        k = f"rnns.rnn{i}.ih_fused"
        if w: cfg[k]          = w
        if a: cfg[k + ".act"] = a
    return cfg

def make_tx_all(w, a, fc2_a=None):
    cfg = {}
    for i in range(18):
        for key in [f"transformer_encoder.{i}.self_attn.wqkv",
                    f"transformer_encoder.{i}.ff.fc1",
                    f"transformer_encoder.{i}.ff.fc2"]:
            if w: cfg[key] = w
            act = fc2_a if (fc2_a and key.endswith(".fc2")) else a
            if act: cfg[key + ".act"] = act
    return cfg

def make_config(model_key, scope, w, a, fc2_a=None):
    if model_key == "lstm":
        return make_lstm_hh(w, a) if scope == "hh" else make_lstm_ih(w, a)
    return make_tx_all(w, a, fc2_a)

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_kl(tag):
    path = os.path.join(RESULTS, f"sens_{tag}.json")
    try:
        d = json.load(open(path))
        return d["kl_mean"], d["kl_max"]
    except Exception:
        return None, None

def run_sensitivity(tag, qc_path, model, reads, env=None):
    sens_path = os.path.join(RESULTS, f"sens_{tag}.json")
    subprocess.run([
        SLORADO, "basecaller",
        "--quant-config", qc_path,
        "--sensitivity", sens_path,
        "-C", "64", "-o", "/dev/null",
        model, reads,
    ], stderr=subprocess.DEVNULL, env=env, check=True)
    return load_kl(tag)

def run_identity(tag, qc_path, model, reads, env=None):
    out_path = os.path.join(RESULTS, f"identity_{tag}.txt")
    with tempfile.NamedTemporaryFile(suffix=".fastq", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [SLORADO, "basecaller", "-C", "64", "-o", tmp_path, model, reads]
        if qc_path:
            cmd += ["--quant-config", qc_path]
        subprocess.run(cmd, stderr=subprocess.DEVNULL, env=env, check=True)
        with open(out_path, "w") as fout:
            mm = subprocess.Popen(
                [MINIMAP2, "-cx", "map-ont", REF, f"-t{NTHREADS}", "--secondary=no", tmp_path],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["awk", "$10>0 && $11>0 {print $10/$11}"],
                stdin=mm.stdout, stdout=fout, check=True,
            )
            mm.wait()
        n    = sum(1 for _ in open(out_path))
        mean = sum(float(l) for l in open(out_path)) / max(n, 1)
        return n, mean
    finally:
        os.unlink(tmp_path)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kl-threshold", type=float, default=0.05,
                        help="Max kl_mean to consider a config 'passing' (default 0.05)")
    args = parser.parse_args()
    threshold = args.kl_threshold

    os.makedirs(RESULTS, exist_ok=True)

    log_path = os.path.join(RESULTS, "phase3.log")
    log = open(log_path, "w")
    def emit(msg=""):
        print(msg)
        log.write(msg + "\n")
        log.flush()

    emit(f"Phase 3 — KL threshold: {threshold}")
    emit()

    # Load phase 1 and 2 results
    p1_pass = [(t, mk, sc, w, a, fc2) for t, mk, sc, w, a, fc2 in PHASE1
               if (load_kl(t)[0] or 999) <= threshold]
    p2_pass = [(t, mk, sc, w, a, fc2) for t, mk, sc, w, a, fc2 in PHASE2
               if (load_kl(t)[0] or 999) <= threshold]

    emit("Passing phase 1 (weight-only):")
    for t, *_ in p1_pass:
        kl, _ = load_kl(t)
        emit(f"  {t}: kl_mean={kl:.5g}")
    emit()
    emit("Passing phase 2 (act-only):")
    for t, *_ in p2_pass:
        kl, _ = load_kl(t)
        emit(f"  {t}: kl_mean={kl:.5g}")
    emit()

    if not p1_pass:
        emit("No phase 1 configs passed the threshold — lower --kl-threshold or re-run phase 1.")
        return
    if not p2_pass:
        emit("No phase 2 configs passed the threshold — lower --kl-threshold or re-run phase 2.")
        return

    # Pair up by model_key + scope
    combos = []
    for t1, mk1, sc1, w1, _, _   in p1_pass:
        for t2, mk2, sc2, _, a2, fc2_2 in p2_pass:
            if mk1 == mk2 and sc1 == sc2:
                combos.append((t1, t2, mk1, sc1, w1, a2, fc2_2))

    ngpu = int(os.environ.get("NGPU", "4"))
    emit(f"Running {len(combos)} combined configs across {ngpu} GPUs:")

    def run_combo(gpu, t1, t2, mk, sc, w, a, fc2):
        tag     = f"p3_{t1}_x_{t2}"
        model   = MODEL_TX   if mk == "tx" else MODEL_LSTM
        reads   = READS_TX   if mk == "tx" else READS_LSTM
        qc      = make_config(mk, sc, w, a, fc2)
        qc_path = f"/tmp/qc_{tag}.json"
        json.dump(qc, open(qc_path, "w"))

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        emit(f"\n  [gpu{gpu}] {tag}  weight={w or 'fp16'}  act={a or 'fp16'}" +
             (f"  fc2_act={fc2}" if fc2 else ""))
        try:
            kl_mean, kl_max = run_sensitivity(tag, qc_path, model, reads, env)
            emit(f"    sensitivity: kl_mean={kl_mean:.5g}  kl_max={kl_max:.5g}")
        except Exception as e:
            emit(f"    sensitivity FAILED: {e}")
            return
        try:
            n, mean_id = run_identity(tag, qc_path, model, reads, env)
            emit(f"    identity:    aligned={n}  mean={mean_id:.4f}")
        except Exception as e:
            emit(f"    identity FAILED: {e}")

    # Dispatch combos across GPUs in batches
    for batch_start in range(0, len(combos), ngpu):
        batch = combos[batch_start:batch_start + ngpu]
        threads = []
        for i, combo in enumerate(batch):
            gpu = i % ngpu
            t = threading.Thread(target=run_combo, args=(gpu, *combo))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    emit()
    emit(f"Done. Results in {RESULTS}/")
    emit(f"Log: {log_path}")
    log.close()

if __name__ == "__main__":
    main()
