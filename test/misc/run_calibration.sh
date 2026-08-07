#!/bin/bash
# Run calibration passes for LSTM and TX models, then generate quant configs.
# Run this once before run_sensitivity.sh.
#
# Weight calibration: instant, reads weights directly from model directory.
# Activation calibration: runs 50 inference batches on each model.
#
# Results:
#   scripts/results/calib_weights_{lstm,tx}.json  (weights, from calib_weights.py)
#   scripts/results/calib_acts_{lstm,tx}.json      (activations, from slorado --calibrate)
#   /tmp/qc_*.json                                 (quant configs for sensitivity)

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_LSTM=models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0
READS_LSTM=test/PGXXXX230339/reads_1k.blow5
MODEL_TX=models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0
READS_TX=test/PGXXXX230339/reads_1k.blow5

RESULTS=scripts/results
mkdir -p "$RESULTS"

CALIB_W_LSTM=$RESULTS/calib_weights_lstm.json
CALIB_W_TX=$RESULTS/calib_weights_tx.json
CALIB_A_LSTM=$RESULTS/calib_acts_lstm.json
CALIB_A_TX=$RESULTS/calib_acts_tx.json

echo "=== Weight calibration (no inference needed) ==="
pyvenv/bin/python scripts/calib_weights.py "$MODEL_LSTM" "$CALIB_W_LSTM"
pyvenv/bin/python scripts/calib_weights.py "$MODEL_TX"   "$CALIB_W_TX"
echo ""

echo "=== Activation calibration (200 batches each) ==="
CUDA_VISIBLE_DEVICES=0 ./slorado basecaller \
    --calibrate "$CALIB_A_LSTM" --debug-break 200 -C 128 -o /dev/null \
    "$MODEL_LSTM" "$READS_LSTM" 2>/dev/null &
PID_LSTM=$!

CUDA_VISIBLE_DEVICES=1 ./slorado basecaller \
    --calibrate "$CALIB_A_TX" --debug-break 200 -C 128 -o /dev/null \
    "$MODEL_TX" "$READS_TX" 2>/dev/null &
PID_TX=$!

wait "$PID_LSTM" && echo "  LSTM activation calibration done → $CALIB_A_LSTM"
wait "$PID_TX"   && echo "  TX activation calibration done   → $CALIB_A_TX"
echo ""

echo "=== Generating calibrated configs ==="
pyvenv/bin/python scripts/gen_calib_configs.py --lstm "$CALIB_A_LSTM" --tx "$CALIB_A_TX"
echo ""
echo "Done. Re-run weight calibration if model changes, activation calibration if reads change."
