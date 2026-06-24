#!/bin/bash
# Run calibration passes for LSTM and TX models, then generate quant configs.
# Run this once before run_sensitivity.sh.
# Results: /tmp/calib_lstm.json, /tmp/calib_tx.json, /tmp/qc_*.json

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_LSTM=models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0
READS_LSTM=test/PGXXXX230339/reads_1k.blow5
MODEL_TX=models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0
READS_TX=test/PGXXXX230339/reads_1k.blow5

RESULTS=scripts/results
mkdir -p "$RESULTS"

CALIB_LSTM=$RESULTS/calib_lstm.json
CALIB_TX=$RESULTS/calib_tx.json

echo "=== Calibration passes (50 batches each) ==="
CUDA_VISIBLE_DEVICES=0 ./slorado basecaller \
    --calibrate "$CALIB_LSTM" --debug-break 50 -C 64 -o /dev/null \
    "$MODEL_LSTM" "$READS_LSTM" 2>/dev/null &
PID_LSTM=$!

CUDA_VISIBLE_DEVICES=1 ./slorado basecaller \
    --calibrate "$CALIB_TX" --debug-break 50 -C 64 -o /dev/null \
    "$MODEL_TX" "$READS_TX" 2>/dev/null &
PID_TX=$!

wait "$PID_LSTM" && echo "  LSTM calibration done → $CALIB_LSTM"
wait "$PID_TX"   && echo "  TX calibration done   → $CALIB_TX"
echo ""

echo "=== Generating calibrated configs ==="
python3 scripts/gen_calib_configs.py --lstm "$CALIB_LSTM" --tx "$CALIB_TX"
echo ""
echo "Done. Re-run this script if the model or reads change."
