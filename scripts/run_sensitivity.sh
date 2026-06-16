#!/bin/bash
# Run sensitivity (KL divergence) tests across all phase 1 and phase 2 configs.
# Runs up to NGPU configs simultaneously, one per GPU.
# Results saved to scripts/results/sens_<tag>.json

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS=scripts/results
NGPU=${NGPU:-4}
mkdir -p "$RESULTS"
exec > >(tee "$RESULTS/sensitivity.log") 2>&1

MODEL_LSTM=models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0
READS_LSTM=test/PGXXXX230339/reads_1k.blow5
MODEL_TX=models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0
READS_TX=/data/slow5-testdata/hg2_prom_lsk114_subsubsample/reads.blow5

echo "=== Generating configs ==="
python3 scripts/gen_configs.py
echo ""

# Run a single sensitivity test on a specific GPU
run_one() {
    local gpu=$1 tag=$2 model=$3 reads=$4
    CUDA_VISIBLE_DEVICES=$gpu ./slorado basecaller \
        --quant-config /tmp/qc_${tag}.json \
        --sensitivity  "$RESULTS/sens_${tag}.json" \
        -C 64 -o /dev/null "$model" "$reads" 2>/dev/null
    python3 -c "
import json; d=json.load(open('$RESULTS/sens_${tag}.json'))
print(f'  [gpu$gpu] $tag: kl_mean={d[\"kl_mean\"]:.5g}  kl_max={d[\"kl_max\"]:.5g}')
"
}

# Run a batch of (gpu, tag, model, reads) tuples in parallel, then wait
run_batch() {
    local pids=()
    while [[ $# -ge 4 ]]; do
        run_one "$1" "$2" "$3" "$4" &
        pids+=($!)
        shift 4
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
}

echo "=== Phase 1: weights only ==="
run_batch \
    0 lstm_hh_w_pc $MODEL_LSTM $READS_LSTM \
    1 lstm_hh_w_pt $MODEL_LSTM $READS_LSTM \
    2 lstm_ih_w_pc $MODEL_LSTM $READS_LSTM \
    3 lstm_ih_w_pt $MODEL_LSTM $READS_LSTM

run_batch \
    0 tx_w_pc $MODEL_TX $READS_TX \
    1 tx_w_pt $MODEL_TX $READS_TX

echo ""
echo "=== Phase 2: activations only ==="
run_batch \
    0 lstm_hh_a_ptoken  $MODEL_LSTM $READS_LSTM \
    1 lstm_hh_a_ptensor $MODEL_LSTM $READS_LSTM \
    2 lstm_hh_a_fixed   $MODEL_LSTM $READS_LSTM \
    3 lstm_ih_a_ptoken  $MODEL_LSTM $READS_LSTM

run_batch \
    0 lstm_ih_a_ptensor $MODEL_LSTM $READS_LSTM \
    1 tx_a_ptoken       $MODEL_TX   $READS_TX \
    2 tx_a_ptensor      $MODEL_TX   $READS_TX \
    3 tx_a_fixed        $MODEL_TX   $READS_TX

echo ""
echo "Done. Results in $RESULTS/"
