#!/bin/bash
# Basecall each config and save per-read identity scores (minimap2 $10/$11).
# Runs up to NGPU configs simultaneously, one per GPU.
# Results saved to scripts/results/identity_<tag>.txt (one score per line).

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS=scripts/results
REF=/data/bonwon/slorado/test/slorado_test_ext_dat/genome/hg38noAlt.fa
NTHREADS=${NTHREADS:-8}
MINIMAP2=${MINIMAP2:-minimap2}
NGPU=${NGPU:-4}

mkdir -p "$RESULTS"
exec > >(tee "$RESULTS/identity.log") 2>&1

MODEL_LSTM=models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0
READS_LSTM=test/PGXXXX230339/reads_1k.blow5
MODEL_TX=models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0
READS_TX=/data/slow5-testdata/hg2_prom_lsk114_subsubsample/reads.blow5

run_one() {
    local gpu=$1 tag=$2 model=$3 reads=$4 qc=$5
    local tmp; tmp=$(mktemp /tmp/identity_XXXXXX.fastq)
    local args=(-C 64 -o "$tmp" "$model" "$reads")
    [[ "$qc" != "none" ]] && args+=(--quant-config "$qc")
    CUDA_VISIBLE_DEVICES=$gpu ./slorado basecaller "${args[@]}" 2>/dev/null
    "$MINIMAP2" -cx map-ont "$REF" -t"$NTHREADS" --secondary=no "$tmp" 2>/dev/null \
        | awk '$10>0 && $11>0 {print $10/$11}' \
        > "$RESULTS/identity_${tag}.txt"
    rm -f "$tmp"
    local n mean
    n=$(wc -l < "$RESULTS/identity_${tag}.txt")
    mean=$(awk '{s+=$1} END {printf "%.4f", s/NR}' "$RESULTS/identity_${tag}.txt")
    echo "  [gpu$gpu] $tag: aligned=$n  mean_identity=$mean"
}

run_batch() {
    local pids=()
    while [[ $# -ge 5 ]]; do
        run_one "$1" "$2" "$3" "$4" "$5" &
        pids+=($!)
        shift 5
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
}

echo "=== Generating configs ==="
python3 scripts/gen_configs.py
echo ""

echo "=== Baselines (fp16) ==="
run_batch \
    0 lstm_fp16 $MODEL_LSTM $READS_LSTM none \
    1 tx_fp16   $MODEL_TX   $READS_TX   none

echo ""
echo "=== Phase 1: weights only ==="
run_batch \
    0 lstm_hh_w_pc $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_w_pc.json \
    1 lstm_hh_w_pt $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_w_pt.json \
    2 lstm_ih_w_pc $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_w_pc.json \
    3 lstm_ih_w_pt $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_w_pt.json

run_batch \
    0 tx_w_pc $MODEL_TX $READS_TX /tmp/qc_tx_w_pc.json \
    1 tx_w_pt $MODEL_TX $READS_TX /tmp/qc_tx_w_pt.json

echo ""
echo "=== Phase 2: activations only ==="
run_batch \
    0 lstm_hh_a_ptoken  $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_a_ptoken.json \
    1 lstm_hh_a_ptensor $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_a_ptensor.json \
    2 lstm_hh_a_fixed   $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_a_fixed.json \
    3 lstm_ih_a_ptoken  $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_a_ptoken.json

run_batch \
    0 lstm_ih_a_ptensor $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_a_ptensor.json \
    1 tx_a_ptoken       $MODEL_TX   $READS_TX   /tmp/qc_tx_a_ptoken.json \
    2 tx_a_ptensor      $MODEL_TX   $READS_TX   /tmp/qc_tx_a_ptensor.json \
    3 tx_a_fixed        $MODEL_TX   $READS_TX   /tmp/qc_tx_a_fixed.json

echo ""
echo "Done. Results in $RESULTS/"
ls "$RESULTS"/identity_*.txt
