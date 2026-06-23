#!/bin/bash
# Run sensitivity (KL divergence) + identity tests for all phase 1 and phase 2 configs.
# Results saved to scripts/results/<tag>.tsv
# Columns: tag  n_batches  kl_mean  kl_max  identity
# One row per aligned read; KL stats repeated per row (NA for fp16 baselines).
# Run scripts/run_calibration.sh first for calibrated per-tensor activation scales.

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS=scripts/results
REF=/data/bonwon/slorado/test/slorado_test_ext_dat/genome/hg38noAlt.fa
NTHREADS=${NTHREADS:-8}
MINIMAP2=${MINIMAP2:-minimap2}

mkdir -p "$RESULTS"
exec > >(tee "$RESULTS/sensitivity.log") 2>&1

MODEL_LSTM=models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0
READS_LSTM=test/PGXXXX230339/reads_1k.blow5
MODEL_TX=models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0
READS_TX=/data/slow5-testdata/hg2_prom_lsk114_subsubsample/reads.blow5

# Generate quant configs if not already present.
if [[ ! -f /tmp/qc_lstm_hh_w_pc.json ]]; then
    echo "=== Generating configs ==="
    python3 scripts/gen_configs.py
    echo ""
fi

# ── run_one: sensitivity + identity → combined <tag>.tsv ──────────────────────
run_one() {
    local gpu=$1 tag=$2 model=$3 reads=$4 qc=$5
    local out="$RESULTS/${tag}.tsv"
    local n_batches="NA" kl_mean="NA" kl_max="NA"

    # Sensitivity (skip for fp16 baselines)
    if [[ "$qc" != "none" ]]; then
        local sens_tmp; sens_tmp=$(mktemp /tmp/sens_XXXXXX.tsv)
        CUDA_VISIBLE_DEVICES=$gpu ./slorado basecaller \
            --flash=yes \
            --quant-config "$qc" \
            --sensitivity "$sens_tmp" \
            -C 64 -o /dev/null "$model" "$reads" 2>/dev/null
        read -r n_batches kl_mean kl_max < <(awk 'NR==2 {print $2, $3, $4}' "$sens_tmp")
        rm -f "$sens_tmp"
        echo "  [gpu$gpu] $tag: kl_mean=$kl_mean  kl_max=$kl_max"
    fi

    # Identity: basecall → minimap2 → per-read scores
    local tmp; tmp=$(mktemp /tmp/identity_XXXXXX.fastq)
    local base_args=(-C 64 -o "$tmp" "$model" "$reads")
    [[ "$qc" != "none" ]] && base_args=(--quant-config "$qc" "${base_args[@]}")
    CUDA_VISIBLE_DEVICES=$gpu ./slorado basecaller "${base_args[@]}" 2>/dev/null

    # Write combined TSV: header + one row per aligned read
    printf 'tag\tn_batches\tkl_mean\tkl_max\tidentity\n' > "$out"
    "$MINIMAP2" -cx map-ont "$REF" -t"$NTHREADS" --secondary=no "$tmp" 2>/dev/null \
        | awk -v tag="$tag" -v nb="$n_batches" -v km="$kl_mean" -v kx="$kl_max" \
              '$10>0 && $11>0 {printf "%s\t%s\t%s\t%s\t%.6f\n", tag, nb, km, kx, $10/$11}' \
        >> "$out"
    rm -f "$tmp"

    local n mean
    n=$(awk 'NR>1' "$out" | wc -l)
    mean=$(awk 'NR>1 {s+=$5; c++} END {printf "%.4f", s/c}' "$out")
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

# ── Baselines (fp16) ───────────────────────────────────────────────────────────
echo "=== Baselines (fp16) ==="
run_batch \
    0 lstm_fp16 $MODEL_LSTM $READS_LSTM none \
    1 tx_fp16   $MODEL_TX   $READS_TX   none

echo ""
echo "=== Phase 1: weights only ==="
run_batch \
    0 lstm_hh_w_pc    $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_w_pc.json \
    1 lstm_hh_w_pt    $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_w_pt.json \
    2 lstm_hh_w_fp8pc $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_w_fp8pc.json \
    3 lstm_hh_w_fp8pt $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_w_fp8pt.json

run_batch \
    0 lstm_ih_w_pc    $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_w_pc.json \
    1 lstm_ih_w_pt    $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_w_pt.json \
    2 lstm_ih_w_fp8pc $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_w_fp8pc.json \
    3 lstm_ih_w_fp8pt $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_w_fp8pt.json

run_batch \
    0 tx_w_pc    $MODEL_TX $READS_TX /tmp/qc_tx_w_pc.json \
    1 tx_w_pt    $MODEL_TX $READS_TX /tmp/qc_tx_w_pt.json \
    2 tx_w_fp8pc $MODEL_TX $READS_TX /tmp/qc_tx_w_fp8pc.json \
    3 tx_w_fp8pt $MODEL_TX $READS_TX /tmp/qc_tx_w_fp8pt.json

echo ""
echo "=== Phase 2: activations only ==="
run_batch \
    0 lstm_hh_a_ptoken    $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_a_ptoken.json \
    1 lstm_hh_a_fp8ptoken $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_a_fp8ptoken.json \
    2 lstm_hh_a_fixed     $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_a_fixed.json \
    3 lstm_ih_a_ptoken    $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_a_ptoken.json

run_batch \
    0 lstm_ih_a_fp8ptoken $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_a_fp8ptoken.json \
    1 tx_a_ptoken         $MODEL_TX   $READS_TX   /tmp/qc_tx_a_ptoken.json \
    2 tx_a_fp8ptoken      $MODEL_TX   $READS_TX   /tmp/qc_tx_a_fp8ptoken.json \
    3 tx_a_fixed          $MODEL_TX   $READS_TX   /tmp/qc_tx_a_fixed.json

echo ""
echo "=== Phase 1 MX: weights only (OCP group-32 microscaling) ==="
run_batch \
    0 lstm_hh_w_mxint8 $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_w_mxint8.json \
    1 lstm_hh_w_mxfp4  $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_w_mxfp4.json \
    2 lstm_hh_w_mxfp6  $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_w_mxfp6.json \
    3 lstm_hh_w_mxfp8  $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_w_mxfp8.json

run_batch \
    0 lstm_ih_w_mxint8 $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_w_mxint8.json \
    1 lstm_ih_w_mxfp4  $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_w_mxfp4.json \
    2 lstm_ih_w_mxfp6  $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_w_mxfp6.json \
    3 lstm_ih_w_mxfp8  $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_w_mxfp8.json

run_batch \
    0 tx_w_mxint8 $MODEL_TX $READS_TX /tmp/qc_tx_w_mxint8.json \
    1 tx_w_mxfp4  $MODEL_TX $READS_TX /tmp/qc_tx_w_mxfp4.json \
    2 tx_w_mxfp6  $MODEL_TX $READS_TX /tmp/qc_tx_w_mxfp6.json \
    3 tx_w_mxfp8  $MODEL_TX $READS_TX /tmp/qc_tx_w_mxfp8.json

echo ""
echo "=== Phase 2 MX: activations only (OCP group-32 microscaling) ==="
run_batch \
    0 lstm_hh_a_mxint8 $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_a_mxint8.json \
    1 lstm_hh_a_mxfp4  $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_a_mxfp4.json \
    2 lstm_hh_a_mxfp6  $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_a_mxfp6.json \
    3 lstm_hh_a_mxfp8  $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_hh_a_mxfp8.json

run_batch \
    0 lstm_ih_a_mxint8 $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_a_mxint8.json \
    1 lstm_ih_a_mxfp4  $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_a_mxfp4.json \
    2 lstm_ih_a_mxfp6  $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_a_mxfp6.json \
    3 lstm_ih_a_mxfp8  $MODEL_LSTM $READS_LSTM /tmp/qc_lstm_ih_a_mxfp8.json

run_batch \
    0 tx_a_mxint8 $MODEL_TX $READS_TX /tmp/qc_tx_a_mxint8.json \
    1 tx_a_mxfp4  $MODEL_TX $READS_TX /tmp/qc_tx_a_mxfp4.json \
    2 tx_a_mxfp6  $MODEL_TX $READS_TX /tmp/qc_tx_a_mxfp6.json \
    3 tx_a_mxfp8  $MODEL_TX $READS_TX /tmp/qc_tx_a_mxfp8.json

echo ""
echo "Done. Results in $RESULTS/"
ls "$RESULTS"/*.tsv
