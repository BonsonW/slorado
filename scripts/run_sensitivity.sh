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
NOTIFY_EMAIL=$1

notify() {
    local subject=$1
    { echo "Host: $(hostname)"; echo "Log: $(pwd)/$RESULTS/sensitivity.log"; echo; cat "$RESULTS/sensitivity.log" 2>/dev/null; } \
        | mail -s "$subject" "$NOTIFY_EMAIL" 2>/dev/null || true
}

mkdir -p "$RESULTS"
exec > >(tee "$RESULTS/sensitivity.log") 2>&1

trap 'notify "slorado sensitivity FAILED on $(hostname)"' ERR

BLOW5=${BLOW5:-test/PGXXXX230339/reads_1k.blow5}
GPU_BATCH=${GPU_BATCH:-128}
MODEL_LSTM=models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0
MODEL_TX=models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0

# Generate quant configs if not already present.
if [[ ! -f /tmp/qc_lstm_dn_ih_w_pc.json ]]; then
    echo "=== Generating configs ==="
    python3 scripts/gen_configs.py
    echo ""
fi

# ── run_one: sensitivity + identity → combined <tag>.tsv ──────────────────────
run_one() {
    local gpu=$1 tag=$2 model=$3 reads=$4 qc=$5
    local kl_out="$RESULTS/${tag}.tsv"
    local id_out="$RESULTS/${tag}_id.tsv"
    local n_batches="NA" kl_mean="NA" kl_max="NA"

    local tmp; tmp=$(mktemp /tmp/identity_XXXXXX.fastq)

    # Sensitivity + basecall in one pass (skip for fp16 baselines)
    if [[ "$qc" != "none" ]]; then
        local sens_tmp; sens_tmp=$(mktemp /tmp/sens_XXXXXX.tsv)
        CUDA_VISIBLE_DEVICES=$gpu ./slorado basecaller \
            --flash=yes \
            --quant-config "$qc" \
            --sensitivity "$sens_tmp" \
            -C $GPU_BATCH -o "$tmp" "$model" "$reads"
        read -r n_batches kl_mean kl_max < <(awk 'NR==2 {print $2, $3, $4}' "$sens_tmp")
        rm -f "$sens_tmp"
        echo "  [gpu$gpu] $tag: kl_mean=$kl_mean  kl_max=$kl_max"
    else
        CUDA_VISIBLE_DEVICES=$gpu ./slorado basecaller \
            --flash=yes -C $GPU_BATCH -o "$tmp" "$model" "$reads"
    fi

    # KL summary: one row per config
    printf 'tag\tn_batches\tkl_mean\tkl_max\n' > "$kl_out"
    printf '%s\t%s\t%s\t%s\n' "$tag" "$n_batches" "$kl_mean" "$kl_max" >> "$kl_out"

    "$MINIMAP2" -cx map-ont "$REF" -t"$NTHREADS" --secondary=no "$tmp" 2>/dev/null \
        | awk '$10>0 && $11>0 {printf "%.6f\n", $10/$11}' \
        > "$id_out"
    rm -f "$tmp"

    local n mean
    n=$(wc -l < "$id_out")
    mean=$(awk '{s+=$1; c++} END {printf "%.4f", s/c}' "$id_out")
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
    0 lstm_fp16 $MODEL_LSTM $BLOW5 none \
    1 tx_fp16   $MODEL_TX   $BLOW5   none

echo ""
echo "=== Phase 1: weights only ==="
for lname in dn_ih up_ih dn_hh up_hh; do
run_batch \
    0 lstm_${lname}_w_pc    $MODEL_LSTM $BLOW5 /tmp/qc_lstm_${lname}_w_pc.json \
    1 lstm_${lname}_w_pt    $MODEL_LSTM $BLOW5 /tmp/qc_lstm_${lname}_w_pt.json \
    2 lstm_${lname}_w_fp8pc $MODEL_LSTM $BLOW5 /tmp/qc_lstm_${lname}_w_fp8pc.json \
    3 lstm_${lname}_w_fp8pt $MODEL_LSTM $BLOW5 /tmp/qc_lstm_${lname}_w_fp8pt.json
done

run_batch \
    0 tx_wqkv_w_pc    $MODEL_TX $BLOW5 /tmp/qc_tx_wqkv_w_pc.json \
    1 tx_wqkv_w_pt    $MODEL_TX $BLOW5 /tmp/qc_tx_wqkv_w_pt.json \
    2 tx_wqkv_w_fp8pc $MODEL_TX $BLOW5 /tmp/qc_tx_wqkv_w_fp8pc.json \
    3 tx_wqkv_w_fp8pt $MODEL_TX $BLOW5 /tmp/qc_tx_wqkv_w_fp8pt.json

run_batch \
    0 tx_op_w_pc    $MODEL_TX $BLOW5 /tmp/qc_tx_op_w_pc.json \
    1 tx_op_w_pt    $MODEL_TX $BLOW5 /tmp/qc_tx_op_w_pt.json \
    2 tx_op_w_fp8pc $MODEL_TX $BLOW5 /tmp/qc_tx_op_w_fp8pc.json \
    3 tx_op_w_fp8pt $MODEL_TX $BLOW5 /tmp/qc_tx_op_w_fp8pt.json

run_batch \
    0 tx_fc1_w_pc    $MODEL_TX $BLOW5 /tmp/qc_tx_fc1_w_pc.json \
    1 tx_fc1_w_pt    $MODEL_TX $BLOW5 /tmp/qc_tx_fc1_w_pt.json \
    2 tx_fc1_w_fp8pc $MODEL_TX $BLOW5 /tmp/qc_tx_fc1_w_fp8pc.json \
    3 tx_fc1_w_fp8pt $MODEL_TX $BLOW5 /tmp/qc_tx_fc1_w_fp8pt.json

run_batch \
    0 tx_fc2_w_pc    $MODEL_TX $BLOW5 /tmp/qc_tx_fc2_w_pc.json \
    1 tx_fc2_w_pt    $MODEL_TX $BLOW5 /tmp/qc_tx_fc2_w_pt.json \
    2 tx_fc2_w_fp8pc $MODEL_TX $BLOW5 /tmp/qc_tx_fc2_w_fp8pc.json \
    3 tx_fc2_w_fp8pt $MODEL_TX $BLOW5 /tmp/qc_tx_fc2_w_fp8pt.json

echo ""
echo "=== Phase 2: activations only ==="
for lname in dn_ih up_ih up_hh; do
run_batch \
    0 lstm_${lname}_a_ptoken    $MODEL_LSTM $BLOW5 /tmp/qc_lstm_${lname}_a_ptoken.json \
    1 lstm_${lname}_a_fp8ptoken $MODEL_LSTM $BLOW5 /tmp/qc_lstm_${lname}_a_fp8ptoken.json \
    2 tx_wqkv_a_ptoken    $MODEL_TX   $BLOW5 /tmp/qc_tx_wqkv_a_ptoken.json \
    3 tx_wqkv_a_fp8ptoken $MODEL_TX   $BLOW5 /tmp/qc_tx_wqkv_a_fp8ptoken.json
done
# dn_hh activations are hidden states hh[t] ∈ [-1,1] — also test fixed scale
run_batch \
    0 lstm_dn_hh_a_ptoken    $MODEL_LSTM $BLOW5 /tmp/qc_lstm_dn_hh_a_ptoken.json \
    1 lstm_dn_hh_a_fp8ptoken $MODEL_LSTM $BLOW5 /tmp/qc_lstm_dn_hh_a_fp8ptoken.json \
    2 lstm_dn_hh_a_fixed     $MODEL_LSTM $BLOW5 /tmp/qc_lstm_dn_hh_a_fixed.json \
    3 lstm_dn_hh_a_fp8fixed  $MODEL_LSTM $BLOW5 /tmp/qc_lstm_dn_hh_a_fp8fixed.json

run_batch \
    0 tx_wqkv_a_fixed  $MODEL_TX $BLOW5 /tmp/qc_tx_wqkv_a_fixed.json \
    1 tx_op_a_ptoken   $MODEL_TX $BLOW5 /tmp/qc_tx_op_a_ptoken.json \
    2 tx_op_a_fp8ptoken$MODEL_TX $BLOW5 /tmp/qc_tx_op_a_fp8ptoken.json \
    3 tx_fc1_a_ptoken  $MODEL_TX $BLOW5 /tmp/qc_tx_fc1_a_ptoken.json

run_batch \
    0 tx_fc1_a_fp8ptoken $MODEL_TX $BLOW5 /tmp/qc_tx_fc1_a_fp8ptoken.json \
    1 tx_fc1_a_fixed     $MODEL_TX $BLOW5 /tmp/qc_tx_fc1_a_fixed.json \
    2 tx_fc2_a_ptoken    $MODEL_TX $BLOW5 /tmp/qc_tx_fc2_a_ptoken.json \
    3 tx_fc2_a_fp8ptoken $MODEL_TX $BLOW5 /tmp/qc_tx_fc2_a_fp8ptoken.json


echo ""
echo "=== Phase 1 MX: weights only (OCP group-32 microscaling) ==="
for lname in dn_ih up_ih dn_hh up_hh; do
run_batch \
    0 lstm_${lname}_w_mxint8 $MODEL_LSTM $BLOW5 /tmp/qc_lstm_${lname}_w_mxint8.json \
    1 lstm_${lname}_w_mxfp4  $MODEL_LSTM $BLOW5 /tmp/qc_lstm_${lname}_w_mxfp4.json \
    2 lstm_${lname}_w_mxfp6  $MODEL_LSTM $BLOW5 /tmp/qc_lstm_${lname}_w_mxfp6.json \
    3 lstm_${lname}_w_mxfp8  $MODEL_LSTM $BLOW5 /tmp/qc_lstm_${lname}_w_mxfp8.json
done

run_batch \
    0 tx_wqkv_w_mxint8 $MODEL_TX $BLOW5 /tmp/qc_tx_wqkv_w_mxint8.json \
    1 tx_wqkv_w_mxfp4  $MODEL_TX $BLOW5 /tmp/qc_tx_wqkv_w_mxfp4.json \
    2 tx_wqkv_w_mxfp6  $MODEL_TX $BLOW5 /tmp/qc_tx_wqkv_w_mxfp6.json \
    3 tx_wqkv_w_mxfp8  $MODEL_TX $BLOW5 /tmp/qc_tx_wqkv_w_mxfp8.json

run_batch \
    0 tx_op_w_mxint8 $MODEL_TX $BLOW5 /tmp/qc_tx_op_w_mxint8.json \
    1 tx_op_w_mxfp4  $MODEL_TX $BLOW5 /tmp/qc_tx_op_w_mxfp4.json \
    2 tx_op_w_mxfp6  $MODEL_TX $BLOW5 /tmp/qc_tx_op_w_mxfp6.json \
    3 tx_op_w_mxfp8  $MODEL_TX $BLOW5 /tmp/qc_tx_op_w_mxfp8.json

run_batch \
    0 tx_fc1_w_mxint8 $MODEL_TX $BLOW5 /tmp/qc_tx_fc1_w_mxint8.json \
    1 tx_fc1_w_mxfp4  $MODEL_TX $BLOW5 /tmp/qc_tx_fc1_w_mxfp4.json \
    2 tx_fc1_w_mxfp6  $MODEL_TX $BLOW5 /tmp/qc_tx_fc1_w_mxfp6.json \
    3 tx_fc1_w_mxfp8  $MODEL_TX $BLOW5 /tmp/qc_tx_fc1_w_mxfp8.json

run_batch \
    0 tx_fc2_w_mxint8 $MODEL_TX $BLOW5 /tmp/qc_tx_fc2_w_mxint8.json \
    1 tx_fc2_w_mxfp4  $MODEL_TX $BLOW5 /tmp/qc_tx_fc2_w_mxfp4.json \
    2 tx_fc2_w_mxfp6  $MODEL_TX $BLOW5 /tmp/qc_tx_fc2_w_mxfp6.json \
    3 tx_fc2_w_mxfp8  $MODEL_TX $BLOW5 /tmp/qc_tx_fc2_w_mxfp8.json

echo ""
echo "=== Phase 2 MX: activations only (OCP group-32 microscaling) ==="
for lname in dn_ih up_ih dn_hh up_hh; do
run_batch \
    0 lstm_${lname}_a_mxint8 $MODEL_LSTM $BLOW5 /tmp/qc_lstm_${lname}_a_mxint8.json \
    1 lstm_${lname}_a_mxfp4  $MODEL_LSTM $BLOW5 /tmp/qc_lstm_${lname}_a_mxfp4.json \
    2 lstm_${lname}_a_mxfp6  $MODEL_LSTM $BLOW5 /tmp/qc_lstm_${lname}_a_mxfp6.json \
    3 lstm_${lname}_a_mxfp8  $MODEL_LSTM $BLOW5 /tmp/qc_lstm_${lname}_a_mxfp8.json
done

run_batch \
    0 tx_wqkv_a_mxint8 $MODEL_TX $BLOW5 /tmp/qc_tx_wqkv_a_mxint8.json \
    1 tx_wqkv_a_mxfp4  $MODEL_TX $BLOW5 /tmp/qc_tx_wqkv_a_mxfp4.json \
    2 tx_wqkv_a_mxfp6  $MODEL_TX $BLOW5 /tmp/qc_tx_wqkv_a_mxfp6.json \
    3 tx_wqkv_a_mxfp8  $MODEL_TX $BLOW5 /tmp/qc_tx_wqkv_a_mxfp8.json

run_batch \
    0 tx_op_a_mxint8 $MODEL_TX $BLOW5 /tmp/qc_tx_op_a_mxint8.json \
    1 tx_op_a_mxfp4  $MODEL_TX $BLOW5 /tmp/qc_tx_op_a_mxfp4.json \
    2 tx_op_a_mxfp6  $MODEL_TX $BLOW5 /tmp/qc_tx_op_a_mxfp6.json \
    3 tx_op_a_mxfp8  $MODEL_TX $BLOW5 /tmp/qc_tx_op_a_mxfp8.json

run_batch \
    0 tx_fc1_a_mxint8 $MODEL_TX $BLOW5 /tmp/qc_tx_fc1_a_mxint8.json \
    1 tx_fc1_a_mxfp4  $MODEL_TX $BLOW5 /tmp/qc_tx_fc1_a_mxfp4.json \
    2 tx_fc1_a_mxfp6  $MODEL_TX $BLOW5 /tmp/qc_tx_fc1_a_mxfp6.json \
    3 tx_fc1_a_mxfp8  $MODEL_TX $BLOW5 /tmp/qc_tx_fc1_a_mxfp8.json

run_batch \
    0 tx_fc2_a_mxint8 $MODEL_TX $BLOW5 /tmp/qc_tx_fc2_a_mxint8.json \
    1 tx_fc2_a_mxfp4  $MODEL_TX $BLOW5 /tmp/qc_tx_fc2_a_mxfp4.json \
    2 tx_fc2_a_mxfp6  $MODEL_TX $BLOW5 /tmp/qc_tx_fc2_a_mxfp6.json \
    3 tx_fc2_a_mxfp8  $MODEL_TX $BLOW5 /tmp/qc_tx_fc2_a_mxfp8.json

echo ""
echo "Done. Results in $RESULTS/"
ls "$RESULTS"/*.tsv "$RESULTS"/*_id.tsv 2>/dev/null | sort -u

notify "slorado sensitivity complete on $(hostname)"
