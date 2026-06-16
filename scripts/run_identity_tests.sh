#!/bin/bash
# Run basecalling for each quant config and save per-read identity scores.
# Identity = matching_bases / alignment_length from minimap2 PAF (cols 10/11).
# Output: one .txt file per config in scripts/results/, one identity score per line.

set -euo pipefail

SLORADO=$(dirname "$0")/../slorado
REF=/data/bonwon/slorado/test/slorado_test_ext_dat/genome/hg38noAlt.fa
RESULTS=$(dirname "$0")/results
NTHREADS=${NTHREADS:-16}
MINIMAP2=${MINIMAP2:-minimap2}

MODEL_LSTM=models/dna_r10.4.1_e8.2_400bps_hac@v6.0.0
READS_LSTM=test/PGXXXX230339/reads_1k.blow5

MODEL_TX=models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0
READS_TX=/data/slow5-testdata/hg2_prom_lsk114_subsubsample/reads.blow5

mkdir -p "$RESULTS"
FASTQ_TMP=$(mktemp /tmp/identity_XXXXXX.fastq)
trap "rm -f $FASTQ_TMP" EXIT

basecall_and_score() {
    local tag=$1 model=$2 reads=$3
    shift 3
    # remaining args passed to slorado (e.g. --quant-config ...)
    echo "=== $tag ==="
    "$SLORADO" basecaller -C 64 -o "$FASTQ_TMP" "$model" "$reads" "$@" 2>/dev/null
    "$MINIMAP2" -cx map-ont "$REF" -t"$NTHREADS" --secondary=no "$FASTQ_TMP" 2>/dev/null \
        | awk '$10>0 && $11>0 {print $10/$11}' \
        > "$RESULTS/identity_${tag}.txt"
    n=$(wc -l < "$RESULTS/identity_${tag}.txt")
    mean=$(awk '{s+=$1} END {printf "%.4f", s/NR}' "$RESULTS/identity_${tag}.txt")
    echo "  aligned reads: $n  mean identity: $mean"
}

# ── FLSTM (HAC v6) ────────────────────────────────────────────────────────────
basecall_and_score lstm_fp16         $MODEL_LSTM $READS_LSTM
basecall_and_score lstm_hh_pc        $MODEL_LSTM $READS_LSTM --quant-config /tmp/qc_all_pc.json
basecall_and_score lstm_hh_w_pc_a_pt $MODEL_LSTM $READS_LSTM --quant-config /tmp/qc_hh_w_pc_a_pt.json
basecall_and_score lstm_hh_fixed     $MODEL_LSTM $READS_LSTM --quant-config /tmp/qc_all_fixed.json
basecall_and_score lstm_hh_pt        $MODEL_LSTM $READS_LSTM --quant-config /tmp/qc_all_pt.json
basecall_and_score lstm_ih_pc        $MODEL_LSTM $READS_LSTM --quant-config /tmp/qc_ih_pc.json
basecall_and_score lstm_ih_w_pc_a_pt $MODEL_LSTM $READS_LSTM --quant-config /tmp/qc_ih_w_pc_a_pt.json
basecall_and_score lstm_ih_pt        $MODEL_LSTM $READS_LSTM --quant-config /tmp/qc_ih_pt.json

# ── Transformer (SUP v5) ──────────────────────────────────────────────────────
basecall_and_score tx_fp16           $MODEL_TX $READS_TX
basecall_and_score tx_int8_pc        $MODEL_TX $READS_TX --quant-config /tmp/qc_tx_int8_pc.json
basecall_and_score tx_w_pc_a_pt      $MODEL_TX $READS_TX --quant-config /tmp/qc_tx_w_pc_a_pt.json
basecall_and_score tx_all_pt         $MODEL_TX $READS_TX --quant-config /tmp/qc_tx_all_pt.json
basecall_and_score tx_wqkv_fixed     $MODEL_TX $READS_TX --quant-config /tmp/qc_tx_wqkv_fixed.json
basecall_and_score tx_fc1_fixed      $MODEL_TX $READS_TX --quant-config /tmp/qc_tx_fc1_fixed.json
basecall_and_score tx_both_fixed     $MODEL_TX $READS_TX --quant-config /tmp/qc_tx_both_fixed.json

echo ""
echo "Done. Results in $RESULTS/"
ls "$RESULTS"/identity_*.txt
