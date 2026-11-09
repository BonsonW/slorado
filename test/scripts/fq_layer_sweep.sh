#!/bin/bash
# Per-layer fake-quant sweep on several GPUs: starts one "fq_sweep.sh layer" per GPU in the background,
# each taking every N_GPUS-th run, all writing to the same OUT. Rerun to resume.
#
# usage: edit the settings below, then run test/scripts/fq_layer_sweep.sh from the slorado directory.

# ---- settings ----
MODEL=models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0
READS=/data/bonwon/PGXXXX230339_reads_chr22.blow5
REF=/genome/hg38noAlt.idx
OUT=/data/bonwon/fq_layer_sweep_chr22_230339
N_GPUS=1
# ------------------

set -e

die() { echo "Error: $*" >&2; exit 1; }

[ -x ./slorado ] || die "./slorado not found, run from the slorado directory"
mkdir -p $OUT

for g in $(seq 0 $((N_GPUS - 1))); do
    nohup test/scripts/fq_sweep.sh layer $MODEL $READS $REF $OUT $g $N_GPUS > $OUT/gpu$g.log 2>&1 &
    echo "started cuda:$g (pid $!), log: $OUT/gpu$g.log"
done

echo "progress: tail -f $OUT/gpu*.log"
echo "stop:     pkill -f 'fq_sweep.sh layer'; pkill slorado; pkill minimap2"
