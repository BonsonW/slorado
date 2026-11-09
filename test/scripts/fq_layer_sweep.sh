#!/bin/bash
# Per-layer fake-quant sweep on several GPUs: starts one "fq_sweep.sh layer" per GPU in the background,
# each taking every n_gpus-th run, all writing to the same outdir. Rerun the same command to resume.
#
# usage: test/scripts/fq_layer_sweep.sh <model> <reads.blow5> <ref.idx|fa> <outdir> <n_gpus>
# Run from the slorado directory. Progress: tail -f <outdir>/gpu*.log

set -e

die() { echo "Error: $*" >&2; exit 1; }
[ $# -eq 5 ] || die "usage: $0 <model> <reads.blow5> <ref.idx|fa> <outdir> <n_gpus>"

MODEL=$1
READS=$2
REF=$3
OUT=$4
N_GPUS=$5

[ -x ./slorado ] || die "./slorado not found, run from the slorado directory"
mkdir -p $OUT

for g in $(seq 0 $((N_GPUS - 1))); do
    nohup test/scripts/fq_sweep.sh layer $MODEL $READS $REF $OUT $g $N_GPUS > $OUT/gpu$g.log 2>&1 &
    echo "started cuda:$g (pid $!), log: $OUT/gpu$g.log"
done

echo "progress: tail -f $OUT/gpu*.log"
echo "stop:     pkill -f 'fq_sweep.sh layer'; pkill slorado"
