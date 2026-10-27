#!/bin/bash
# Fake-quant sensitivity sweep: basecall with each FQ_FORMAT x FQ_LAYERS combination, align, and
# write per-read identity (matches / alignment block length of the longest primary alignment).
# Results: <outdir>/<format>__<layer tag>/{reads.fastq,reads.paf,identity.tsv}. Finished runs are skipped.
#
# usage: scripts/fq_sweep.sh <model> <reads.blow5> <ref.idx|fa> <outdir>
# env:   FORMATS   formats to test (default: all)
#        LAYERS    space-separated FQ_LAYERS values, one run each (default: "*"). Each value is a
#                  comma-separated list of globs over tx<i>.{wqkv,out_proj,fc1,fc2}, upsample, crf
#                  e.g. per layer type: LAYERS='tx*.wqkv tx*.out_proj tx*.fc1 tx*.fc2 upsample crf'
#                  e.g. per block:      LAYERS="$(seq -f 'tx%g.*' 0 17)"
#                  e.g. per layer:      LAYERS="$(for i in $(seq 0 17); do echo tx$i.{wqkv,out_proj,fc1,fc2}; done) upsample crf"
#        TARGET    FQ_TARGET: wa (default) | w | a
#        SLORADO_ARGS extra slorado args (default: -x cuda:0)

set -e
set -f  # LAYERS globs are for slorado, not the shell

die() { echo "Error: $*" >&2; exit 1; }
[ $# -eq 4 ] || die "usage: $0 <model> <reads.blow5> <ref.idx|fa> <outdir>"

MODEL=$1; READS=$2; REF=$3; OUT=$4
SLORADO=${SLORADO:-./slorado}
MINIMAP2=${MINIMAP2:-minimap2}
NTHREADS=${NTHREADS:-16}
FORMATS=${FORMATS:-"int8 int4 fp8_e4m3 fp8_e5m2 mxint8 mxfp8_e4m3 mxfp8_e5m2 mxfp6_e2m3 mxfp6_e3m2 mxfp4"}
LAYERS=${LAYERS:-"*"}
TARGET=${TARGET:-wa}
SLORADO_ARGS=${SLORADO_ARGS:-"-x cuda:0"}

$MINIMAP2 --version > /dev/null || die "minimap2 not found, set MINIMAP2"

run() { # <format> <layer globs> <run dir>
    local dir=$3
    [ -s $dir/identity.tsv ] && { echo "skip $dir"; return; }
    mkdir -p $dir
    echo "run $dir"
    FQ_FORMAT=$1 FQ_LAYERS=$2 FQ_TARGET=$TARGET $SLORADO basecaller $SLORADO_ARGS -o $dir/reads.fastq $MODEL $READS 2> $dir/slorado.log || die "slorado failed, see $dir/slorado.log"
    $MINIMAP2 -cx map-ont --secondary=no -t $NTHREADS $REF $dir/reads.fastq > $dir/reads.paf 2> $dir/minimap2.log || die "minimap2 failed"
    # longest primary alignment per read; identity = col10 / col11
    awk -v OFS='\t' '/tp:A:P/ && $11 > len[$1] { len[$1] = $11; id[$1] = $10 / $11 }
        END { print "read_id", "identity"; for (r in id) print r, id[r] }' $dir/reads.paf > $dir/identity.tsv.tmp
    printf "format\tlayers\ttarget\tn_reads\n%s\t%s\t%s\t%s\n" $1 "$2" $TARGET $(( $(wc -l < $dir/reads.fastq) / 4 )) > $dir/info.tsv
    mv $dir/identity.tsv.tmp $dir/identity.tsv
}

run none "*" $OUT/none
for layers in $LAYERS; do
    tag=$(echo "$layers" | sed 's/^\*$/all/; s/[^A-Za-z0-9]\+/_/g; s/^_//; s/_$//')
    for f in $FORMATS; do
        run $f "$layers" $OUT/${f}__${tag}_${TARGET}
    done
done
echo "done. plot with: pyvenv/bin/python scripts/fq_plot.py $OUT"
