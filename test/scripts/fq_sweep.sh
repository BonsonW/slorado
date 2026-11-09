#!/bin/bash
# Fake-quant sweep: basecall with each fake-quant setting, align with minimap2, and write the identity
# (matches / alignment block length) of every alignment, as scripts/calculate_basecalling_accuracy.sh does.
#
# usage: test/scripts/fq_sweep.sh <all|layer> <model> <reads.blow5> <ref.idx|fa> <outdir> [gpu] [n_gpus]
#   all    quantise tx*.qkv_proj, tx*.up_proj and tx*.down_proj together in every run
#   layer  quantise one layer (tx<i>.qkv_proj, tx<i>.up_proj or tx<i>.down_proj) per run
#   gpu n_gpus  run every n_gpus-th run starting at run number gpu, on cuda:<gpu> (default: 0 1).
#               For 4 GPUs start four copies with gpu = 0, 1, 2, 3 and n_gpus = 4, same outdir.
#
# Each run writes <outdir>/<run>/{reads.fastq,reads.paf,identity.tsv,info.tsv}; <outdir>/none is the fp16
# baseline. Runs with an identity.tsv are skipped, so rerunning the same command resumes.
# Run from the slorado directory. minimap2 is test/tools/bin/minimap2, installed by test/extensive.sh.

set -e

die() { echo "Error: $*" >&2; exit 1; }
[ $# -eq 5 ] || [ $# -eq 7 ] || die "usage: $0 <all|layer> <model> <reads.blow5> <ref.idx|fa> <outdir> [gpu] [n_gpus]"

MODE=$1
MODEL=$2
READS=$3
REF=$4
OUT=$5
GPU=${6:-0}
N_GPUS=${7:-1}

SLORADO=./slorado
MINIMAP2=test/tools/bin/minimap2
NTHREADS=$(( $(nproc) / N_GPUS ))

[ -x $SLORADO ] || die "$SLORADO not found, run from the slorado directory"
[ -x $MINIMAP2 ] || die "$MINIMAP2 not found, run test/extensive.sh once to install it"

# run <name> <format> <gran> <target> <layers>: basecall, align and score one fake-quant setting
run() {
    local name=$1 format=$2 gran=$3 target=$4 layers=$5
    local dir=$OUT/$name

    if [ -s $dir/identity.tsv ]; then
        echo "skip $name"
        return
    fi
    echo "[cuda:$GPU] $(date '+%F %T') run $name"
    mkdir -p $dir

    # 1. basecall
    export FQ_FORMAT=$format FQ_GRAN=$gran FQ_TARGET=$target FQ_LAYERS=$layers
    $SLORADO basecaller -x cuda:$GPU -o $dir/reads.fastq $MODEL $READS \
        < /dev/null 2> $dir/slorado.log || die "slorado failed, see $dir/slorado.log"

    # 2. align
    $MINIMAP2 -cx map-ont --secondary=no -t $NTHREADS $REF $dir/reads.fastq \
        > $dir/reads.paf 2> $dir/minimap2.log || die "minimap2 failed, see $dir/minimap2.log"

    # 3. identity = matches / alignment block length (PAF columns 10 and 11) for every alignment,
    #    as in scripts/calculate_basecalling_accuracy.sh. A read split into several alignments has one row each.
    awk -v OFS='\t' '
        BEGIN { print "read_id", "identity" }
        { print $1, $10 / $11 }
    ' $dir/reads.paf > $dir/identity.tsv.tmp

    # 4. run settings
    local n_reads=$(( $(wc -l < $dir/reads.fastq) / 4 ))
    printf "format\tlayers\tgran\ttarget\tn_reads\n" > $dir/info.tsv
    printf "%s\t%s\t%s\t%s\t%s\n" $format "$layers" $gran $target $n_reads >> $dir/info.tsv

    # identity.tsv is written last, so a run that was cut off is redone next time
    mv $dir/identity.tsv.tmp $dir/identity.tsv
}

# runs: one line per run, "<run name> <FQ_FORMAT> <FQ_GRAN> <FQ_TARGET> <FQ_LAYERS>"
runs_all() {
    L='tx*.qkv_proj,tx*.up_proj,tx*.down_proj'
    T=tx_qkv_proj_tx_up_proj_tx_down_proj
    echo "none none row wa *"
    # weight only
    echo "int8__${T}_w             int8       row    w  $L"
    echo "int8_tensor__${T}_w      int8       tensor w  $L"
    echo "fp8_e4m3__${T}_w         fp8_e4m3   row    w  $L"
    echo "fp8_e4m3_tensor__${T}_w  fp8_e4m3   tensor w  $L"
    echo "fp8_e5m2__${T}_w         fp8_e5m2   row    w  $L"
    echo "fp8_e5m2_tensor__${T}_w  fp8_e5m2   tensor w  $L"
    # activation only
    echo "int8__${T}_a             int8       row    a  $L"
    echo "int8_tensor__${T}_a      int8       tensor a  $L"
    echo "fp8_e4m3__${T}_a         fp8_e4m3   row    a  $L"
    echo "fp8_e4m3_tensor__${T}_a  fp8_e4m3   tensor a  $L"
    echo "fp8_e5m2__${T}_a         fp8_e5m2   row    a  $L"
    echo "fp8_e5m2_tensor__${T}_a  fp8_e5m2   tensor a  $L"
    # microscaling, weights and activations
    echo "mxint8__${T}_wa              mxint8      row    wa $L"
    echo "mxint8_tensor__${T}_wa       mxint8      tensor wa $L"
    echo "mxfp8_e4m3__${T}_wa          mxfp8_e4m3  row    wa $L"
    echo "mxfp8_e4m3_tensor__${T}_wa   mxfp8_e4m3  tensor wa $L"
    echo "mxfp8_e5m2__${T}_wa          mxfp8_e5m2  row    wa $L"
    echo "mxfp8_e5m2_tensor__${T}_wa   mxfp8_e5m2  tensor wa $L"
    echo "mxfp6_e2m3__${T}_wa          mxfp6_e2m3  row    wa $L"
    echo "mxfp6_e2m3_tensor__${T}_wa   mxfp6_e2m3  tensor wa $L"
    echo "mxfp6_e3m2__${T}_wa          mxfp6_e3m2  row    wa $L"
    echo "mxfp6_e3m2_tensor__${T}_wa   mxfp6_e3m2  tensor wa $L"
    echo "mxfp4__${T}_wa               mxfp4       row    wa $L"
    echo "mxfp4_tensor__${T}_wa        mxfp4       tensor wa $L"
}

runs_layer() {
    DEPTH=$(awk -F= '/^[ \t]*depth[ \t]*=/ { gsub(/[ \t]/, "", $2); print $2; exit }' $MODEL/config.toml)
    [ -n "$DEPTH" ] || die "no transformer depth in $MODEL/config.toml"
    echo "none none row wa *"
    for i in $(seq 0 $((DEPTH - 1))); do
        for t in qkv_proj up_proj down_proj; do
            L=tx$i.$t
            T=tx${i}_$t
            # weight only
            echo "int8__${T}_w             int8       row    w  $L"
            echo "int8_tensor__${T}_w      int8       tensor w  $L"
            echo "fp8_e4m3__${T}_w         fp8_e4m3   row    w  $L"
            echo "fp8_e4m3_tensor__${T}_w  fp8_e4m3   tensor w  $L"
            echo "fp8_e5m2__${T}_w         fp8_e5m2   row    w  $L"
            echo "fp8_e5m2_tensor__${T}_w  fp8_e5m2   tensor w  $L"
            # activation only
            echo "int8__${T}_a             int8       row    a  $L"
            echo "int8_tensor__${T}_a      int8       tensor a  $L"
            echo "fp8_e4m3__${T}_a         fp8_e4m3   row    a  $L"
            echo "fp8_e4m3_tensor__${T}_a  fp8_e4m3   tensor a  $L"
            echo "fp8_e5m2__${T}_a         fp8_e5m2   row    a  $L"
            echo "fp8_e5m2_tensor__${T}_a  fp8_e5m2   tensor a  $L"
            # microscaling, weights and activations
            echo "mxint8__${T}_wa              mxint8      row    wa $L"
            echo "mxint8_tensor__${T}_wa       mxint8      tensor wa $L"
            echo "mxfp8_e4m3__${T}_wa          mxfp8_e4m3  row    wa $L"
            echo "mxfp8_e4m3_tensor__${T}_wa   mxfp8_e4m3  tensor wa $L"
            echo "mxfp8_e5m2__${T}_wa          mxfp8_e5m2  row    wa $L"
            echo "mxfp8_e5m2_tensor__${T}_wa   mxfp8_e5m2  tensor wa $L"
            echo "mxfp6_e2m3__${T}_wa          mxfp6_e2m3  row    wa $L"
            echo "mxfp6_e2m3_tensor__${T}_wa   mxfp6_e2m3  tensor wa $L"
            echo "mxfp6_e3m2__${T}_wa          mxfp6_e3m2  row    wa $L"
            echo "mxfp6_e3m2_tensor__${T}_wa   mxfp6_e3m2  tensor wa $L"
            echo "mxfp4__${T}_wa               mxfp4       row    wa $L"
            echo "mxfp4_tensor__${T}_wa        mxfp4       tensor wa $L"
        done
    done
}

set -f  # FQ_LAYERS globs are for slorado, not the shell
case $MODE in
    all)   RUNS=$(runs_all) ;;
    layer) RUNS=$(runs_layer) ;;
    *)     die "mode must be all or layer, got '$MODE'" ;;
esac

echo "$(echo "$RUNS" | wc -l) runs in total; this is cuda:$GPU, doing every ${N_GPUS}th starting at run $GPU"
i=0
while read -r name format gran target layers; do
    [ $((i % N_GPUS)) -eq $GPU ] && run $name $format $gran $target "$layers"
    i=$((i + 1))
done <<< "$RUNS"
echo "[cuda:$GPU] done"
