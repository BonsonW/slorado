#!/bin/bash

# Ad-hoc test for v6.0.0 model support against local build.
# Usage: ./test/v6_models.sh [device]
# device defaults to cuda:0

DEVICE="${1:-cuda:0}"

HAC_V6="dna_r10.4.1_e8.2_400bps_hac@v6.0.0"
HAC_RNA_V6="rna004_hac@v6.0.0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR=${SCRIPT_DIR}/slorado_test_ext_dat

SUBSUBSAMPLE="test/PGXXXX230339/reads_1k.blow5"
# SUBSUBSAMPLE="${DATA_DIR}/PGXXXX230339_reads_20k.blow5"
SUBSUBSAMPLE_RNA="${DATA_DIR}/PNXRXX240011_reads_20k.blow5"
REF_DNA="${DATA_DIR}/genome/hg38noAlt.idx"
REF_RNA="${DATA_DIR}/genome/gencode.v40.transcripts.fa"

die() {
    echo "Error: $@" >&2
    exit 1
}

download_model() {
    [ -d models/$1 ] && return 0
    wget https://cdn.oxfordnanoportal.com/software/analysis/dorado/$1.zip -O $1.zip || die "Downloading $1 failed"
    unzip $1.zip || die "Unzipping $1 failed"
    mkdir -p models
    mv $1 models/ || die "Moving $1 failed"
    rm -f $1.zip
}

check_acc() {
    local threshold=$1
    MEDIAN=$(minimap2 -cx map-ont $REF_DNA -t$(nproc) tmp.fastq --secondary=no 2>/dev/null | awk '{print $10/$11}' | datamash median 1)
    (( $(echo "$MEDIAN >= $threshold" | bc -l) )) || die "accuracy $MEDIAN < $threshold"
    echo "accuracy: $MEDIAN (>= $threshold) OK"
}

check_acc_rna() {
    local threshold=$1
    MEDIAN=$(minimap2 -cx splice -uf -k14 $REF_RNA -t$(nproc) --secondary=no tmp.fastq 2>/dev/null | awk '{print $10/$11}' | datamash median 1)
    (( $(echo "$MEDIAN >= $threshold" | bc -l) )) || die "accuracy $MEDIAN < $threshold"
    echo "accuracy: $MEDIAN (>= $threshold) OK"
}

test -e $SUBSUBSAMPLE  || die "missing $SUBSUBSAMPLE (run extensive.sh first to download test data)"
test -e $REF_DNA       || die "missing $REF_DNA"
test -e $SUBSUBSAMPLE_RNA || die "missing $SUBSUBSAMPLE_RNA"
test -e $REF_RNA       || die "missing $REF_RNA"

download_model $HAC_V6
download_model $HAC_RNA_V6

echo "--- DNA HAC v6.0.0 (FLSTM) ---"
./slorado basecaller models/$HAC_V6 $SUBSUBSAMPLE -x$DEVICE -c10000 -C256 > tmp.fastq || die "basecaller failed"
check_acc 0.97

echo "--- RNA HAC v6.0.0 ---"
./slorado basecaller models/$HAC_RNA_V6 $SUBSUBSAMPLE_RNA -x$DEVICE -c10000 -C256 > tmp.fastq || die "basecaller failed"
check_acc_rna 0.95

echo "all v6 model tests passed!"
