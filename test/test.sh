#!/bin/bash

FAST="dna_r10.4.1_e8.2_400bps_fast@v4.2.0"

# terminate script
die() {
	echo "$1" >&2
	echo
	exit 1
}

usage() {
    echo "usage: $0 [mem] [chr22]"
    echo "  mem    run the basecaller under valgrind"
    echo "  chr22     also run the chr22 1k read test (hg38 chr22 reference)"
    exit 1
}

mem=0
chr22=0

for arg in "$@"; do
    case $arg in
        mem ) mem=1 ;;
        chr22 ) chr22=1 ;;
        * ) usage ;;
    esac
done

ex() {
    if [ $mem -eq 1 ]; then
        ${VALGRIND} --error-exitcode=1  --leak-check=full --show-leak-kinds=all --suppressions=test/valgrind.supp --gen-suppressions=all  "$@"
    else
        "$@"
    fi
}

test -z "$DEVICE" && DEVICE=cuda:0
test -z "$VALGRIND" && VALGRIND=valgrind

download_model () {
    test -e $1.zip && rm $1.zip
    test -d $1 && rm -r $1
    wget https://cdn.oxfordnanoportal.com/software/analysis/dorado/${1}.zip -O $1.zip || die "Downloading the model failed"
    unzip $1.zip || die "Unzipping the model failed"
    test -d models || mkdir models || die "Creating the models directory failed"
    mv $1 models/ || die "Moving the model failed"
    rm -f $1.zip || die "Removing the model failed"
}

download_minimap2 () {
    uname -m || die "Could not determine the architecture. "
    ARCH=$(uname -m)

    if [ ${ARCH} = "x86_64" ];
    then
        wget https://github.com/lh3/minimap2/releases/download/v2.24/minimap2-2.24_x64-linux.tar.bz2
        tar xf minimap2-2.24_x64-linux.tar.bz2
        mv minimap2-2.24_x64-linux minimap2
        rm minimap2-2.24_x64-linux.tar.bz2
    elif [ ${ARCH} = "aarch64" ];
    then
        wget https://github.com/lh3/minimap2/releases/download/v2.24/minimap2-2.24.tar.bz2
        tar xf minimap2-2.24.tar.bz2
        mv minimap2-2.24 minimap2
        rm minimap2-2.24.tar.bz2
        cd minimap2
        make arm_neon=1 aarch64=1
        cd ..
    else
        die "Unsupported architecture"
    fi
}

check_accuracy () {
    if (( $(echo "$1 >= $2" | bc -l) ));
    then
        return 0
    fi

    die "Failed accuracy test with value of $1 (expected >= $2)"
}

# basecall a BLOW5 file, map it against a reference and check the median identity
# usage: basecall_and_check <blow5> <reference.fa> <min_accuracy> [extra slorado args...]
basecall_and_check () {
    BLOW5=$1
    REF=$2
    MIN_ACC=$3
    shift 3

    test -e $BLOW5 || die "Missing test data $BLOW5"
    test -e $REF || die "Missing reference $REF"

    ex ./slorado basecaller models/$FAST $BLOW5 "$@" --device $DEVICE -v 6 > test/tmp.fastq || die "Running the tool failed"
    minimap2/minimap2 -cx map-ont $REF test/tmp.fastq --secondary=no > test/tmp.paf || die "minimap2 failed"
    MEDIAN=$(awk '{print $10/$11}' test/tmp.paf | datamash median 1)
    echo "accuracy: $MEDIAN"
    check_accuracy $MEDIAN $MIN_ACC
}

test -d models/$FAST || download_model $FAST
test -e minimap2/minimap2 || download_minimap2

echo "test: PGXXXX230339 reads_1 vs chr3:34011000-34012000"
basecall_and_check test/PGXXXX230339/reads_1.blow5 test/chr3_34011000_34012000.fa 0.8 -c 1000 -C 1

if [ $chr22 -eq 1 ]; then
    echo "test: HG2 PGXXXX230339 chr22:23700000-23900000 1k reads vs hg38 chr22"
    basecall_and_check test/hg2_PGXXXX230339_chr22_23700000_23900000_1k_reads.blow5 test/hg38_chr22.fa 0.9
fi

echo "tests passed!"
