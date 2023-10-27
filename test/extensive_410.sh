#!/bin/bash

MINIMAP2=minimap2/minimap2
REFERENC_GENOME="/genome/hg38noAlt.idx"

SUBSAMPLE="/data/slow5-testdata/hg2_prom_lsk114_subsubsample/reads.blow5"

FAST="dna_r10.4.1_e8.2_400bps_fast@v4.1.0"
HAC="dna_r10.4.1_e8.2_400bps_hac@v4.1.0"
SUP="dna_r10.4.1_e8.2_400bps_sup@v4.1.0"

die() {
    echo "Error: $@" >&2
    exit 1
}

if [ "$1" = 'mem' ]; then
    mem=1
else
    mem=0
fi

ex() {
    "$@"
}

check_accuracy () {
    case $1 in
    $FAST )
        if (( $(echo "$2 >= 0.90" | bc -l) ));
        then
            return 0
        fi
        ;;

    $HAC )
        if (( $(echo "$2 >= 0.96" | bc -l) ));
        then
            return 0
        fi
        ;;

    $SUP )
        if (( $(echo "$2 >= 0.98" | bc -l) ));
        then
            return 0
        fi
        ;;

    *)
        die "Invalid model provided"
        ;;
    esac

    die "$1 failed accuracy test with value of $2"
}

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
    wget https://github.com/lh3/minimap2/releases/download/v2.24/minimap2-2.24_x64-linux.tar.bz2
    tar xf minimap2-2.24_x64-linux.tar.bz2
    mv minimap2-2.24_x64-linux minimap2
    rm minimap2-2.24_x64-linux.tar.bz2
}

# download models
test -d models/$FAST || download_model $FAST
test -d models/$HAC || download_model $HAC
test -d models/$SUP || download_model $SUP

# download minimap2
test -e minimap2/minimap2 || download_minimap2

# accuracy check
make clean && make -j cuda=1 koi=1 CUDA_ROOT=/data/install/cuda-11.8/

echo "GPU - FAST model - 20k reads"
ex  ./slorado basecaller models/$FAST $SUBSAMPLE -xcuda:0,1,2,3 -B500M -c10000 -C1900 > test/tmp.fastq || die "Running the tool failed"
minimap2/minimap2 -cx map-ont $REFERENC_GENOME test/tmp.fastq --secondary=no > test/tmp.paf || die "minimap2 failed"
MEDIAN=$(awk '{print $10/$11}' test/tmp.paf | datamash median 1)
check_accuracy $FAST $MEDIAN
echo ""
echo "********************************************************************"

echo "GPU - HAC model - 20k reads"
ex  ./slorado basecaller models/$HAC $SUBSAMPLE -xcuda:0,1,2,3 -B500M -c10000 -C832 > test/tmp.fastq || die "Running the tool failed"
minimap2/minimap2 -cx map-ont $REFERENC_GENOME test/tmp.fastq --secondary=no > test/tmp.paf || die "minimap2 failed"
MEDIAN=$(awk '{print $10/$11}' test/tmp.paf | datamash median 1 || die "datamash failed")
check_accuracy $HAC $MEDIAN
echo ""
echo "********************************************************************"

echo "GPU - SUP model - 20k reads"
ex  ./slorado basecaller models/$SUP $SUBSAMPLE -xcuda:0,1,2,3 -B500M -c10000 -C192 > test/tmp.fastq || die "Running the tool failed"
minimap2/minimap2 -cx map-ont $REFERENC_GENOME test/tmp.fastq --secondary=no > test/tmp.paf || die "minimap2 failed"
MEDIAN=$(awk '{print $10/$11}' test/tmp.paf | datamash median 1 || die "datamash failed")
check_accuracy $SUP $MEDIAN
echo ""
echo "********************************************************************"
