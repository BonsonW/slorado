#!/bin/bash

# =========================================================================================================

# run like: ./test/extensive_500.sh
# if you want to check flash attention too: ./test/extensive_500.sh flash

# =========================================================================================================
# change these

MINIMAP2=minimap2/minimap2 # this will be automatically download
DATAMASH=datamash
SLORADO=./slorado

BUILD_FROM_SOURCE=0 # run only if in slorado repo, required for memory checks
RUN_500K=0 # run 500k DNA dataset for HAC

# batch sizes for each model
FAST_BATCH=1000
HAC_BATCH=800
SUP_BATCH=500

# basecaller options
NTHREADS=32
CHUNKSIZE=10000
READ_MEM=500M
READ_BATCH=2000

# models
FAST="dna_r10.4.1_e8.2_400bps_fast@v5.0.0"
HAC="dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
SUP="dna_r10.4.1_e8.2_400bps_sup@v5.0.0"

FAST_RNA="rna004_130bps_fast@v5.1.0"
HAC_RNA="rna004_130bps_hac@v5.1.0"
SUP_RNA="rna004_130bps_sup@v5.1.0"

# =========================================================================================================
# make sure these exist, will automatically check at start

REF_DNA="/genome/hg38noAlt.idx"
REF_RNA="/genome/gencode.v40.transcripts.fa"

SUBSUBSAMPLE="/data/slow5-testdata/hg2_prom_lsk114_5khz_subsubsample/PGXXXX230339_reads_20k.blow5"
SUBSUBSAMPLE_RNA="/data/slow5-testdata/uhr_prom_rna004_subsubsample/PNXRXX240011_reads_20k.blow5"

# optional only checks if $RUN_500K = 1
SUBSAMPLE="/data/slow5-testdata/hg2_prom_lsk114_5khz_subsample/PGXXXX230339_reads_500k.blow5"

# =========================================================================================================

die() {
    echo "Error: $@" >&2
    exit 1
}

if [ "$1" = 'flash' ]; then
    flash=1
else
    flash=0
fi

ex() {
    "$@"
}

check_accuracy() {
    case $1 in
    $FAST )
        if (( $(echo "$2 >= 0.93" | bc -l) ));
        then
            return 0
        fi
        ;;

    $HAC )
        if (( $(echo "$2 >= 0.97" | bc -l) ));
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
    $FAST_RNA )
        if (( $(echo "$2 >= 0.91" | bc -l) ));
        then
            return 0
        fi
        ;;

    $HAC_RNA )
        if (( $(echo "$2 >= 0.95" | bc -l) ));
        then
            return 0
        fi
        ;;

    $SUP_RNA )
        if (( $(echo "$2 >= 0.97" | bc -l) ));
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

download_minimap2() {
    wget https://github.com/lh3/minimap2/releases/download/v2.24/minimap2-2.24_x64-linux.tar.bz2
    tar xf minimap2-2.24_x64-linux.tar.bz2
    mv minimap2-2.24_x64-linux minimap2
    rm minimap2-2.24_x64-linux.tar.bz2
}

download_model() {
    test -e $1.zip && rm $1.zip
    test -d $1 && rm -r $1
    wget https://cdn.oxfordnanoportal.com/software/analysis/dorado/${1}.zip -O $1.zip || die "Downloading the model failed"
    unzip $1.zip || die "Unzipping the model failed"
    test -d models || mkdir models || die "Creating the models directory failed"
    mv $1 models/ || die "Moving the model failed"
    rm -f $1.zip || die "Removing the model failed"
}

check_acc_dna() {
    $MINIMAP2 -cx map-ont $REF_DNA -t $NTHREADS tmp.fastq --secondary=no > tmp.paf || die "minimap2 failed"
    MEDIAN=$(awk '{print $10/$11}' tmp.paf | $DATAMASH median 1)
    check_accuracy $1 $MEDIAN
}

check_acc_rna() {
    $MINIMAP2 -cx splice -uf -k14 $REF_RNA -t $NTHREADS --secondary=no tmp.fastq > tmp.paf || die "minimap2 failed"
    MEDIAN=$(awk '{print $10/$11}' tmp.paf | $DATAMASH median 1)
    check_accuracy $1 $MEDIAN
}

# check files
test -e $REF_DNA || die "missing DNA reference genome"
test -e $REF_RNA || die "missing RNA reference genome"
test -e $SUBSUBSAMPLE || die "missing DNA BLOW5 subsubsample"
test -e $RNA_SUBSUBSAMPLE || die "missing RNA BLOW5 subsubsample"

if [ $RUN_500K -eq 1 ]; then
    test -e $SUBSAMPLE || die "missing DNA BLOW5 subsample"
fi

# download minimap2
test -e $MINIMAP2 || download_minimap2
$DATAMASH --version > /dev/null || die "datamash not found in PATH"

# download models
test -d models/$FAST || download_model $FAST
test -d models/$HAC || download_model $HAC
test -d models/$SUP || download_model $SUP

test -d models/$FAST_RNA || download_model $FAST_RNA
test -d models/$HAC_RNA || download_model $HAC_RNA
test -d models/$SUP_RNA || download_model $SUP_RNA

# memory check with asan if building from source

if [ $BUILD_FROM_SOURCE -eq 1 ]; then
    make clean && make -j asan=1

    echo "Memory Check - CPU - FAST model - 1 5khz reads"
    ex $SLORADO basecaller models/$FAST test/5khz_r10/one_5khz.blow5 -xcpu -c200 -K10 > test/tmp.fastq  || die "Running the tool failed"

    echo "Memory Check - CPU - FAST model - 2 batch 1 thread"
    ex $SLORADO basecaller models/$FAST test/4khz_r10/10_reads.blow5 -xcpu -c200 -K5 -t1 > test/tmp.fastq  || die "Running the tool failed"

    echo "Memory Check - CPU - FAST model - incomplete batch 1 thread"
    ex $SLORADO basecaller models/$FAST test/4khz_r10/10_reads.blow5 -xcpu -c200 -K6 -t1 > test/tmp.fastq  || die "Running the tool failed"

    echo "Memory Check - CPU - FAST model - 2 batch 2 thread"
    ex $SLORADO basecaller models/$FAST test/4khz_r10/10_reads.blow5 -xcpu -c200 -K5 -t2 > test/tmp.fastq  || die "Running the tool failed"

    echo "Memory Check - CPU - FAST model - incomplete batch 2 thread"
    ex $SLORADO basecaller models/$FAST test/4khz_r10/10_reads.blow5 -xcpu -c200 -K6 -t2 > test/tmp.fastq  || die "Running the tool failed"

    echo "Memory Check - CPU - FAST model - 2 batch 3 thread"
    ex $SLORADO basecaller models/$FAST test/4khz_r10/10_reads.blow5 -xcpu -c200 -K5 -t3 > test/tmp.fastq  || die "Running the tool failed"

    echo "Memory Check - CPU - FAST model - incomplete batch 3 thread"
    ex $SLORADO basecaller models/$FAST test/4khz_r10/10_reads.blow5 -xcpu -c200 -K6 -t3 > test/tmp.fastq  || die "Running the tool failed"
fi

# accuracy check DNA

if [ $BUILD_FROM_SOURCE -eq 1 ]; then
    make clean && make -j cuda=1
fi

if [ $RUN_500K -eq 1 ]; then
    echo "GPU - HAC model - 500k reads"
    ex $SLORADO basecaller models/$HAC $SUBSAMPLE -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $HAC_BATCH > tmp.fastq || die "Running the tool failed"
    check_accuracy $HAC $MEDIAN
    echo ""
    echo "********************************************************************"
fi

echo "GPU - FAST model - 20k reads"
ex $SLORADO basecaller models/$FAST $SUBSUBSAMPLE -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $FAST_BATCH > tmp.fastq || die "Running the tool failed"
check_acc_dna $FAST
echo ""
echo "********************************************************************"

echo "GPU - HAC model - 20k reads"
ex $SLORADO basecaller models/$HAC $SUBSUBSAMPLE -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $HAC_BATCH > tmp.fastq || die "Running the tool failed"
check_acc_dna $HAC
echo ""
echo "********************************************************************"

echo "GPU - SUP model - 20k reads"
ex $SLORADO basecaller models/$SUP $SUBSUBSAMPLE -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $SUP_BATCH > tmp.fastq || die "Running the tool failed"
check_acc_dna $SUP
echo ""
echo "********************************************************************"

# accuracy check RNA
echo "GPU - FAST RNA model - 20k reads"
ex $SLORADO basecaller models/$FAST_RNA $SUBSUBSAMPLE_RNA -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $FAST_BATCH > tmp.fastq || die "Running the tool failed"
check_acc_rna $FAST
echo ""
echo "********************************************************************"

echo "GPU - HAC RNA model - 20k reads"
ex $SLORADO basecaller models/$HAC_RNA $SUBSUBSAMPLE_RNA -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $HAC_BATCH > tmp.fastq || die "Running the tool failed"
check_acc_rna $HAC
echo ""
echo "********************************************************************"

echo "GPU - SUP RNA model - 20k reads"
ex $SLORADO basecaller models/$SUP_RNA $SUBSUBSAMPLE_RNA -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $SUP_BATCH > tmp.fastq || die "Running the tool failed"
check_acc_rna $SUP
echo ""
echo "********************************************************************"

# check flash
if [ $flash -eq 1 ]; then
    echo "GPU - SUP model (flash) - 20k reads"
    ex $SLORADO basecaller models/$SUP $SUBSUBSAMPLE --flash yes -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $SUP_BATCH > tmp.fastq || die "Running the tool failed"
    check_acc_dna $SUP
    echo ""
    echo "********************************************************************"

    echo "GPU - SUP RNA model (flash) - 20k reads"
    ex $SLORADO basecaller models/$SUP_RNA $SUBSUBSAMPLE_RNA --flash yes -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $SUP_BATCH > tmp.fastq || die "Running the tool failed"
    check_acc_rna $SUP
    echo ""
    echo "********************************************************************"
fi