#!/bin/bash

MINIMAP2="/install/minimap2-2.26/minimap2"
REFERENC_GENOME="/genome/hg38noAlt.idx"

SUBSUBSAMPLE="/data/slow5-testdata/hg2_prom_lsk114_5khz_subsubsample/PGXXXX230339_reads_20k.blow5"
SUBSAMPLE="/data/slow5-testdata/hg2_prom_lsk114_5khz_subsample/PGXXXX230339_reads_500k.blow5"

FAST="dna_r10.4.1_e8.2_400bps_fast@v4.2.0"
HAC="dna_r10.4.1_e8.2_400bps_hac@v4.2.0"
SUP="dna_r10.4.1_e8.2_400bps_sup@v4.2.0"

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

    *)
        die "Invalid model provided"
        ;;
    esac

    die "$1 failed accuracy test with value of $2"
}

check_qscore_accuracy () {
    case $1 in
    $FAST )
        if (( $(echo "$2 <= 0.36" | bc -l) ));
        then
            return 0
        fi
        ;;

    $HAC )
        if (( $(echo "$2 <= 0.36" | bc -l) ));
        then
            return 0
        fi
        ;;

    $SUP )
        if (( $(echo "$2 <= 0.36" | bc -l) ));
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

qscore_compare() {
    fastq_a=$2
    fastq_b=$3
    model=$1

    tmp_a="a.tmp"
    tmp_b="b.tmp"

    if test -f $tmp_a; then
    rm $tmp_a
    fi

    if test -f $tmp_b; then
    rm $tmp_b
    fi

    echo "calculating avg qscores for \"${fastq_a}\"..."
    awk 'NR % 4 == 0' $fastq_a | \
    while read line; do
        awk -v qstring="$line" -f scripts/get_qstring_score.awk >> $tmp_a
    done

    echo "calculating avg qscores for \"${fastq_b}\"..."
    awk 'NR % 4 == 0' $fastq_b | \
    while read line; do
        awk -v qstring="$line" -f scripts/get_qstring_score.awk >> $tmp_b
    done

    echo "comparing avg qscores..."
    sum=0
    n=0

    shopt -s lastpipe
    while
        read score_a &&
        read score_b <&3
    do
        diff=$(echo "$score_a - $score_b" | bc)
        diff=$(echo ${diff#-})
        sum=$(echo "$sum + $diff" | bc)
        n=$(($n+1))
    done < $tmp_a 3<$tmp_b

    avg=$(echo "scale=5; $sum / $n" | bc)

    rm $tmp_a
    rm $tmp_b

    check_qscore_accuracy $model $avg
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

# memory check
make clean && make -j asan=1

echo "Memory Check - CPU - FAST model - 1 5khz reads"
ex ./slorado basecaller models/$FAST test/5khz_r10/one_5khz.blow5 -xcpu -c200 -K10 > test/tmp.fastq  || die "Running the tool failed"

echo "Memory Check - CPU - FAST model - 2 batch 1 thread"
ex ./slorado basecaller models/$FAST test/4khz_r10/10_reads.blow5 -xcpu -c200 -K5 -t1 > test/tmp.fastq  || die "Running the tool failed"

echo "Memory Check - CPU - FAST model - incomplete batch 1 thread"
ex ./slorado basecaller models/$FAST test/4khz_r10/10_reads.blow5 -xcpu -c200 -K6 -t1 > test/tmp.fastq  || die "Running the tool failed"

echo "Memory Check - CPU - FAST model - 2 batch 2 thread"
ex ./slorado basecaller models/$FAST test/4khz_r10/10_reads.blow5 -xcpu -c200 -K5 -t2 > test/tmp.fastq  || die "Running the tool failed"

echo "Memory Check - CPU - FAST model - incomplete batch 2 thread"
ex ./slorado basecaller models/$FAST test/4khz_r10/10_reads.blow5 -xcpu -c200 -K6 -t2 > test/tmp.fastq  || die "Running the tool failed"

echo "Memory Check - CPU - FAST model - 2 batch 3 thread"
ex ./slorado basecaller models/$FAST test/4khz_r10/10_reads.blow5 -xcpu -c200 -K5 -t3 > test/tmp.fastq  || die "Running the tool failed"

echo "Memory Check - CPU - FAST model - incomplete batch 3 thread"
ex ./slorado basecaller models/$FAST test/4khz_r10/10_reads.blow5 -xcpu -c200 -K6 -t3 > test/tmp.fastq  || die "Running the tool failed"

# accuracy check
make clean && make -j cuda=1

echo "GPU - HAC model - 500k reads"
ex ./slorado basecaller models/$HAC $SUBSAMPLE -xcuda:all -B500M -c10000 -C900 > test/tmp.fastq || die "Running the tool failed"
minimap2/minimap2 -cx map-ont $REFERENC_GENOME test/tmp.fastq --secondary=no > test/tmp.paf || die "minimap2 failed"
MEDIAN=$(awk '{print $10/$11}' test/tmp.paf | datamash median 1 || die "datamash failed")
check_accuracy $HAC $MEDIAN
echo ""
echo "********************************************************************"

echo "GPU - FAST model - 20k reads"
ex ./slorado basecaller models/$FAST $SUBSUBSAMPLE -xcuda:all -B500M -c10000 -C1000 > test/tmp.fastq || die "Running the tool failed"
minimap2/minimap2 -cx map-ont $REFERENC_GENOME test/tmp.fastq --secondary=no > test/tmp.paf || die "minimap2 failed"
MEDIAN=$(awk '{print $10/$11}' test/tmp.paf | datamash median 1)
check_accuracy $FAST $MEDIAN
qscore_compare $FAST /data/bonwon/reads_koi_fast_20k.fastq test/tmp.fastq
echo ""
echo "********************************************************************"

echo "GPU - HAC model - 20k reads"
ex ./slorado basecaller models/$HAC $SUBSUBSAMPLE -xcuda:all -B500M -c10000 -C900 > test/tmp.fastq || die "Running the tool failed"
minimap2/minimap2 -cx map-ont $REFERENC_GENOME test/tmp.fastq --secondary=no > test/tmp.paf || die "minimap2 failed"
MEDIAN=$(awk '{print $10/$11}' test/tmp.paf | datamash median 1 || die "datamash failed")
check_accuracy $HAC $MEDIAN
qscore_compare $HAC /data/bonwon/reads_koi_hac_20k.fastq test/tmp.fastq
echo ""
echo "********************************************************************"

echo "GPU - SUP model - 20k reads"
ex ./slorado basecaller models/$SUP $SUBSUBSAMPLE -xcuda:all -B500M -c10000 -C200 > test/tmp.fastq || die "Running the tool failed"
minimap2/minimap2 -cx map-ont $REFERENC_GENOME test/tmp.fastq --secondary=no > test/tmp.paf || die "minimap2 failed"
MEDIAN=$(awk '{print $10/$11}' test/tmp.paf | datamash median 1 || die "datamash failed")
check_accuracy $SUP $MEDIAN
qscore_compare $SUP /data/bonwon/reads_koi_sup_20k.fastq test/tmp.fastq
echo ""
echo "********************************************************************"
