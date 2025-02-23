#!/bin/bash

DEVICE=cuda:2
MODEL=$1
BATCH_SIZE=500
THREADS=40
VERSION=4.2.0

if [ "$MODEL" = "fast" ]; then
    BATCH_SIZE=1000
fi
if [ "$MODEL" = "hac" ]; then
    BATCH_SIZE=1000
fi
if [ "$MODEL" = "sup" ]; then
    BATCH_SIZE=300
fi

# BLOW5=/data/slow5-testdata/hg2_prom_lsk114_5khz_subsample/PGXXXX230339_reads_500k.blow5
BLOW5=/data/slow5-testdata/hg2_prom_lsk114_5khz_subsubsample/PGXXXX230339_reads_20k.blow5
# BLOW5=/data/bonwon/slorado/test/4khz_r10/one_read.blow5
# BLOW5=test/4khz_r10/1k_reads.blow5

/usr/bin/time  --verbose ./slorado basecaller -t $THREADS -C $BATCH_SIZE -x $DEVICE --verbose 7 models/dna_r10.4.1_e8.2_400bps_${MODEL}@v${VERSION} $BLOW5 > reads.fastq

./scripts/calculate_basecalling_accuracy.sh /genome/hg38noAlt.idx reads.fastq 