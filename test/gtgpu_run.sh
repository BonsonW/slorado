#!/bin/bash

DEVICE=cuda:2,3
MODEL=$1
BATCH_SIZE=500
THREADS=40

if [ "$MODEL" = "fast" ]; then
    BATCH_SIZE=1000
fi
if [ "$MODEL" = "hac" ]; then
    BATCH_SIZE=400
fi
if [ "$MODEL" = "sup" ]; then
    BATCH_SIZE=200
fi

# BLOW5=/data/slow5-testdata/hg2_prom_lsk114_5khz_subsample/PGXXXX230339_reads_500k.blow5
BLOW5=/data/slow5-testdata/hg2_prom_lsk114_5khz_subsubsample/PGXXXX230339_reads_20k.blow5
# BLOW5=/data/bonwon/slorado/test/4khz_r10/one_read.blow5
# BLOW5=test/4khz_r10/1k_reads.blow5

/usr/bin/time  --verbose ./slorado basecaller -t $THREADS -C $BATCH_SIZE -x $DEVICE --verbose 5 /install/dorado-0.3.4/models/dna_r10.4.1_e8.2_400bps_${MODEL}@v4.2.0 $BLOW5 > reads.fastq

./scripts/calculate_basecalling_accuracy.sh /genome/hg38noAlt.idx reads.fastq 