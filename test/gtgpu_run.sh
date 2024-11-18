#!/bin/bash

DEVICE=cuda:1
MODEL=hac
BATCHSIZE=700
CHUNKSIZE=10000
THREADS=40

# BLOW5=/data/slow5-testdata/hg2_prom_lsk114_5khz_subsample/PGXXXX230339_reads_500k.blow5
BLOW5=/data/slow5-testdata/hg2_prom_lsk114_5khz_subsubsample/PGXXXX230339_reads_20k.blow5
# BLOW5=/data/bonwon/slorado/test/one_5khz.blow5
# BLOW5=test/4khz_r10/1k_reads.blow5

/usr/bin/time  --verbose ./slorado basecaller -t $THREADS -c $CHUNKSIZE -C $BATCHSIZE -B500M -x $DEVICE --verbose 5 /install/dorado-0.3.4/models/dna_r10.4.1_e8.2_400bps_${MODEL}@v4.2.0 $BLOW5 > reads.fastq
