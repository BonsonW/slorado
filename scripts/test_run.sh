#!/bin/bash

make clean && make -j cuda=1 CUDA_ROOT=/install/cuda-11.8/

DEVICE=cuda:2,3
# BLOW5=/data/slow5-testdata/hg2_prom_lsk114_5khz_subsubsample/PGXXXX230339_reads_20k.blow5
BLOW5=/data/bonwon/slorado/test/one_5khz.blow5
MODEL=fast

/usr/bin/time  --verbose ./slorado basecaller -c 10000 -C 1000 -B500M -x $DEVICE --verbose 6 /install/dorado-0.3.4/models/dna_r10.4.1_e8.2_400bps_${MODEL}@v4.2.0 $BLOW5 > reads.fastq