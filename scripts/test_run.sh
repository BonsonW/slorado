#!/bin/bash

make clean && make -j cuda=1 CUDA_ROOT=/install/cuda-11.8/

BLOW20K="/data/slow5-testdata/hg2_prom_lsk114_5khz_subsubsample/PGXXXX230339_reads_20k.blow5"
BLOW2K="reads_2k.blow5"
BLOW5HUND="reads_5_hund.blow5"
BLOW1="test/one_5khz.blow5"

MODEL=fast

./slorado basecaller --verbose 5 -x cuda:0 -c 10000 -C 1000 -B500M /install/dorado-0.3.4/models/dna_r10.4.1_e8.2_400bps_${MODEL}@v4.2.0 ${BLOW5HUND} > reads.fastq
