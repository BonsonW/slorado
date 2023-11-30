#!/bin/bash

make clean && make -j koi=1 cuda=1 CUDA_ROOT=/install/cuda-11.8/

LARGE="/data/slow5-testdata/hg2_prom_lsk114_5khz_subsubsample/PGXXXX230339_reads_20k.blow5"
SMALL="test/one_5khz.blow5"

./slorado basecaller -x cuda:all -c 10000 -C 1000 -B500M /install/dorado-0.3.4/models/dna_r10.4.1_e8.2_400bps_fast@v4.2.0 $LARGE > reads.fastq