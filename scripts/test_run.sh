#!/bin/bash

make clean && make -j koi=1 cuda=1 CUDA_ROOT=/install/cuda-11.8/

./slorado basecaller -c 10000 -C 128 -B500M --verbose 6 /install/dorado-0.3.4/models/dna_r10.4.1_e8.2_400bps_sup@v4.2.0 test/oneread_r10.blow5 > reads.fastq