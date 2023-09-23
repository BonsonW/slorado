make clean && make cuda=1 koi=1 -j CUDA_ROOT=/install/cuda-11.3/

./slorado basecaller /install/dorado-0.3.4/models/dna_r10.4.1_e8.2_400bps_sup@v4.2.0 /data/slow5-testdata/hg2_prom_lsk114_subsubsample/reads.blow5 -x cuda:0,1,2,3 -o reads.fastq -C 182 --verbose 5 -c 10000