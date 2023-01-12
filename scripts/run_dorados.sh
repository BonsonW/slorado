#!/bin/bash

MINIMAP2=../minimap2-2.24_x64-linux/minimap2 #path to minimpa2 executable if not added to $PATH
REFERENC_GENOME="/genome/hg38noAlt.idx" #path to reference genome
OUTPUT_DIR="logs"
POD5_FILE="/data/slow5-testdata/hg2_prom_lsk114_subsubsample/pod5"
BLOW5_FILE="/data/slow5-testdata/hg2_prom_lsk114_subsubsample/reads.blow5"
MODEL="/data/install/dorado-0.1.1/models/dna_r10.4.1_e8.2_400bps_sup@v4.0.0/"

mkdir ${OUTPUT_DIR}

get_accuracy () {
    ${MINIMAP2} -cx map-ont ${REFERENC_GENOME} -t32 --secondary=no "${OUTPUT_DIR}/${1}" | awk '{print $10/$11}' | datamash mean 1 sstdev 1 q1 1 median 1 q3 1 > "${OUTPUT_DIR}/${2}"
}

for var in "$@"
do
    clean_fscache
    if [ "$var" = "slorado" ]; then
        /usr/bin/time -v ./slorado basecaller -o ${OUTPUT_DIR}/slorado_calls.fastq -t 40 -x cuda:0 -B500M -K1000 ${MODEL} ${BLOW5_FILE} 2>${OUTPUT_DIR}/slorado_log.txt
        get_accuracy "slorado_calls.fastq" "slorado_accuracy.txt"
        rm ${OUTPUT_DIR}/slorado_calls.fastq
    elif [ "$var" = "dorado_release" ]; then
        /usr/bin/time -v /data/install/dorado-0.1.1/bin/dorado basecaller ${MODEL} ${POD5_FILE} -x cuda:0 -r 1 --emit-fastq > ${OUTPUT_DIR}/dorado_release_calls.fastq 2>${OUTPUT_DIR}/dorado_release_log.txt
        get_accuracy "dorado_release_calls.fastq" "dorado_release_accuracy.txt"
        rm ${OUTPUT_DIR}/dorado_release_calls.fastq
    elif [ "$var" = "dorado" ]; then
        /usr/bin/time -v /data/hiruna/dorado/cmake-build/bin/dorado basecaller ${MODEL} ${POD5_FILE} -x cuda:0 -r 1 --emit-fastq > ${OUTPUT_DIR}/dorado_calls.fastq 2>${OUTPUT_DIR}/dorado_log.txt
        get_accuracy "dorado_calls.fastq" "dorado_accuracy.txt"
        rm ${OUTPUT_DIR}/dorado.fastq
    fi
done