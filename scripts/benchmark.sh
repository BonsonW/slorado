#!/bin/bash

MINIMAP2=../minimap2-2.24_x64-linux/minimap2 #path to minimpa2 executable if not added to $PATH
REFERENC_GENOME="/genome/hg38noAlt.idx" #path to reference genome

MODEL="dna_r10.4.1_e8.2_400bps_fast@v4.0.0/"
MODEL_PATH=/data/install/dorado-0.1.1/models/${MODEL}

OUTPUT_DIR=test_${MODEL}
DEVICE="cuda:0"
GPU_BATCHSIZE="1900"

POD5_FILE="/data/bonwon/hg2_prom_lsk114_subsample_pod5"
BLOW5_FILE="/data/slow5-testdata/hg2_prom_lsk114_subsample/reads.blow5"
# POD5_FILE="/data/bonwon/slorado/test/pod5_dir"
# BLOW5_FILE="/data/bonwon/slorado/test/oneread_r10.blow5"

[ -d ${OUTPUT_DIR} ] || mkdir ${OUTPUT_DIR}

get_accuracy () {
    ${MINIMAP2} -cx map-ont ${REFERENC_GENOME} -t32 --secondary=no "${OUTPUT_DIR}/${1}" | awk '{print $10/$11}' | datamash mean 1 sstdev 1 q1 1 median 1 q3 1 > "${OUTPUT_DIR}/${2}"
}

for var in "$@"
do
    clean_fscache
    if [ "$var" = "slorado" ]; then
        /usr/bin/time -v ./slorado basecaller -o ${OUTPUT_DIR}/slorado_calls.fastq -t 40 -x ${DEVICE} -B500M -C${GPU_BATCHSIZE} -K4000 -c10000 ${MODEL_PATH} ${BLOW5_FILE} 2>${OUTPUT_DIR}/slorado_log_${DEVICE}.txt
        # get_accuracy "slorado_calls.fastq" "slorado_accuracy.txt"
        rm ${OUTPUT_DIR}/slorado_calls.fastq
    elif [ "$var" = "dorado_release" ]; then
        /usr/bin/time -v /data/install/dorado-0.1.1/bin/dorado basecaller ${MODEL_PATH} ${POD5_FILE} -x ${DEVICE} -r 1 --emit-fastq > ${OUTPUT_DIR}/dorado_release_calls.fastq 2>${OUTPUT_DIR}/dorado_release_log.txt
        # get_accuracy "dorado_release_calls.fastq" "dorado_release_accuracy.txt"
        rm ${OUTPUT_DIR}/dorado_release_calls.fastq
    elif [ "$var" = "dorado" ]; then
        /usr/bin/time -v /data/bonwon/dorado_fixed/dorado/bin/dorado basecaller ${MODEL_PATH} ${POD5_FILE} -x ${DEVICE} -r 1 --emit-fastq > ${OUTPUT_DIR}/dorado_calls.fastq 2>${OUTPUT_DIR}/dorado_log.txt
        # get_accuracy "dorado_calls.fastq" "dorado_accuracy.txt"
        rm ${OUTPUT_DIR}/dorado_calls.fastq
    fi
done
