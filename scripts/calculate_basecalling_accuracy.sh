#!/bin/bash

MINIMAP2=minimap2 #path to minimpa2 executable if not added to $PATH
REFERENC_GENOME="test/Corona_NC_045512.2.fasta" #path to reference genome

FASTQ_FILE="slorado.fastq" #path to basecalled fastq file
${MINIMAP2} -cx map-ont ${REFERENC_GENOME} -t32 --secondary=no ${FASTQ_FILE} | awk '{print $10/$11}' | datamash mean 1 sstdev 1 q1 1 median 1 q3 1

FASTQ_FILE="dbinary.fastq" #path to basecalled fastq file
${MINIMAP2} -cx map-ont ${REFERENC_GENOME} -t32 --secondary=no ${FASTQ_FILE} | awk '{print $10/$11}' | datamash mean 1 sstdev 1 q1 1 median 1 q3 1
