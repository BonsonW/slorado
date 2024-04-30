#!/bin/bash

die() {
    echo "$@" >&2
    exit 1
}

if [ $# -ne 2 ]; then
    die "Usage: $0 <reference genome> <fastq file>"
fi

MINIMAP2="/install/minimap2-2.26/minimap2"
${MINIMAP2} --version &> /dev/null || { echo -e $RED"minimap2 not found! Either put minimap2 under path or set MINIMAP2 variable, e.g.,export SIGTK=/path/to/minimap2"$NORMAL; exit 1;}

datamash --version &> /dev/null || { echo -e $RED"datamash not found! Please install datamash. e.g., apt-get install datamash"$NORMAL; exit 1;}

REFERENC_GENOME=$1 #path to reference genome
FASTQ_FILE=$2 #path to basecalled fastq file

echo "identity scores:"
echo -e "mean\tstdev\tq1\tmedian\tq3"
${MINIMAP2} -cx map-ont ${REFERENC_GENOME} -t8 --secondary=no ${FASTQ_FILE} | awk '{print $10/$11}' | datamash mean 1 sstdev 1 q1 1 median 1 q3 1
