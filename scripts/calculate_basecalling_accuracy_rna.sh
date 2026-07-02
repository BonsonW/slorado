#!/bin/bash

NTHREADS="${NTHREADS:-32}"

die() {
    echo "$@" >&2
    exit 1
}

if [ $# -ne 2 ]; then
    die "Usage: $0 <reference genome> <fastq file> "
fi

[ -z ${MINIMAP2} ] && MINIMAP2=minimap2
[ -z ${DATAMASH} ] && DATAMASH=datamash

${MINIMAP2} --version &> /dev/null || { echo -e $RED"minimap2 not found! Either put minimap2 under path or set MINIMAP2 variable, e.g.,export MINIMAP2=/path/to/minimap2"$NORMAL; exit 1;}
${DATAMASH} --version &> /dev/null || { echo -e $RED"datamash not found! Either put datamash under path or set DATAMASH variable, e.g.,export DATAMASH=/path/to/datamash, or if not installed: \`apt-get install datamash\`"$NORMAL; exit 1;}

REFERENC_GENOME=$1 #path to reference genome
FASTQ_FILE=$2 #path to basecalled fastq file

echo "identity scores:"
echo -e "mean\tstdev\tq1\tmedian\tq3\tn"
${MINIMAP2} -cx splice -uf -k14 ${REFERENC_GENOME} -t${NTHREADS} --secondary=no ${FASTQ_FILE} | awk '{print $10/$11}' | ${DATAMASH} mean 1 sstdev 1 q1 1 median 1 q3 1 count 1 || die "Error in identity calculation"
