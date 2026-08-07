#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "usage : $0 genome.fa/gencode.v40.transcripts.fa reads.fastq"
        exit
fi

die() {
    local msg="${1}"
    echo "Error: ${msg}" >&2
    exit 1
}

test -e $1 || die "$1 does not exist"
FASTQ=$2
test -e $FASTQ || die "$FASTQ does not exist"

minimap2 --version > /dev/null || die "minimap2 not found in PATH"
datamash --version > /dev/null || die "datamash not found in PATH"

OUTPUT=$(minimap2 -cx splice -uf -k14  $1 -t8 --secondary=no ${FASTQ} | awk '{print $10/$11}' | datamash mean 1 sstdev 1 q1 1 median 1 q3 1 count 1)
echo -en "sample\tmean\tsstdev\tq1\tmedian\tq3\tn\n"
echo -en "${FASTQ}\t$OUTPUT\n"