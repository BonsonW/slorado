#!/bin/bash

#SBATCH -p mi210_vck_u55c --cpus-per-task=16

die() {
    echo "$@" >&2
    exit 1
}

if [ $# -ne 2 ]; then
    die "Usage: $0 <device> <blow5>"
fi

make clean && make -j rocm=1

DEVICE=$1
BLOW5=$2
MODEL=fast
BATCHSIZE=1000
CHUNKSIZE=10000
THREADS=16

/usr/bin/time  --verbose ./slorado basecaller -t $THREADS -c $CHUNKSIZE -C $BATCHSIZE -B500M -x $DEVICE --verbose 5 models/dna_r10.4.1_e8.2_400bps_${MODEL}@v4.2.0 $BLOW5 > reads.fastq
