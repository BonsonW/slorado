#!/bin/bash

NTHREADS="${NTHREADS:-32}"

die() {
    echo "$@" >&2
    exit 1
}

# map ref genome
map() {
	bam=$1
	genome=$2

	$SAMTOOLS fastq -@${NTHREADS} -TMM,ML "$bam" | $MINIMAP2 -t${NTHREADS} -x map-ont -a -y -Y --secondary=no "$genome" - | $SAMTOOLS sort -@${NTHREADS} - 
}

if [ $# -lt 2 ]; then
	die "Usage: $0 <reference genome>.fa <fastq/sam/bam file>"
fi

GENOME=$1 # path to reference genome
BAM=$2 # path to unmapped sam/bam output

$SAMTOOLS --version > /dev/null 2>&1 || die "samtools not found! Either put samtools under path or set SAMTOOLS variable, e.g.,export SAMTOOLS=/path/to/samtools"
$MINIMOD --version > /dev/null 2>&1 || die "minimod not found! Either put minimod under path or set MINIMOD variable, e.g.,export MINIMOD=/path/to/minimod"
$MINIMAP2 --version > /dev/null 2>&1 || die "minimap2 not found! Either put minimap2 under path or set MINIMAP2 variable, e.g.,export MINIMAP2=/path/to/minimap2"

BAM_MAP=mapped.sam # path to mapped bam output

map "$BAM" "$GENOME" > "$BAM_MAP" || die "mapping failed"

$SAMTOOLS index "$BAM_MAP" || die "indexing failed"

# get meth freq
$MINIMOD freq "$GENOME" "$BAM_MAP" -b || die "mod freq failed"

rm $BAM_MAP