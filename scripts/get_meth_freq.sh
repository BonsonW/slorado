#!/bin/bash

die() {
    echo "$@" >&2
    exit 1
}

# map ref genome
map() {
	bam=$1
	genome=$2

	$SAMTOOLS fastq -@64 -TMM,ML "$bam" | minimap2 -t64 -x map-ont -a -y -Y --secondary=no "$genome" - | $SAMTOOLS sort -@64 - 
}

if [ $# -lt 3 ]; then
	die "Usage: $0 <reference genome> <fastq/sam/bam file> <output file>.mm.tsv|<output file>.mm.bedmethyl"
fi

GENOME=$1 # path to reference genome
BAM=$2 # path to unmapped sam/bam output
OUT=$3 # output file path

test -z "$SAMTOOLS" && SAMTOOLS=samtools # path to samtools
test -z "$MINIMOD" && MINIMOD=minimod # path to minimod

$SAMTOOLS --version > /dev/null 2>&1 || die "samtools not found! Either put samtools under path or set SAMTOOLS variable, e.g.,export SAMTOOLS=/path/to/samtools"
$MINIMOD --version > /dev/null 2>&1 || die "minimod not found! Either put minimod under path or set MINIMOD variable, e.g.,export MINIMOD=/path/to/minimod"

BAM_MAP=mapped.sam # path to mapped bam output

map "$BAM" "$GENOME" > "$BAM_MAP" || die "mapping failed"

$SAMTOOLS index "$BAM_MAP" || die "indexing failed"

# get meth freq
if [[ "$OUT" == *.mm.bedmethyl ]]; then
	$MINIMOD freq "$GENOME" "$BAM_MAP" -b > "$OUT" || die "mod freq failed"
elif [[ "$OUT" == *.mm.tsv ]]; then
	$MINIMOD freq "$GENOME" "$BAM_MAP" > "$OUT" || die "mod freq failed"
else
	die "unsupported output suffix: '$OUT' (use .mm.tsv or .mm.bedmethyl)"
fi

rm $BAM_MAP