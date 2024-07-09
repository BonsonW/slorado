#!/bin/bash

die() {
    echo "$@" >&2
    exit 1
}

# if [ $# -ne 2 ]; then
#     die "Usage: $0 <fastq file a> <fastq file b>"
# fi

fastq_a="reads_cpu.fastq"
fastq_b="reads.fastq"

tmp_a="a.tmp"
tmp_b="b.tmp"


if test -f $tmp_a; then
  rm $tmp_a
fi

if test -f $tmp_b; then
  rm $tmp_b
fi

awk 'NR % 4 == 0' $fastq_a | \
while read line; do
    awk -v qstring="$line" -f scripts/get_qstring_score.awk >> $tmp_a
done

awk 'NR % 4 == 0' $fastq_b | \
while read line; do
    awk -v qstring="$line" -f scripts/get_qstring_score.awk >> $tmp_b
done

# while 
#   read a &&
#   read b <&3
# do
  
# done < $tmp_a 3<$tmp_b

# rm $tmp_a
# rm $tmp_b