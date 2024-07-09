#!/bin/bash

die() {
    echo "$@" >&2
    exit 1
}

if [ $# -ne 2 ]; then
    die "Usage: $0 <fastq file a> <fastq file b>"
fi

fastq_a=$1
fastq_b=$2

tmp_a="a.tmp"
tmp_b="b.tmp"


if test -f $tmp_a; then
  rm $tmp_a
fi

if test -f $tmp_b; then
  rm $tmp_b
fi

echo "calculating avg qscores for \"${fastq_a}\"..."
awk 'NR % 4 == 0' $fastq_a | \
while read line; do
    awk -v qstring="$line" -f scripts/get_qstring_score.awk >> $tmp_a
done

echo "calculating avg qscores for \"${fastq_b}\"..."
awk 'NR % 4 == 0' $fastq_b | \
while read line; do
    awk -v qstring="$line" -f scripts/get_qstring_score.awk >> $tmp_b
done

echo "comparing avg qscores..."
sum=0
n=0

shopt -s lastpipe
while 
  read score_a &&
  read score_b <&3
do
  diff=$(echo "$score_a - $score_b" | bc)
  diff=$(echo ${diff#-})
  sum=$(echo "$sum + $diff" | bc) 
  n=$(($n+1))
done < $tmp_a 3<$tmp_b

avg=$(echo "scale=5; $sum / $n" | bc)
echo "average diff in qscore for ${n} reads: ${avg}"

rm $tmp_a
rm $tmp_b

echo "all done!"