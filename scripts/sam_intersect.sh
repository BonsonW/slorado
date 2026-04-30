#!/bin/bash

file1=$1
file2=$2

# Extract read names from file1 (exclude header)
grep -v '^@' "$file1" | cut -f1 > filter_tmp.txt

# Filter file2
awk 'NR==FNR {keep[$1]; next} /^@/ || ($1 in keep)' \
    filter_tmp.txt "$file2"

rm filter_tmp.txt