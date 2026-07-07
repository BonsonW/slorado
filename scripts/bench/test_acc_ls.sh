#!/bin/bash
# Accuracy for all fastqs in each given ENAME's _out/, written to that ENAME's _logs/accuracy.txt.
# DNA map-ont vs a reference; identity = residue matches / alignment block length (PAF cols 10/11).
#
# Usage: ./test_acc_ls.sh [ENAME ...]
#   ENAME = experiment base name (dirs are ${OUTBASE}/${ENAME}_out and _logs).
# Env overrides:
#   OUTBASE (default /data/bonwon) - where the ENAME dirs live
#   REF     (default /genome/hg38noAlt.idx) - minimap2 target; if missing, accuracy is SKIPPED
#           (so a dry-pass on a system without the reference still validates basecalling).

REF="${REF:-/genome/hg38noAlt.idx}"
OUTBASE="${OUTBASE:-/data/bonwon}"
MINIMAP2="${MINIMAP2:-minimap2}"
DATAMASH="${DATAMASH:-datamash}"

warn() { echo "warn: $1" >&2; }
die()  { echo "Error: $1" >&2; exit 1; }

[ "$#" -gt 0 ] || die "usage: $0 <ENAME> [ENAME ...]   (dirs are \${OUTBASE}/<ENAME>_out)"
DIRS=("$@")

$MINIMAP2 --version >/dev/null 2>&1 || die "minimap2 not found (set MINIMAP2=...)"
$DATAMASH --version >/dev/null 2>&1 || die "datamash not found (set DATAMASH=...)"
if [ ! -e "$REF" ]; then warn "REF '$REF' not found -- skipping accuracy (set REF=...)"; exit 0; fi

run() {
    local fastq="$1" log="$2"
    test -e "$fastq" || die "$fastq does not exist"
    local out
    out=$($MINIMAP2 -cx map-ont "$REF" -t64 --secondary=no "$fastq" 2>/dev/null \
        | awk '{print $10/$11}' | $DATAMASH mean 1 sstdev 1 q1 1 median 1 q3 1 count 1)
    test -z "$out" && die "mapping produced no output for $fastq"
    printf '\n%s\n' "$fastq" >> "$log"
    printf 'sample\tmean\tsstdev\tq1\tmedian\tq3\tn\n' >> "$log"
    printf '%s\n' "$out" >> "$log"
}

for dir in "${DIRS[@]}"; do
    outdir="${OUTBASE}/${dir}_out"
    logdir="${OUTBASE}/${dir}_logs"
    log="${logdir}/accuracy.txt"
    test -d "$outdir" || { echo "skip: $outdir missing"; continue; }
    mkdir -p "$logdir"
    printf 'accuracies for %s\n' "$dir" > "$log"
    shopt -s nullglob
    for fastq in "${outdir}"/*.fastq; do
        echo "[acc] $fastq"
        run "$fastq" "$log"
    done
    shopt -u nullglob
    echo "[acc] wrote $log"
    cat "$log"
done

echo "accuracy done!"
