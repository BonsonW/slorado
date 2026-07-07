#!/bin/bash
# Unified basecaller benchmark + portable dry-pass.
#   ./bench.sh <slorado|dorado|slow5dorado> [--dry] [device ...]
#
# ENAME is auto: <gpu>_<prog>_v<version>_<dataset>  (e.g. a100_slorado_v0.5.0-beta_PGXXXX230339_500k).
# Per-run output/log files are named just <model>_<ndev>dev.{fastq,txt}, all under the one ENAME dir.
#
# --dry / -n : quick end-to-end sanity on ANY system using the single read bundled next to this
#              script, one model (fast), cuda:0. Validates binary + flags + the accuracy pipeline.
#
# Real mode sweeps every entry in DEVICES (default cuda:0; override by passing devices as args) over
# MODELS, then runs accuracy via ./test_acc_ls.sh <ENAME> if all runs succeeded.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================== CONFIG (edit paths here) ==============================
# basecaller binaries
SLORADO_BIN="/data/bonwon/slorado/slorado"
DORADO_BIN="/data/bonwon/dorado-2.0.0-linux-x64/bin/dorado"
SLOW5DORADO_BIN="/data/bonwon/slow5-dorado/bin/slow5-dorado"

# models
MODEL_DIR="/data/bonwon/slorado/models"
MODEL_TYPE="dna_r10.4.1_e8.2_400bps"

# real-mode inputs (blow5 -> slorado/slow5dorado, pod5 -> dorado)
BLOW5="/data/slow5-testdata/hg2_prom_lsk114_5khz_subsample/PGXXXX230339_reads_500k.blow5"
POD5="/data/slow5-testdata/hg2_prom_lsk114_5khz_subsample/PGXXXX230339_reads_500k.pod5"

# where ENAME dirs are written, and the accuracy reference (passed to test_acc_ls.sh)
OUTBASE="/data/bonwon"
REF="/genome/hg38noAlt.idx"

# bundled single read for --dry (kept next to this script)
DRY_BLOW5="${SCRIPT_DIR}/reads_1.blow5"
DRY_POD5="${SCRIPT_DIR}/reads_1.pod5"

# what to run
THREADS=64
MODELS=( fast@v5.0.0 hac@v6.0.0 )   # add sup@v5.0.0 to also bench sup
DEVICES=( cuda:0 )                  # e.g. cuda:0  cuda:0,1,2,3  cuda:all
# =====================================================================================

die() { echo "$1" >&2; exit 1; }

PROG="${1:-}"
case "$PROG" in slorado|dorado|slow5dorado) ;; *)
    echo "usage: $0 <slorado|dorado|slow5dorado> [--dry] [device ...]" >&2; exit 1 ;;
esac
shift

DRY=0; DEV_ARGS=()
for a in "$@"; do
    case "$a" in --dry|-n) DRY=1 ;; *) DEV_ARGS+=("$a") ;; esac
done

case "$PROG" in
  slorado)     BIN="$SLORADO_BIN" ;;
  dorado)      BIN="$DORADO_BIN" ;;
  slow5dorado) BIN="$SLOW5DORADO_BIN" ;;
esac
test -e "$BIN" || die "binary not found: $BIN"

if [ "$DRY" -eq 1 ]; then
    MODELS=( fast@v5.0.0 ); DEVICES=( cuda:0 ); OUTBASE="$SCRIPT_DIR"; DATASET="dry"
    case "$PROG" in dorado) INPUT="$DRY_POD5" ;; *) INPUT="$DRY_BLOW5" ;; esac
else
    case "$PROG" in dorado) INPUT="$POD5" ;; *) INPUT="$BLOW5" ;; esac
    DATASET=$(basename "$INPUT" | sed -E 's/\.(blow5|pod5)$//; s/_reads//')
    [ "${#DEV_ARGS[@]}" -gt 0 ] && DEVICES=("${DEV_ARGS[@]}")
fi
test -e "$INPUT"     || die "input not found: $INPUT"
test -d "$MODEL_DIR" || die "model dir not found: $MODEL_DIR"

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1 | awk '{print $2}' | cut -d- -f1 | tr 'A-Z' 'a-z')
[ -z "$GPU" ] && GPU="gpu"
VER=$("$BIN" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z]+)?' | head -1)
[ -z "$VER" ] && VER="unknown"
PREFIX=""; [ "$DRY" -eq 1 ] && PREFIX="dry_"
ENAME="${PREFIX}${GPU}_${PROG}_v${VER}_${DATASET}"

LOGS="${OUTBASE}/${ENAME}_logs"; OUT="${OUTBASE}/${ENAME}_out"
mkdir -p "$LOGS" "$OUT"
echo "=== bench: $PROG  dry=$DRY  ENAME=$ENAME  input=$INPUT  devices=[${DEVICES[*]}]  models=[${MODELS[*]}] ==="

ndev() { case "$1" in *all*) nvidia-smi -L | wc -l ;; cuda:*) echo "${1#cuda:}" | tr ',' '\n' | grep -c . ;; *) echo 1 ;; esac; }

run() {
    local device="$1" modeltag="$2"
    local model="${MODEL_TYPE}_${modeltag}"
    local n; n=$(ndev "$device")
    local tag="${modeltag}_${n}dev"
    local fq="${OUT}/${tag}.fastq" lg="${LOGS}/${tag}.txt"
    echo "[$PROG] ${model} on ${device} (${n} dev) -> ${tag}"
    case "$PROG" in
      slorado)     /usr/bin/time --verbose "$BIN" basecaller --stream=yes --flash=yes --quant=int8 -B4G -t "$THREADS" -x "$device" "${MODEL_DIR}/${model}" "$INPUT" > "$fq" 2> "$lg" ;;
      dorado)      /usr/bin/time --verbose "$BIN" basecaller -v "${MODEL_DIR}/${model}" "$INPUT" --emit-fastq --device "$device" > "$fq" 2> "$lg" ;;
      slow5dorado) /usr/bin/time --verbose "$BIN" basecaller "${MODEL_DIR}/${model}" "$INPUT" --emit-fastq --slow5-threads "$THREADS" --slow5-batchsize 1000 -x "$device" > "$fq" 2> "$lg" ;;
    esac
}

FAIL=0
for DEVICE in "${DEVICES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    run "$DEVICE" "$MODEL"; rc=$?
    [ $rc -ne 0 ] && { echo "[$PROG] FAILED ${MODEL} ${DEVICE} (rc=$rc) -- see ${LOGS}/${MODEL}_*dev.txt"; FAIL=1; }
  done
done

if [ $FAIL -eq 0 ]; then
    echo "[$PROG] all runs ok -> accuracy on ${ENAME}"
    OUTBASE="$OUTBASE" REF="$REF" bash "${SCRIPT_DIR}/test_acc_ls.sh" "$ENAME"; arc=$?
    if [ "$DRY" -eq 1 ]; then
        [ $arc -eq 0 ] && echo "=== DRY PASS OK ($PROG): basecall + accuracy pipeline works ===" \
                       || echo "=== DRY ($PROG): basecall OK, accuracy step returned $arc (ref missing?) ==="
    fi
else
    echo "[$PROG] had FAILURES - skipping accuracy"
fi
exit $FAIL
