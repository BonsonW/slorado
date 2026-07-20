#!/bin/bash
# Unified basecaller benchmark + portable dry-pass.
#   ./bench.sh <slorado|dorado|slow5dorado> [--dry] [device ...]
#
# ENAME is auto: <gpu>_<prog>_v<version>_<dataset>.
# Per-run output/log files are named <model>_<ndev>dev.{fastq,txt}, all under the one ENAME dir.
# Each run also prints a wall-clock + reads/s line and appends it to the run log.
#
# --dry / -n : quick end-to-end sanity using the single read bundled next to this script.
#
# Platform aware: on Linux the GPU is CUDA (nvidia-smi) and slorado runs the fused/quant path; on
# macOS (Darwin) the GPU is Apple Metal (-x metal), inputs come from this repo, and slorado runs the
# pure-fp16 Metal path (no --flash/--quant, which are CUDA/ROCm-only). dorado always reads POD5,
# slorado/slow5dorado read BLOW5.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shared thirdparty-tool installer -> $MINIMAP2/$DATAMASH (built into test/tools) + install_tools()
source "${SCRIPT_DIR}/../install_tools.sh"

OS="$(uname -s)"

# ============================== CONFIG (edit paths here) ==============================
MODEL_TYPE="dna_r10.4.1_e8.2_400bps"
# space-separated list, env-overridable: MODELS="fast@v5.0.0 hac@v5.0.0 sup@v5.0.0"
read -ra MODELS <<< "${MODELS:-fast@v5.0.0}"
THREADS="${THREADS:-$(getconf _NPROCESSORS_ONLN)}"

if [ "$OS" = "Darwin" ]; then
    # macOS / Apple Silicon (Metal). Defaults resolve to this repo; override via env.
    SLORADO_BIN="${SLORADO_BIN:-${REPO_ROOT}/slorado}"
    DORADO_BIN="${DORADO_BIN:-$(find "${REPO_ROOT}/test/dorado_bin" -type f -name dorado -perm +111 2>/dev/null | head -1)}"
    SLOW5DORADO_BIN="${SLOW5DORADO_BIN:-slow5-dorado}"
    MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models}"
    BLOW5="${BLOW5:-${REPO_ROOT}/test/PGXXXX230339/reads_1k.blow5}"
    POD5="${POD5:-${REPO_ROOT}/test/PGXXXX230339/reads_1k.pod5}"
    OUTBASE="${OUTBASE:-${REPO_ROOT}/test/bench_out}"
    REF="${REF:-${REPO_ROOT}/test/slorado_test_ext_dat/genome/hg38noAlt.idx}"   # bundled minimap2 index -> accuracy runs
    DEVICES=( metal )
    # slorado's dylibs live under thirdparty/torch; make them findable at runtime.
    export DYLD_LIBRARY_PATH="${REPO_ROOT}/thirdparty/torch/libtorch/lib:${DYLD_LIBRARY_PATH:-}"
else
    # Linux / CUDA.
    SLORADO_BIN="${SLORADO_BIN:-/data/bonwon/slorado/slorado}"
    DORADO_BIN="${DORADO_BIN:-/data/bonwon/dorado-2.0.0-linux-x64/bin/dorado}"
    SLOW5DORADO_BIN="${SLOW5DORADO_BIN:-/data/bonwon/slow5-dorado/bin/slow5-dorado}"
    MODEL_DIR="${MODEL_DIR:-/data/bonwon/slorado/models}"
    BLOW5="${BLOW5:-/data/slow5-testdata/hg2_prom_lsk114_5khz_subsample/PGXXXX230339_reads_500k.blow5}"
    POD5="${POD5:-/data/slow5-testdata/hg2_prom_lsk114_5khz_subsample/PGXXXX230339_reads_500k.pod5}"
    OUTBASE="${OUTBASE:-/data/bonwon}"
    REF="${REF:-/genome/hg38noAlt.idx}"
    DEVICES=( cuda:0 )
fi

# bundled single read for --dry (kept next to this script)
DRY_BLOW5="${SCRIPT_DIR}/reads_1.blow5"
DRY_POD5="${SCRIPT_DIR}/reads_1.pod5"
# =====================================================================================

die() { echo "$1" >&2; exit 1; }

# Portable wall-clock: NOW returns epoch seconds with sub-second precision where available.
NOW() { python3 -c 'import time; print("%.3f" % time.time())' 2>/dev/null || date +%s; }

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
test -n "$BIN" && test -e "$BIN" || die "binary not found: $BIN"

if [ "$DRY" -eq 1 ]; then
    OUTBASE="$SCRIPT_DIR"; DATASET="dry"   # dry run keeps the requested MODELS so every model is exercised
    [ "$OS" = "Darwin" ] && DEVICES=( metal ) || DEVICES=( cuda:0 )
    case "$PROG" in dorado) INPUT="$DRY_POD5" ;; *) INPUT="$DRY_BLOW5" ;; esac
else
    case "$PROG" in dorado) INPUT="$POD5" ;; *) INPUT="$BLOW5" ;; esac
    DATASET=$(basename "$INPUT" | sed -E 's/\.(blow5|pod5)$//; s/_reads//')
    [ "${#DEV_ARGS[@]}" -gt 0 ] && DEVICES=("${DEV_ARGS[@]}")
fi
test -e "$INPUT"     || die "input not found: $INPUT"
test -d "$MODEL_DIR" || die "model dir not found: $MODEL_DIR"

if [ "$OS" = "Darwin" ]; then
    GPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null | tr 'A-Z ' 'a-z_' | tr -cd 'a-z0-9_')
else
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1 | awk '{print $2}' | cut -d- -f1 | tr 'A-Z' 'a-z')
fi
[ -z "$GPU" ] && GPU="gpu"
VER=$("$BIN" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z]+)?' | head -1)
[ -z "$VER" ] && VER="unknown"
PREFIX=""; [ "$DRY" -eq 1 ] && PREFIX="dry_"
ENAME="${PREFIX}${GPU}_${PROG}_v${VER}_${DATASET}"

LOGS="${OUTBASE}/${ENAME}_logs"; OUT="${OUTBASE}/${ENAME}_out"
mkdir -p "$LOGS" "$OUT"
echo "=== bench: $PROG  os=$OS  dry=$DRY  ENAME=$ENAME  input=$INPUT  devices=[${DEVICES[*]}]  models=[${MODELS[*]}] ==="

ndev() {
    case "$1" in
        metal|cpu|auto) echo 1 ;;
        *all*) nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' ;;
        cuda:*) echo "${1#cuda:}" | tr ',' '\n' | grep -c . ;;
        *) echo 1 ;;
    esac
}

# Per-model Metal GPU batch (-C). SUP's transformer GEMMs exceed MPS's max single-buffer size at
# the default 384 (6 GB alloc -> abort), so cap SUP. Override via FAST_BATCH/HAC_BATCH/SUP_BATCH.
metal_gpu_batch() {
    case "$1" in
        fast*) echo "${FAST_BATCH:-384}" ;;
        hac*)  echo "${HAC_BATCH:-384}" ;;
        sup*)  echo "${SUP_BATCH:-64}" ;;
        *)     echo "${GPU_BATCH:-384}" ;;
    esac
}

# Per-model Metal chunk size (-c). HAC's default ~10k chunk makes a ~235MB LSTM working buffer that
# stalls at batch 384 on 8GB unified memory; a 4800 chunk keeps it under the stall threshold so the
# efficient batch-384 kernel is actually usable (~1.8x faster, same accuracy). 0 = model default.
metal_chunk() {
    case "$1" in
        hac*) echo "${HAC_CHUNK:-4800}" ;;
        # SUP's CRF output [N,2T,4096] (state_len 5) stalls MPS at large N*T on 8GB; a 3072 chunk with
        # -C64 keeps it ~0.5GB (0.6 vs 0.17 reads/s baseline, same accuracy).
        sup*) echo "${SUP_CHUNK:-3072}" ;;
        *)    echo "0" ;;
    esac
}

run() {
    local device="$1" modeltag="$2"
    local model="${MODEL_TYPE}_${modeltag}"
    local n; n=$(ndev "$device")
    local tag="${modeltag}_${n}dev"
    local fq="${OUT}/${tag}.fastq" lg="${LOGS}/${tag}.txt"
    echo "[$PROG] ${model} on ${device} (${n} dev) -> ${tag}"
    local t0 t1 rc
    t0=$(NOW)
    case "$PROG" in
      slorado)
        if [ "$OS" = "Darwin" ]; then
            # Metal: pure fp16 path (--flash/--quant are CUDA/ROCm-only). Per-model -C/-c (see
            # metal_gpu_batch / metal_chunk) tuned for 8GB unified memory.
            local cbatch cchunk copt; cbatch=$(metal_gpu_batch "$modeltag"); cchunk=$(metal_chunk "$modeltag")
            copt=""; [ "$cchunk" -gt 0 ] && copt="-c $cchunk"
            "$BIN" basecaller --stream=yes -C "$cbatch" $copt -B4G -t "$THREADS" -x "$device" "${MODEL_DIR}/${model}" "$INPUT" > "$fq" 2> "$lg"
        else
            "$BIN" basecaller --stream=yes --flash=yes --quant=int8 -B4G -t "$THREADS" -x "$device" "${MODEL_DIR}/${model}" "$INPUT" > "$fq" 2> "$lg"
        fi ;;
      dorado)      "$BIN" basecaller -v "${MODEL_DIR}/${model}" "$INPUT" --emit-fastq --device "$device" > "$fq" 2> "$lg" ;;
      slow5dorado) "$BIN" basecaller "${MODEL_DIR}/${model}" "$INPUT" --emit-fastq --slow5-threads "$THREADS" --slow5-batchsize 1000 -x "$device" > "$fq" 2> "$lg" ;;
    esac
    rc=$?
    t1=$(NOW)
    local secs reads rps
    secs=$(python3 -c "print('%.2f' % ($t1 - $t0))" 2>/dev/null || echo "$((t1 - t0))")
    reads=$(grep -c '^@' "$fq" 2>/dev/null || echo 0)
    rps=$(python3 -c "print('%.1f' % ($reads / $secs)) if $secs > 0 else print('na')" 2>/dev/null || echo na)
    local summary="[$PROG] ${tag}: wall=${secs}s reads=${reads} reads/s=${rps} rc=${rc}"
    echo "$summary"
    echo "$summary" >> "$lg"
    return $rc
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
    install_tools minimap2 datamash   # ensure accuracy tools (built into test/tools if missing)
    OUTBASE="$OUTBASE" REF="$REF" bash "${SCRIPT_DIR}/test_acc_ls.sh" "$ENAME"; arc=$?
    if [ "$DRY" -eq 1 ]; then
        [ $arc -eq 0 ] && echo "=== DRY PASS OK ($PROG): basecall + accuracy pipeline works ===" \
                       || echo "=== DRY ($PROG): basecall OK, accuracy step returned $arc (ref missing?) ==="
    fi
else
    echo "[$PROG] had FAILURES - skipping accuracy"
fi
exit $FAIL
