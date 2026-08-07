#!/bin/bash
# compare_identity.sh — align FASTQ files with minimap2 and plot identity distributions
#
# Usage:
#   compare_identity.sh [OPTIONS] file1.fastq [file2.fastq ...]
#
# Options:
#   --ref FILE      Reference genome (default: $DEFAULT_REF)
#   --threads N     minimap2 threads per job (default: 32)
#   --labels L,...  Comma-separated legend labels (default: filenames)
#   --out FILE      Output plot path (default: identity_density.png/.pdf/.svg)
#   --xlim MIN MAX  X-axis limits (default: 0.8 1.0)
#   --scoredir DIR  Directory to save/reuse per-read score files (default: .)
#   --force         Re-align even if score file already exists
#   --plot-only     Skip alignment, just re-plot from existing score files
#
# Score files are saved as <scoredir>/<input-basename>.scores.txt and reused
# on subsequent runs if they already exist (skip alignment).
#
# Example — first run (aligns and saves scores):
#   compare_identity.sh --labels "slorado HAC v6,dorado HAC v6" \
#     --out comparison.pdf slorado_hac.fastq dorado_hac.fastq
#
# Re-plot with different labels (no re-alignment):
#   compare_identity.sh --labels "New label 1,New label 2" \
#     --out comparison_v2.svg slorado_hac.fastq dorado_hac.fastq

set -euo pipefail

DEFAULT_REF="/data/bonwon/tmp/genome/hg38noAlt.fa"
MINIMAP2="${MINIMAP2:-minimap2}"
THREADS=32
REF="$DEFAULT_REF"
OUT="identity_density.png"
XLIM="0.8 1.0"
SCOREDIR="."
FORCE=0
PLOT_ONLY=0
LABELS=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLOT_SCRIPT="$SCRIPT_DIR/plot_identity.py"

# ── argument parsing ─────────────────────────────────────────────────────────
FASTQ_FILES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref)       REF="$2";      shift 2 ;;
        --threads)   THREADS="$2";  shift 2 ;;
        --labels)    LABELS="$2";   shift 2 ;;
        --out)       OUT="$2";      shift 2 ;;
        --xlim)      XLIM="$2 $3";  shift 3 ;;
        --scoredir)  SCOREDIR="$2"; shift 2 ;;
        --force)     FORCE=1;       shift ;;
        --plot-only) PLOT_ONLY=1;   shift ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *)  FASTQ_FILES+=("$1"); shift ;;
    esac
done

if [[ ${#FASTQ_FILES[@]} -eq 0 ]]; then
    echo "Error: no FASTQ files specified. Use --help for usage." >&2
    exit 1
fi

mkdir -p "$SCOREDIR"

# ── build label array ────────────────────────────────────────────────────────
IFS=',' read -ra LABEL_ARR <<< "$LABELS"
FINAL_LABELS=()
for i in "${!FASTQ_FILES[@]}"; do
    if [[ -n "${LABEL_ARR[$i]:-}" ]]; then
        FINAL_LABELS+=("${LABEL_ARR[$i]}")
    else
        FINAL_LABELS+=("$(basename "${FASTQ_FILES[$i]}" .fastq)")
    fi
done

# ── align each file (skip if score file already exists) ──────────────────────
SCORE_FILES=()
PIDS=()
ALIGN_NEEDED=()

for i in "${!FASTQ_FILES[@]}"; do
    fq="${FASTQ_FILES[$i]}"
    label="${FINAL_LABELS[$i]}"
    base="$(basename "$fq" .fastq)"
    base="$(basename "$base" .fq)"
    score_file="$SCOREDIR/${base}.scores.txt"
    SCORE_FILES+=("$score_file")

    if [[ $PLOT_ONLY -eq 1 ]]; then
        ALIGN_NEEDED+=(0)
        if [[ ! -f "$score_file" ]]; then
            echo "ERROR: --plot-only set but score file not found: $score_file" >&2
            exit 1
        fi
        echo "[$(date '+%H:%M:%S')] Using existing scores: $score_file ($(wc -l < "$score_file") reads)"
    elif [[ -f "$score_file" && $FORCE -eq 0 ]]; then
        ALIGN_NEEDED+=(0)
        echo "[$(date '+%H:%M:%S')] Reusing existing scores: $score_file ($(wc -l < "$score_file") reads)"
    else
        ALIGN_NEEDED+=(1)
        echo "[$(date '+%H:%M:%S')] Starting alignment: $label"
        (
            "$MINIMAP2" -cx map-ont "$REF" -t"$THREADS" --secondary=no "$fq" \
                2>"${score_file%.txt}.log" \
                | awk '{if ($10+0>0 && $11+0>0) print $10/$11}' \
                > "$score_file"
        ) &
        PIDS+=($! $i)
    fi
done

# ── wait for running alignments ───────────────────────────────────────────────
if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo ""
    ALL_OK=1
    # PIDS contains alternating pid,index pairs
    for (( j=0; j<${#PIDS[@]}; j+=2 )); do
        pid="${PIDS[$j]}"
        i="${PIDS[$((j+1))]}"
        label="${FINAL_LABELS[$i]}"
        score_file="${SCORE_FILES[$i]}"
        if wait "$pid"; then
            echo "[$(date '+%H:%M:%S')] Done: $label ($(wc -l < "$score_file") reads) → $score_file"
        else
            echo "[$(date '+%H:%M:%S')] ERROR: alignment failed for $label (see ${score_file%.txt}.log)" >&2
            ALL_OK=0
        fi
    done

    if [[ $ALL_OK -eq 0 ]]; then
        echo "One or more alignments failed, aborting." >&2
        exit 1
    fi
fi

# ── plot ─────────────────────────────────────────────────────────────────────
echo ""
echo "[$(date '+%H:%M:%S')] Plotting → $OUT"

python3.12 "$PLOT_SCRIPT" \
    "${SCORE_FILES[@]}" \
    --labels "${FINAL_LABELS[@]}" \
    --out "$OUT" \
    --xlim $XLIM

echo "[$(date '+%H:%M:%S')] Saved: $OUT"
