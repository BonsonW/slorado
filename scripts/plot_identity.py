#!/usr/bin/python3.12
# numpy/matplotlib/scipy pulled from available venv on this machine
import sys; sys.path.insert(0, '/data/hirsam/sigc/venv3/lib/python3.12/site-packages')
"""Plot read identity score distribution (smooth KDE) from SAM or FASTQ files.

For SAM input, identity is computed the same way paftools.js sam2paf does:
  identity = matches / block_length
  block_length = sum(M) + sum(I) + sum(D)
  matches = sum(M) - (NM - sum(I) - sum(D))

For FASTQ input, minimap2 is run internally and identity comes from PAF
fields 10/11 (residue matches / alignment block length).

A pre-computed scores file (one float per line, or two tab-separated
columns <read_id> <score>) can also be provided directly.
"""

import os
import re
import sys
import argparse
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

MINIMAP2 = os.environ.get('MINIMAP2', 'minimap2')
DEFAULT_REF = '/data/bonwon/tmp/genome/hg38noAlt.fa'

CIGAR_RE = re.compile(r'(\d+)([MIDNSHP=X])')


def cigar_counts(cigar):
    counts = {}
    for length, op in CIGAR_RE.findall(cigar):
        counts[op] = counts.get(op, 0) + int(length)
    return counts


def identity_from_sam_record(fields):
    """Return identity score for one SAM record, or None to skip."""
    flag = int(fields[1])
    if flag & 4:
        return None
    if flag & 2048:
        return None
    seq = fields[9]
    if seq == '*' or len(seq) < 200:
        return None
    cigar = fields[5]
    if cigar == '*':
        return None

    nm = None
    for tag in fields[11:]:
        if tag.startswith('NM:i:'):
            nm = int(tag[5:])
            break
    if nm is None:
        return None

    c = cigar_counts(cigar)
    m = c.get('M', 0)
    ins = c.get('I', 0)
    d = c.get('D', 0)
    block_length = m + ins + d
    if block_length == 0:
        return None

    mismatches = nm - ins - d
    matches = m - mismatches
    return matches / block_length


def read_identities_sam(path):
    identities = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('@'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 11:
                continue
            score = identity_from_sam_record(fields)
            if score is not None:
                identities.append(score)
    return np.array(identities)


def read_identities_fastq(path, ref, threads):
    """Stream minimap2 PAF output and return per-read identity array."""
    cmd = [MINIMAP2, '-cx', 'map-ont', ref, f'-t{threads}', '--secondary=no', path]
    print(f'  running: {" ".join(cmd)}', file=sys.stderr)
    scores = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) as proc:
        for line in proc.stdout:
            fields = line.split('\t')
            if len(fields) < 12:
                continue
            try:
                matches = int(fields[9])
                block_len = int(fields[10])
                if block_len > 0:
                    scores.append(matches / block_len)
            except (ValueError, IndexError):
                continue
    return np.array(scores)


def read_identities_scores(path):
    """Read a pre-computed scores file: one float per line or <id>\\t<score>."""
    scores = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            try:
                scores.append(float(parts[-1]))
            except ValueError:
                continue
    return np.array(scores)


def read_identities(path, ref, threads):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.fastq', '.fq'):
        return read_identities_fastq(path, ref, threads)
    elif ext in ('.sam', '.bam'):
        return read_identities_sam(path)
    else:
        return read_identities_scores(path)


def print_stats(label, scores):
    if len(scores) == 0:
        print(f'{label}: no valid reads', file=sys.stderr)
        return
    print(
        f'{label}: n={len(scores):,}  '
        f'mean={scores.mean():.6f}  '
        f'median={np.median(scores):.6f}  '
        f'q1={np.percentile(scores, 25):.6f}  '
        f'q3={np.percentile(scores, 75):.6f}  '
        f'stdev={scores.std():.6f}',
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='+', help='SAM, FASTQ, or pre-computed scores file(s)')
    parser.add_argument('--labels', nargs='+', help='Legend labels (one per file)')
    parser.add_argument('--ref', default=DEFAULT_REF, help='Reference genome (needed for FASTQ input)')
    parser.add_argument('--threads', type=int, default=32, help='minimap2 threads (default: 32)')
    parser.add_argument('--out', default='identity_density.png', help='Output image path')
    parser.add_argument('--xlim', nargs=2, type=float, default=[0.8, 1.0], metavar=('MIN', 'MAX'),
                        help='X-axis limits (default: 0.8 1.0)')
    parser.add_argument('--bw', type=float, default=None, help='KDE bandwidth (default: Scott\'s rule)')
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(p) for p in args.inputs]
    if len(labels) != len(args.inputs):
        parser.error('--labels count must match number of input files')

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (path, label) in enumerate(zip(args.inputs, labels)):
        color = colors[i % len(colors)]
        print(f'Reading {path} ...', file=sys.stderr)
        scores = read_identities(path, args.ref, args.threads)
        print_stats(label, scores)
        if len(scores) == 0:
            continue

        # Clip to xlim range for KDE (avoids long tails distorting bandwidth)
        lo, hi = args.xlim
        clipped = scores[(scores >= lo) & (scores <= hi)]

        bw = args.bw if args.bw else 'scott'
        kde = gaussian_kde(clipped, bw_method=bw)
        x = np.linspace(lo, hi, 2000)
        y = kde(x)

        ax.plot(x, y, color=color, linewidth=2, label=label)
        ax.fill_between(x, y, alpha=0.15, color=color)

        mean_val = scores.mean()
        median_val = float(np.median(scores))

        ax.axvline(mean_val, color=color, linestyle='--', linewidth=1.5,
                   label=f'{label} mean={mean_val:.4f}')
        ax.axvline(median_val, color=color, linestyle=':', linewidth=1.5,
                   label=f'{label} median={median_val:.4f}')

    ax.set_xlabel('Identity score (matches / alignment block length)')
    ax.set_ylabel('Density')
    ax.set_title('Read identity score distribution')
    ax.set_xlim(args.xlim)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'Saved to {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
