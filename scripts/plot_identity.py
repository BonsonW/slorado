#!/usr/bin/python3.12
# numpy/matplotlib pulled from available venv on this machine
import sys; sys.path.insert(0, '/data/hirsam/sigc/venv3/lib/python3.12/site-packages')
"""Plot read identity score histograms from aligned SAM files.

Identity is computed the same way paftools.js sam2paf does:
  identity = matches / block_length
  block_length = sum(M) + sum(I) + sum(D) in CIGAR
  matches     = sum(M) - mismatches
  mismatches  = NM - sum(I) - sum(D)
"""

import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

CIGAR_RE = re.compile(r'(\d+)([MIDNSHP=X])')


def cigar_counts(cigar):
    counts = {}
    for length, op in CIGAR_RE.findall(cigar):
        counts[op] = counts.get(op, 0) + int(length)
    return counts


def identity_from_record(fields):
    """Return identity score for one SAM record, or None to skip."""
    flag = int(fields[1])
    if flag & 4:        # unmapped
        return None
    if flag & 2048:     # supplementary — skip to avoid double-counting
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


def read_identities(path):
    identities = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('@'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) < 11:
                continue
            score = identity_from_record(fields)
            if score is not None:
                identities.append(score)
    return np.array(identities)


def print_stats(label, scores):
    if len(scores) == 0:
        print(f"{label}: no valid reads", file=sys.stderr)
        return
    print(
        f"{label}: n={len(scores)}  "
        f"mean={scores.mean():.4f}  "
        f"median={np.median(scores):.4f}  "
        f"q1={np.percentile(scores, 25):.4f}  "
        f"q3={np.percentile(scores, 75):.4f}  "
        f"stdev={scores.std():.4f}",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sam_files', nargs='+', help='Aligned SAM file(s)')
    parser.add_argument('--labels', nargs='+', help='Legend labels (one per file)')
    parser.add_argument('--bins', type=int, default=60, help='Histogram bins (default: 60)')
    parser.add_argument('--out', default='identity_histogram.png', help='Output image path')
    parser.add_argument('--xlim', nargs=2, type=float, default=None, metavar=('MIN', 'MAX'),
                        help='X-axis limits, e.g. --xlim 0.8 1.0')
    args = parser.parse_args()

    labels = args.labels or [p.replace('.sam', '') for p in args.sam_files]
    if len(labels) != len(args.sam_files):
        parser.error('--labels count must match number of SAM files')

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (path, label) in enumerate(zip(args.sam_files, labels)):
        scores = read_identities(path)
        print_stats(label, scores)
        if len(scores) == 0:
            continue
        ax.hist(scores, bins=args.bins, alpha=0.6, label=label,
                color=colors[i % len(colors)], density=True)

    ax.set_xlabel('Identity score (matches / alignment block length)')
    ax.set_ylabel('Density')
    ax.set_title('Read identity score distribution')
    if args.xlim:
        ax.set_xlim(args.xlim)
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
