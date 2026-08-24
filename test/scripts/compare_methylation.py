#!/usr/bin/env python3
# Pair two bedMethyl files by site for ONE modification code and emit the comparison TSV
# consumed by scripts/plot_methylation.R (columns: key depth_1 frequency_1 depth_2 frequency_2).
#
# Generalized from f5c/nanopolish compare_methylation.py: mod-code aware (bedMethyl col 4) and
# CpG-strand merge is optional. Use --merge-cpg for symmetric CpG marks (5mC 'm', 5hmC 'h');
# leave it off for strand-specific / all-context marks (6mA 'a', RNA m6A 'a').
#
# Usage:
#   compare_methylation.py A.bedmethyl B.bedmethyl m  --merge-cpg > cmp.tsv    # 5mC
#   compare_methylation.py A.bedmethyl B.bedmethyl h  --merge-cpg > cmp.tsv    # 5hmC
#   compare_methylation.py A.bedmethyl B.bedmethyl a             > cmp.tsv    # 6mA / m6A
#
# bedMethyl cols: chrom0 start1 end2 code3 score4 strand5 ... coverage9 percent10
import sys

def load(fn, code, merge_cpg):
    out = {}
    for line in open(fn):
        f = line.split()
        if len(f) < 11 or f[3] != code:
            continue
        contig, start, strand = f[0], int(f[1]), f[5]
        cov = float(f[9]); meth = (float(f[10]) / 100.0) * cov
        if merge_cpg:
            pos = start if strand == "+" else start - 1
        else:
            pos = start if strand == "+" else -start - 1   # keep strands distinct
        key = f"{contig}:{pos}"
        c, m = out.get(key, (0.0, 0.0))
        out[key] = (c + cov, m + meth)
    return out

def main():
    args = [a for a in sys.argv[1:] if a != "--merge-cpg"]
    merge = "--merge-cpg" in sys.argv
    if len(args) != 3:
        sys.exit("usage: compare_methylation.py A.bedmethyl B.bedmethyl MODCODE [--merge-cpg] > cmp.tsv")
    A, B, code = args
    a, b = load(A, code, merge), load(B, code, merge)
    n = 0
    print("key\tdepth_1\tfrequency_1\tdepth_2\tfrequency_2")
    for k in a:
        if k not in b:
            continue
        ca, ma = a[k]; cb, mb = b[k]
        if ca == 0 or cb == 0:
            continue
        print("%s\t%d\t%.4f\t%d\t%.4f" % (k, ca, ma / ca, cb, mb / cb))
        n += 1
    sys.stderr.write("mod=%s set1=%d set2=%d shared_output=%d\n" % (code, len(a), len(b), n))

if __name__ == "__main__":
    main()
