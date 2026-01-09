# Basecalling on AMD GPUs

With slorado, now you can do some nanopore basecalling on AMD GPUs. We have some compiled binaries which should work on Linux if you have at least the minimum GLIBC listed [below](#tested-versions-and-requirements). Invoke `ldd --version` to check your GLIB version. You should also have the GPU driver installed.  Note that we have done testing on limited number of GPU systems listed [below](#tested-versions-and-requirements). If you encounter a problem feel free to open an [issue]([issue](https://github.com/BonsonW/slorado/issues)).


GLIBC >= (invoke `ldd --version` to check). You should also have the GPU driver installed. Currently, we have tested the binaries only on systems in [Note 1](#note-1). If you encounter a problem feel free to open an [issue]([issue](https://github.com/BonsonW/slorado/issues)).


## Getting Started

First, download and extract the slorado rocm Linux binaries tarball.

```
VERSION=v0.3.0-beta
wget "https://github.com/BonsonW/slorado/releases/download/$VERSION/slorado-$VERSION-x86_64-rocm-linux-binaries.tar.xz"
tar xvf slorado-$VERSION-x86_64-rocm-linux-binaries.tar.xz
cd slorado-$VERSION
bin/slorado --help
```

Download the test dataset with 20,000 reads and run slorado:
```
wget -O PGXXXX230339_reads_20k.blow5 https://slow5.bioinf.science/hg2_prom_5khz_subsubsample
./bin/slorado basecaller models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 PGXXXX230339_reads_20k.blow5  -o out.fastq -x cuda:all
```

Refer to [troubleshoot](docs/troubleshoot.md) for help on resolving common problems.

## Optional Testing

Test if the output maps and identity scores are good (required  minimap2, the human genome and datamash):
```
minimap2 -cx map-ont hg38noAlt.fa out.fastq --secondary=no -t16  | awk '{print $10/$11}' | datamash mean 1 median 1 count 1
```
It should print the mean identity score, median identity score and the number of alignments. The numbers are expected to be close to the following (would not be identical due to floating point deviations):
```
0.94113261325097        0.9771185       24768
```

### Tested versions and requirements

| Slorado binary version | minimum GLIBC | tested systems |
|---              | ---             | ---         |
| 0.4.0           | X           | xzzz        |
||||
| 0.3.0     |  2.28          |     |
| 0.2.0     |  2.28          | SUSE Linux 15 - Instinct MI250X;  Ubuntu 20 - Instinct MI50/MI60, MI100, MI210; Ubuntu 22 - Instinct MI210 |


slorado v0.2.0, we have tested these binaries on following systems:
1. O/S: SUSE Linux Enterprise Server 15; GPU: Instinct MI250X
2. O/S: Ubuntu 20; GPUs: Instinct MI50/MI60, Instinct MI100 and Instinct MI210
3. O/S:
