# Basecalling on NVIDIA GPUs

We have some compiled binaries which should work On Linux if you have GLIBC >= 2.17 (invoke `ldd --version` to check).  We have tested them only on systems in [Note 1](#note-1). If you encounter a problem feel free to open an [issue]([issue](https://github.com/BonsonW/slorado/issues)).


## Getting Started

First, download and extract the slorado cuda Linux binaries tar ball. Note that this is for testing only and the link will not be persistent!!!!

```
VERSION=v0.2.0-beta
wget "https://github.com/BonsonW/slorado/releases/download/$VERSION/slorado-$VERSION-x86_64-cuda-linux-binaries.tar.gz" && tar xvf slorado-$VERSION-x86_64-cuda-linux-binaries.tar.gz && cd slorado-$VERSION
tar xvf slorado-$VERSION-x86_64-cuda-linux-binaries.tar.gz
cd slorado-$VERSION
bin/slorado --help
```

Download the test dataset with 20,000 reads and run slorado:
```
wget -O PGXXXX230339_reads_20k.blow5 https://slow5.bioinf.science/hg2_prom_5khz_subsubsample
./bin/slorado basecaller models/dna_r10.4.1_e8.2_400bps_hac@v4.2.0 PGXXXX230339_reads_20k.blow5  -o out.fastq -x cuda:all
```

Refer to [troubleshoot](troubleshoot.md) for help on resolving common problems.

## Optional Testing

Test if the output maps and identity scores are good (required  minimap2, the human genome and datamash):
```
minimap2 -cx map-ont hg38noAlt.fa out.fastq --secondary=no -t16  | awk '{print $10/$11}' | datamash mean 1 median 1 count 1
```
It should print the mean identity score, median identity score and the number of alignments. The numbers are expected to be close to the following (would not be identical due to floating point deviations):
```
0.94328430832131        0.978048       27027
```

### Note 1

Currently, we have tested these binaries on following systems:
1. O/S: Ubuntu 22, GPU: V100
