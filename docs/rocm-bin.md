# Basecalling on AMD GPUs

With slorado, now you can do some nanopore basecalling on AMD GPUs. We have some compiled binaries which should work On Linux if you have GLIBC >= 2.28 (invoke `ldd --version` to check).  Note that this is in early development and we have tested them only on systems in [Note 1](note-1). If you enocunter a problem feel free to open an [issue]([issue](https://github.com/BonsonW/slorado/issues)).


## Getting Started

First, download and extract the slorado rocm linux binaries tar ball. Note that this is for testing only and the link will not be persistent!!!!

```
wget -O slorado-16-11-2024-e7c7e54.tar.gz "https://www.dropbox.com/s/odgcqtpp0d1hi4f/slorado-16-11-2024-e7c7e54.tar.gz?dl=1"
tar xf slorado-16-11-2024-e7c7e54.tar.gz
cd slorado-e7c7e54
bin/slorado --help
```

Download the test dataset with 20,000 reads and run slorado:
```
wget -O PGXXXX230339_reads_20k.blow5 https://slow5.bioinf.science/hg2_prom_5khz_subsubsample
./bin/slorado basecaller models/dna_r10.4.1_e8.2_400bps_hac@v4.2.0 PGXXXX230339_reads_20k.blow5  -o out.fastq -x cuda:all  -C 500 -v5
```
## Optional Testing

Test if the output maps and identity scores are good (required  minimap2, the human genome and datamash):
```
minimap2 -cx map-ont hg38noAlt.fa out.fastq --secondary=no -t16  | awk '{print $10/$11}' | datamash mean 1 median 1 count 1
```
It should print the mean identity score, median identity score and the number of alignments. The numbers are expected to be close to the following (would not be identical due to floating point deviations):
```
0.94113261325097        0.9771185       24768
```

### Note 1

Currently, we have tested these binaries on following systems:
1. O/S: SUSE Linux Enterprise Server 15, GPU: AMD MI250X
2. O/S: Ubuntu 20, GPUs: MI50/MI60, MI100 and MI210
