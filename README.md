# slorado

A simplified version of Dorado.

## Compilation and running

```
sudo apt-get install zlib1g-dev   #install zlib development libraries
git clone --recursive https://github.com/BonsonW/slorado
cd slorado
```

The commands to install zlib development libraries on some popular distributions:

```
On Debian/Ubuntu : sudo apt-get install zlib1g-dev
On Fedora/CentOS : sudo dnf/yum install zlib-devel
On OS X : brew install zlib
```

### Make

CPU version:

```
scripts/install-torch12.sh
make -j
./slorado basecaller models/dna_r9.4.1_e8.1_fast@v3.4 test/example.blow5
```

CUDA GPU version:
```
scripts/install-torch12.sh
bash scripts/install-koi.sh
make cuda=1 -j
./slorado basecaller models/dna_r9.4.1_e8.1_fast@v3.4 test/example.blow5
```

advanced options:
```
make cuda=1 LIBTORCH_DIR=/path/to/torachlib CUDA_LIB=/path/to/cuda/library/
```


### CMake

```
# For CMake
sudo apt-get install zlib1g-dev   #install zlib development libraries
git clone --recursive https://github.com/BonsonW/slorado
cd slorado
bash scripts/install-torch10.sh
bash scripts/install-koi.sh
cd build
rm -r ./* && cmake ..
make -j && ./slorado basecaller -o one_read.fastq ../models/dna_r9.4.1_e8.1_fast@v3.4 ../test/one_read.slow5 -v 5
```

if different cuda location, before cmake:
export PATH=/usr/local/cuda-10.2/bin/:$PATH



## Calculate basecalling accuracy
```
set variables MINIMAP2, REFERENCE_GENOME, FASTQ_FILE in the script and run
scripts/calculate_basecalling_accuarcy.sh
```

## Acknowledgement
Code snippets have been taken from [Minimap2](https://github.com/lh3/minimap2).

## Walkthrough

The repository contains the following:
1. slow5lib as a git submodule
1. A [Makefile](Makefile)
2. C programme[source files](src/)
3. [shell script](test/test.sh) for testing
4. [GitHub actions workflow](.github/workflows/c-cpp.yml)
5. Miscellaneous files such as license, .gitignore, etc


## Links

SLOW5 specification: https://hasindu2008.github.io/slow5specs<br/>
slow5lib: https://hasindu2008.github.io/slow5lib/<br/>
slow5tools: https://hasindu2008.github.io/slow5tools<br/>
Pre-print: https://www.biorxiv.org/content/10.1101/2021.06.29.450255v1<br/>
Publication: https://www.nature.com/articles/s41587-021-01147-4<br/>




