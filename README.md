# Slorado

Slorado is a simplified version of [Dorado](https://github.com/nanoporetech/dorado) built on top of [S/BLOW5 format](https://www.nature.com/articles/s41587-021-01147-4) and reduced dependecies so that it can be (relatively) easily compiled. A minimum g++ version of 5.4 is required. Currently slorado only supports Linux on x86_64 architecture.

Slorado is mainly for research and educational purposes and performance is currently not the key goal. Slorado will only support a minimal set of features and may not be up to date with Dorado.
A feature rich, fast and up to date version of Dorado that supports S/BLOW5 (called slow5-dorado) can be found [here](https://github.com/hiruna72/slow5-dorado).

## Compilation and running

### Dependencies

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

### Downloading Models

Download fast, high accuracy, and super accuracy simplex basecalling models (dna_r10.4.1_e8.2_400bps_fast@v4.0.0, dna_r10.4.1_e8.2_400bps_hac@v4.0.0 and dna_r10.4.1_e8.2_400bps_sup@v4.0.0). We have tested slorado only on these models.

```
scripts/download-models.sh
```

### Make

### Option 1:

CUDA GPU version that uses ONT's closed-source koi library binaries (minimum requirement: CUDA 11.3). This is the fastest:
```
scripts/install-torch12.sh
make cuda=1 koi=1 -j
./slorado basecaller models/dna_r10.4.1_e8.2_400bps_fast@v4.0.0 test/oneread_r10.blow5
```

If you do not have CUDA 11.3 or higher installed system wide, you can install CUDA 11.3 using following commands:
```
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
chmod +x cuda_11.3.0_465.19.01_linux.run
./cuda_11.3.0_465.19.01_linux.run --toolkit --toolkitpath=/local/path/cuda/
```
Then compile slorado by specifying the custom CUDA location to CUDA_ROOT variable as:
```
make cuda=1 koi=1 -j CUDA_ROOT=/local/path/cuda/
```

### Option 2:

CUDA GPU version without close-source koi library (uses CPU decoder, thus considerably slower). CUDA 10.2 or higher is adequate for this:
```
scripts/install-torch12.sh
make cuda=1 -j
./slorado basecaller models/dna_r10.4.1_e8.2_400bps_fast@v4.0.0 test/oneread_r10.blow5
```

### Option 3:

CPU-only version (horribly slow):

```
scripts/install-torch12.sh
make -j
./slorado basecaller -x cpu models/dna_r10.4.1_e8.2_400bps_fast@v4.0.0 test/oneread_r10.blow5
```

### advanced options

- Custom libtorch path:
```
make cuda=1 LIBTORCH_DIR=/path/to/torchlib
```

- C++11 ABI:
```
make cxx11_abi=1
```

- You can optionally enable zstd support for builtin slow5lib when building slorado by invoking make zstd=1. This requires zstd 1.3 development libraries installed on your system (libzstd1-dev package for apt, libzstd-devel for yum/dnf and zstd for homebrew).


## Calculate basecalling accuracy
```
set environment variable MINIMAP2 if minimap2 is not in PATH.
scripts/calculate_basecalling_accuarcy.sh /genome/hg38noAlt.idx reads.fastq
```

## Acknowledgement

- A lot of code is coming from [Dorado](https://github.com/nanoporetech/dorado) which is licensed under [Oxford Nanopore Technologies PLC. Public License Version 1.0](thirdparty/dorado/LICENCE). Those files are located at [thirdparty/dorado](thirdparty/dorado).

- [tomlc99](https://github.com/cktan/tomlc99) library under [thirdparty/tomlc99](thirdparty/tomlc99), is licensed under [MIT license](thirdparty/tomlc99/LICENSE).

- [koi_lib] library under [thirdparty/koi_lib](thirdparty/koi_lib) are proprietary binaries from Oxford Nanopore Technologies.

- Some code snippets have been taken from [Minimap2](https://github.com/lh3/minimap2).



