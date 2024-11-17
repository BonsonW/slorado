# Slorado

Slorado is a simplified version of [Dorado](https://github.com/nanoporetech/dorado) built on top of [S/BLOW5 format](https://www.nature.com/articles/s41587-021-01147-4) and reduced dependecies so that it can be (relatively) easily compiled.  Currently slorado only supports Linux on x86_64 architecture or aarm64 Jetson-based devices.

Slorado is mainly for research and educational purposes and performance is currently not the key goal. Slorado will only support a minimal set of features and may not be up to date with Dorado.
A feature rich, fast and up to date version of Dorado that supports S/BLOW5 (called slow5-dorado) can be found [here](https://github.com/hiruna72/slow5-dorado).

## Compilation and Running

### 1. Dependencies

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

A minimum g++ version of 5.4 is required for CPU or CUDA versions due to libtorch.
For CUDA version, you need the cuda toolkit installed.

### 2. Downloading Models

Download fast, high accuracy, and super accuracy simplex basecalling models (dna_r10.4.1_e8.2_400bps_fast@v4.2.0, dna_r10.4.1_e8.2_400bps_hac@v4.2.0 and dna_r10.4.1_e8.2_400bps_hac@v4.2.0). We have tested slorado only on these models.

```
scripts/download-models.sh
```

### 3. Make

<details><summary> <b>Option 1:</b> CUDA GPU version for NVIDIA GPUs.  </summary>

```
scripts/install-torch2.sh cuda
make cuda=1 -j
./slorado basecaller models/dna_r10.4.1_e8.2_400bps_fast@v4.2.0 test/oneread_r10.blow5
```
</details>

<details><summary> <b>Option 2:</b> ROCM GPU version for AMD GPUs.  </summary>

```
scripts/install-torch2.sh rocm
make rocm=1 -j
./slorado basecaller models/dna_r10.4.1_e8.2_400bps_fast@v4.2.0 test/oneread_r10.blow5
```
</details>

<details><summary> <b>Option 3:</b>  CPU-only version (horribly slow): </summary>

```
scripts/install-torch2.sh cpu
make -j
./slorado basecaller -x cpu models/dna_r10.4.1_e8.2_400bps_fast@v4.2.0 test/oneread_r10.blow5
```

</details>

### Building for ARM64 Jetson-based devices

<details><summary>Click to expand</summary>

1. Install and activate python venv.

    ```
    sudo apt install python3.8-venv
    python3 -m venv pytorch_venv
    source pytorch_venv/bin/activate
    ```

2. Update pip and install pytorch for your specific Nvidia Jetpack version. You can find this by running `sudo apt-cache show nvidia-jetpack | grep "Version"`, or browse https://developer.download.nvidia.com/compute/redist/jp/ to find a suitable version of pytorch. We tested on a Jetson Xavier board with Jetpack 5.0 installed and the commands used were:

    ```
    pip3 install --upgrade pip
    pip3 install --no-cache  https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.0.0a0+8aa34602.nv23.03-cp38-cp38-linux_aarch64.whl
    ```

3. Clone and build.

    ```
    git clone --recursive https://github.com/BonsonW/slorado.git
    cd slorado
    make -j cuda=1 cxx11_abi=1 LIBTORCH_DIR=/path/to/pytorch_venv/lib64/python3.8/site-packages/torch/
    ```
</details>

### Advanced Options

- Custom libtorch path:
    ```
    make cuda=1 LIBTORCH_DIR=/path/to/libtorch
    ```

- C++11 ABI:
    ```
    make cxx11_abi=1
    ```

- You can optionally enable zstd support for builtin slow5lib when building slorado by invoking make zstd=1. This requires zstd 1.3 development libraries installed on your system (libzstd1-dev package for apt, libzstd-devel for yum/dnf and zstd for homebrew).

### 4. Running, options and testing

```
./slorado basecaller models/dna_r10.4.1_e8.2_400bps_fast@v4.2.0 test/oneread_r10.blow5
```

Using a large batch size may take up a significant amount of RAM during run-time. Similarly, your GPU batch size will determine how much GPU memory is used.
Currently, slorado does not implement automatic batch size selection based on available memory. Thus, if you see an out of RAM error, reduce the batch size using -K or -B. If you see an out of GPU memory error, reduce the GPU batch size using -C option. All options supported by slorado are detailed below:


| Option:           | Decription:                                           | Default Value: |
|-------------------|-------------------------------------------------------|----------------|
| -t INT            | number of processing threads.                         | 8              |
| -K INT            | batch size (max number of reads loaded at once).      | 2000           |
| -C INT            | gpu batch size (max number of chunks loaded at once)  | 800            |
| -B FLOAT[K/M/G]   | max number of bytes loaded at once                    | 200.0M         |
| -o FILE           | output to file                                        | stdout         |
| -c INT            | chunk size                                            | 8000           |
| -p INT            | overlap                                               | 150            |
| -x DEVICE         | specify device (e.g., cpu, cuda:0, cuda:1,2: cuda:all)| cuda:0         |
| -h                | shows help message and exits                          | -              |
| --verbose INT     | verbosity level                                       | 4              |
| --version         | print version                                         |                |

A script to calculate Basecalling Accuracy is provided:
```
set environment variable MINIMAP2 if minimap2 is not in PATH.
scripts/calculate_basecalling_accuarcy.sh /genome/hg38noAlt.idx reads.fastq
```

## Acknowledgement

- A lot of code is coming from [Dorado](https://github.com/nanoporetech/dorado) which is licensed under [Oxford Nanopore Technologies PLC. Public License Version 1.0](thirdparty/dorado/LICENCE). Those files are located at [thirdparty/dorado](thirdparty/dorado).
- [tomlc99](https://github.com/cktan/tomlc99) library under [thirdparty/tomlc99](thirdparty/tomlc99), is licensed under [MIT license](thirdparty/tomlc99/LICENSE).
- Some code snippets have been taken from [Minimap2](https://github.com/lh3/minimap2).



