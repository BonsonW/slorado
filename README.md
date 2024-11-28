# Slorado

Slorado is a simplified version of [Dorado](https://github.com/nanoporetech/dorado) built on top of [S/BLOW5 format](https://www.nature.com/articles/s41587-021-01147-4). Slorado is an extremely lean basecaller with fewer external dependencies and is thus relatively easier to compile than Dorado.  Slorado is developed using C/C++ and depends on [torchlib](https://pytorch.org/cppdocs/). Currently, slorado only supports the Linux operating system (or Windows through WSL). slorado can **utilise NVIDIA or AMD GPU accelerators** on x86_64 CPUs. Slorado also works on ARM64-based NVIDIA Jetson devices.

Slorado is mainly for our research and educational purposes. Thus, only a minimal set of basecalling features are supported and may not be up-to-date with Dorado. For a feature-rich and up-to-date S/BLOW5-based basecaller for routine use, please see [buttery-eel](https://github.com/Psy-Fer/buttery-eel).

## Quick start

We provide compiled binaries for [NVIDIA (cuda)](https://docs.nvidia.com/cuda) and [AMD (rocm)](https://rocm.docs.amd.com/en/latest) GPU accelerators on x86_64 CPUs for Linux. You can download the latest relevant binary release that includes the most recent supported basecalling models from [releases](https://github.com/BonsonW/slorado/releases) as below:

```
VERSION=v0.2.0-beta
GPU=cuda   # GPU=rocm for AMD GPUs
wget "https://github.com/BonsonW/slorado/releases/download/$VERSION/slorado-$VERSION-x86_64-$GPU-linux-binaries.tar.gz"
tar xvf slorado-$VERSION-x86_64-$GPU-linux-binaries.tar.gz
cd slorado-$VERSION
./bin/slorado basecaller models/dna_r10.4.1_e8.2_400bps_hac@v4.2.0 reads.blow5  -o out.fastq -x cuda:all
```

Detailed instructions are available at:
- [NVIDIA GPUs (cuda) on x84_64 systems](docs/cuda-bin.md)
- [AMD GPUs (rocm) on x84_64 systems](docs/rocm-bin.md)

Binaries for the CPU-only version are not provided as basecalling on the CPU is impractically slow. Nevertheless, the CPU-only version is easier to build compared to the GPU version (see [below](#compilation-and-running)).

Refer to [troubleshoot](docs/troubleshoot.md) for help resolving common problems.

## Compilation and running

### Compilation

Compilation instructions differ based on the system. Please pick one of the following that matches your system:

- [x84_64 CPU-only (basecalling will be horribly slow)](docs/cpu-build.md)
- [NVIDIA GPUs (cuda) on x84_64 systems](docs/cuda-build.md)
- [AMD GPUs (rocm) on x84_64 systems](docs/rocm-build.md)
- [ARM-based NVIDIA Jetson (cuda) systems](docs/jetson-build.md)

### Running

We have tested this slorado version on basecalling models `dna_r10.4.1_e8.2_400bps_fast@v4.2.0`, `dna_r10.4.1_e8.2_400bps_hac@v4.2.0` and `dna_r10.4.1_e8.2_400bps_hac@v4.2.0`. You can download them using the provided script (the binary releases already include these):

```
scripts/download-models.sh
```

Now run on a test dataset:
```
# for CPU
./slorado basecaller -x cpu models/dna_r10.4.1_e8.2_400bps_fast@v4.2.0 test/oneread_r10.blow5 -o reads.fastq
# for GPU
./slorado basecaller -x cuda:all models/dna_r10.4.1_e8.2_400bps_fast@v4.2.0 test/oneread_r10.blow5 -o reads.fastq
```

Refer to [troubleshoot](docs/troubleshoot.md) for help resolving common problems. We are currently working on supporting the newer v5 basecalling models.

## Testing

After running on a test dataset, you can use minimap2 to align the reads to the reference and calculate the identity score statistics. If the identity score statistics are close enough to what we expect from these models then things are good.

A script to calculate basecalling accuracy is provided:
```
set environment variable MINIMAP2, if minimap2 is not in PATH.
scripts/calculate_basecalling_accuarcy.sh hg38noAlt.fa reads.fastq
```

## Options

All options supported by slorado basecaller are detailed below:


| Option:           | Decription:                                           | Default Value: |
|-------------------|-------------------------------------------------------|----------------|
| -t INT            | number of processing threads                          | 8              |
| -K INT            | batch size (max number of reads loaded at once)       | 2000           |
| -C INT            | gpu batch size (max number of chunks loaded at once)  | 500            |
| -B FLOAT[K/M/G]   | max number of bytes loaded at once                    | 500.0M         |
| -o FILE           | output to file                                        | stdout         |
| -c INT            | chunk size                                            | 10000           |
| -p INT            | overlap                                               | 150            |
| -x DEVICE         | specify device (e.g., cpu; cuda:0; cuda:1,2; cuda:all)| cuda:all (GPU version) or cpu (CPU version)         |
| -h                | shows help message and exits                          | -              |
| --verbose INT     | verbosity level                                       | 4              |
| --version         | print version                                         |                |

## Batchsizes

A large batch size (-K and -B) may take up significant RAM during run-time. Similarly, your GPU batch size (-C) will determine how much GPU memory is used. Slorado currently does not implement automatic batch size selection based on available memory. Thus, if you see an out-of-RAM error, reduce the batch size using -K or -B. If you see an out-of-GPU memory error, reduce the GPU batch size using the -C option.

## Acknowledgement

- A lot of code is coming from [Dorado](https://github.com/nanoporetech/dorado) which is licensed under [Oxford Nanopore Technologies PLC. Public License Version 1.0](thirdparty/dorado/LICENCE). Those files are located at [thirdparty/dorado](thirdparty/dorado).
- [tomlc99](https://github.com/cktan/tomlc99) library under [thirdparty/tomlc99](thirdparty/tomlc99), is licensed under [MIT license](thirdparty/tomlc99/LICENSE).
- Some code snippets have been taken from [Minimap2](https://github.com/lh3/minimap2).



