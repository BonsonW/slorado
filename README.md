# Slorado

Slorado is a simplified version of [Dorado](https://github.com/nanoporetech/dorado) built on top of [S/BLOW5 format](https://www.nature.com/articles/s41587-021-01147-4) and reduced external dependencies (mainly [torchlib](https://pytorch.org/cppdocs/)) for relatively easier compilation.  slorado is developed using C/C++. Currently, slorado only supports Linux operating system (works on Windows through WSL). slorado can utilise NVIDIA or AMD GPU accelerators on x86_64 CPUs. Slorado also works on ARM64-based NVIDIA Jetson devices.

Slorado is mainly for our research and educational purposes. Thus, only a minimal set of basecalling features are supported and will not be up to date with Dorado. For a feature rich and up-to-date S/BLOW5-based basecaller for routine use, please see [buttery-eel](https://github.com/Psy-Fer/buttery-eel).


## Compilation from Github

Compilation instructions differs based on the system. Please pick one of the following that matches your system:

- [x84_64 CPU-only (basecalling will be horribly slow)](docs/cpu-build.md)
- [NVIDIA GPUs (CUDA) on x84_64 systems](docs/cuda-build.md)
- [AMD GPUs (ROCM) on x84_64 systems](docs/rocm-build.md)
- [ARM-based NVIDIA Jetson (CUDA) systems](docs/jetson-build.md)

## Running, options and testing

Download fast, high accuracy, and super accuracy simplex basecalling models (dna_r10.4.1_e8.2_400bps_fast@v4.2.0, dna_r10.4.1_e8.2_400bps_hac@v4.2.0 and dna_r10.4.1_e8.2_400bps_hac@v4.2.0). We have tested slorado only on these models.

```
scripts/download-models.sh
```

Now run on a test dataset
```
# for CPU
./slorado basecaller -x cpu models/dna_r10.4.1_e8.2_400bps_fast@v4.2.0 test/oneread_r10.blow5 -o reads.fastq
# for GPU
./slorado basecaller -x cuda:all models/dna_r10.4.1_e8.2_400bps_fast@v4.2.0 test/oneread_r10.blow5 -o reads.fastq
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



