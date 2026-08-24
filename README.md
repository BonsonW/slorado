# Slorado

Slorado is a simplified version of [Dorado](https://github.com/nanoporetech/dorado) built on top of [S/BLOW5 format](https://www.nature.com/articles/s41587-021-01147-4). Slorado is an extremely lean basecaller with fewer external dependencies and is thus relatively easier to compile than Dorado.  Slorado is developed using C/C++ and depends on [torchlib](https://pytorch.org/cppdocs/). Currently, slorado only supports the Linux operating system (or Windows through WSL). slorado can **utilise NVIDIA or AMD GPU accelerators** on x86_64 CPUs. Slorado also works on ARM64-based NVIDIA Jetson devices.

Slorado is mainly for our research and educational purposes. Thus, only a minimal set of basecalling features are supported and may not be up-to-date with Dorado. For a feature-rich and up-to-date S/BLOW5-based basecaller for routine use on NVIDIA GPUs, please see [buttery-eel](https://github.com/Psy-Fer/buttery-eel) or [slow5-dorado](https://github.com/hiruna72/slow5-dorado/releases).

[![GitHub Downloads](https://img.shields.io/github/downloads/BonsonW/slorado/total?logo=GitHub)](https://github.com/BonsonW/slorado/releases) [![slorado](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/warp9seq/git-count/refs/heads/main/sum/slorado.json)](https://github.com/BonsonW/slorado)

## Quick start

We provide compiled binaries for [NVIDIA (cuda)](https://docs.nvidia.com/cuda) and [AMD (rocm)](https://rocm.docs.amd.com/en/latest) GPU accelerators on x86_64 CPUs for Linux. You can download the latest relevant binary release that includes the most recent supported basecalling models from [releases](https://github.com/BonsonW/slorado/releases) as below:

```
VERSION=v0.6.0
GPU=cuda   # GPU=rocm for AMD GPUs
wget "https://cdn.bioinf.science/slorado/slorado-$VERSION-x86_64-$GPU-linux-binaries.tar.xz"
tar xvf slorado-$VERSION-x86_64-$GPU-linux-binaries.tar.xz
cd slorado-$VERSION
./bin/slorado basecaller models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 reads.blow5  -o out.fastq -x cuda:all
```

This may take up to several minutes to download and extract.

Detailed instructions are available at:
- [NVIDIA GPUs (cuda) on x86_64 systems](docs/cuda-bin.md)
- [AMD GPUs (rocm) on x86_64 systems](docs/rocm-bin.md)

Basecalling on Australia's [Pawsey](https://pawsey.org.au/) supercomputer: [Pawsey Guide](docs/pawsey.md)

Binaries for the CPU-only version are not provided as basecalling on the CPU is impractically slow. Nevertheless, the CPU-only version is easier to build compared to the GPU version (see [below](#compilation-and-running)).

Refer to [troubleshoot](docs/troubleshoot.md) for help resolving common problems.


## Compilation

Compilation instructions differ based on the system. Please pick one of the following that matches your system:

- [x86_64 CPU-only (basecalling will be horribly slow)](docs/cpu-build.md)
- [NVIDIA GPUs (cuda) on x86_64 systems](docs/cuda-build.md)
- [AMD GPUs (rocm) on x86_64 systems](docs/rocm-build.md)
- [ARM-based NVIDIA Jetson (cuda) systems](docs/jetson-build.md)

Note: building from source will first require downloading and extracting Libtorch, which may take up to an hour depending on your network speed. Compilation should only take up to several minutes.

## Running

We have tested slorado on a limited number of basecalling models listed [below](#tested-models). You can download them using the provided script (the binary releases already include these):

```
scripts/download-models.sh
```

Now run on a test dataset:
```
# for CPU
./slorado basecaller -x cpu models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0 test/PGXXXX230339/reads_1.blow5 -o reads.fastq
# for GPU
./slorado basecaller -x cuda:all models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0 test/PGXXXX230339/reads_1k.blow5 -o reads.fastq
```

Refer to [troubleshoot](docs/troubleshoot.md) for help resolving common problems.

### Basecalling POD5 files

Slorado reads signal data in S/BLOW5 format. POD5 files from ONT sequencers can be converted to BLOW5 using [blue-crab](https://github.com/Psy-Fer/blue-crab):

```
# install blue-crab
pip install blue-crab

# convert a pod5 file (or a directory of pod5 files) to blow5
blue-crab p2s reads.pod5 -o reads.blow5
blue-crab p2s pod5_dir/ -o reads.blow5

# then basecall as usual
./slorado basecaller models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 reads.blow5 -o reads.fastq -x cuda:all
```

For convenience, we also provide a wrapper script [pod5-slorado](scripts/pod5-slorado) that converts a POD5 file to a temporary BLOW5 file and then invokes slorado on it. It takes the same arguments as `slorado basecaller`, but with a POD5 file as input:

```
# set environment variable SLORADO, if slorado is not in PATH (export SLORADO=/path/to/slorado).
# set environment variable BLUECRAB, if blue-crab is not in PATH (export BLUECRAB=/path/to/blue-crab).
scripts/pod5-slorado basecaller models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0 reads.pod5 -o reads.fastq
```

### Docker

Pre-built images are available on [Docker Hub](https://hub.docker.com/r/hasindu2008/slorado). Refer to [here](docs/docker.md) for detailed instructions.

## Testing

After running on a test dataset, you can use minimap2 to align the reads to the reference and calculate the identity score statistics. If the identity score statistics are close enough to what we expect from these models, then things are good.

A script to calculate basecalling accuracy is provided:
```
# Download the reference
wget -O hg38noAlt.fa.gz seq.bioinf.science/hg38noAlt && gunzip hg38noAlt.fa.gz

# set environment variable MINIMAP2, if minimap2 is not in PATH (export MINIMAP2=/path/to/minimap2).
# set environment variable DATAMASH, if datamash is not in PATH (export DATAMASH=/path/to/datamash).
scripts/calculate_basecalling_accuracy.sh hg38noAlt.fa reads.fastq

# expected median identity scores for test/PGXXXX230339/reads_1k.blow5:
# FAST v5.0.0: 0.940374
# HAC v5.0.0:  0.977594
# SUP v5.0.0:  0.988561
```

The repo also includes a minimal test that runs FAST v5.0.0 in [test.sh](test/test.sh), which will automatically install minimap2 and run on a single small dataset and reference included in the repo.
```
# run on a single read:
./test/test.sh
# accuracy: 0.967836

# run on 1k reads mapped to chr22:
./test/test.sh chr22
# accuracy: 0.941745
```

For a more exhaustive test of slorado's features (on GPU setups), we have provided an [extensive test script](test/extensive.sh). This will automatically download the requisite test data and tools to test DNA/RNA basecalling, methylation detection, and flash attention support on your device. We highly recommend running this to ensure basecalling works on your machine. Excluding the automated binary release test mode, this script is meant to work on both ARM and x86 architectures.

Here is an example of how to run it:
```
# optionally customise parameters before running
export FAST_BATCH=512   # FAST model GPU batch size
export HAC_BATCH=256    # HAC model GPU batch size
export SUP_BATCH=128    # SUP model GPU batch size
export NTHREADS=8       # number of CPU threads (set to _NPROCESSORS_ONLN if unspecified)
export READ_MEM=512M    # max read batch memory in host memory
export READ_BATCH=2048  # max number of reads loaded into host memory

# go into slorado root directory
cd slorado

# test an existing slorado binary by providing the path
./test/extensive.sh /path/to/slorado

# test the latest binary (x86) release on your machine
./test/extensive.sh cuda bin
# OR
./test/extensive.sh rocm bin

# build and test from the repo (after installing the appropriate torch version)
./test/extensive.sh cuda build
# OR
./test/extensive.sh rocm build

```

## Known issues

As of May 1st 2026, LSTM models (HAC and FAST or SUP < v5.0.0) on the 9700 AI Pro (and possibly other newer AMD GPUs) produce incorrect outputs (Transformer models unaffected). This issue is known and can be tracked [here](https://github.com/pytorch/pytorch/issues/177834).

## Demultiplexing

Slorado does not currently support demultiplexing. You can demultiplex reads generated by Slorado by passing them into Dorado. This will work regardless of the device since demultiplexing occurs on the CPU.
```
# basecall reads
./slorado basecaller models/dna_r10.4.1_e8.2_400bps_fast@v5.0.0 reads.blow5 -o reads.fastq

# demux reads
./dorado demux --kit-name <kit-name> --output-dir demux_reads/ reads.fastq
```

## Modification Detection (experimental)

Slorado (from v0.5.0-beta) supports methylation detection for GPU basecalling for HAC v5.0.0 and SUP v5.0.0 DNA basecalling models. Enable methylation detection by appending `--mod 5mCG_5hmCG@v3` when running slorado. Slorado (from v0.6.0) supports m6A_DRACH for RNA HAC v6.0.0, which can be specified as `--mod m6A_DRACH@v1`.
Adding modification detection will automatically output in [SAM](https://samtools.github.io/hts-specs/SAMv1.pdf) format.

```
# example modification calling
./slorado basecaller models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 reads.blow5 --mod 5mCG_5hmCG@v3 -xcuda:all -o reads.fastq
```

## Options

All options supported by slorado basecaller are detailed below:

| Option:           | Description:                                           | Default Value: |
|-------------------|-------------------------------------------------------|----------------|
| -t INT            | number of processing threads                          | 8              |
| -K INT            | batch size (max number of reads loaded at once)       | 4096           |
| -C INT            | gpu batch size (max number of chunks loaded at once)  | 512            |
| -B FLOAT[K/M/G]   | max number of bytes loaded at once                    | 512M           |
| -o FILE           | output to file                                        | stdout         |
| -c INT            | chunk size                                            | 12288          |
| -p INT            | overlap                                               | 150            |
| -x DEVICE         | specify device (e.g., cpu; cuda:0; cuda:1,2; cuda:all)| cuda:all (GPU build) or cpu (CPU build)         |
| -h                | shows help message and exits                          | -              |
| --verbose INT     | verbosity level                                       | 4              |
| --version         | print version                                         |                |
| --flash yes|no    | enable flash attention (from v0.4.0-beta)             | No             |
| --mod STR         | add modification detection (from v0.5.0-beta)         | NULL           |

## Batchsizes

A large batch size (-K and -B) may take up significant RAM during run-time. Similarly, your GPU batch size (-C) will determine how much GPU memory is used. Slorado currently does not implement automatic batch size selection based on available memory. Thus, if you see an out-of-RAM error, reduce the batch size using -K or -B. If you see an out-of-GPU memory error, reduce the GPU batch size using the -C option.

## Flash Attention

Slorado v0.4.0-beta now supports Flash Attention for SUP basecalling models >= v5.0.0 when compiled with CUDA Torch >= v2.4.0 and ROCm Torch >= 2.9.0. This is not guaranteed to work on older GPUs, so we have kept it disabled by default for maximum compatibility. For best runtime performance on modern GPUs (Ampere GPUs or newer on NVIDIA, CDNA2/RDNA3 or newer on AMD), enable Flash Attention with the option `--flash yes`. Other older GPUs maybe supported but are not tested yet.

## Tested models

| slorado version | Tested models |
| ---             | ---           |
| 0.6.0           | dna_r10.4.1_e8.2_400bps v5.0.0 (5mCG_5hmCG@v3); rna004_130bps v5.1.0; rna004_hac v6.0.0 (m6A_DRACH@v1) |
| 0.5.0-beta           | dna_r10.4.1_e8.2_400bps v5.0.0 (5mCG_5hmCG@v3); rna004_130bps v5.1.0 |
| 0.4.0-beta           | dna_r10.4.1_e8.2_400bps v5.0.0; rna004_130bps v5.1.0 |
| 0.3.0-beta           | dna_r10.4.1_e8.2_400bps v4.2.0 and v5.0.0 |
| 0.2.0-beta           | dna_r10.4.1_e8.2_400bps v4.2.0 |

## Acknowledgement

- Slorado uses code from [Dorado](https://github.com/nanoporetech/dorado) which is licensed under [Oxford Nanopore Technologies PLC. Public License Version 1.0](thirdparty/dorado/LICENCE). Those files are located at [thirdparty/dorado](thirdparty/dorado) and thus not covered by the MIT Licence.
- [tomlc99](https://github.com/cktan/tomlc99) library under [thirdparty/tomlc99](thirdparty/tomlc99), is licensed under [MIT license](thirdparty/tomlc99/LICENSE).
- Some code snippets have been taken from [Minimap2](https://github.com/lh3/minimap2).

## Citation:

Please cite the following in your publications when using Slorado:

> Wong, B., Singh, G., Javaid, H., Denolf, K., Liyanage, K., Samarakoon, H., Deveson, I.W. and Gamaarachchi, H., 2026. Open-source, Hardware-Independent GPU Acceleration for Scalable Nanopore Basecalling with Slorado and Openfish. bioRxiv, pp.2026-03.

```
@article{wong2026open,
  title={Open-source, Hardware-Independent GPU Acceleration for Scalable Nanopore Basecalling with Slorado and Openfish},
  author={Wong, Bonson and Singh, Gagandeep and Javaid, Haris and Denolf, Kristof and Liyanage, Kisaru and Samarakoon, Hiruna and Deveson, Ira W and Gamaarachchi, Hasindu},
  journal={bioRxiv},
  pages={2026--03},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```
