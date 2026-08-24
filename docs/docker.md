
# Docker

Pre-built images are available on [Docker Hub](https://hub.docker.com/r/hasindu2008/slorado). The image bundles the `slorado` binary (on `PATH`, at `/slorado/bin`) along with the well tested models under `/slorado/models`.
The image also bundles `blue-crab` and `slow5tools`.

Builds for both AMD (rocm) and NVIDIA (cuda) are available.

## For AMD (rocm)

```
docker pull slorado:0.6.0-rocm
```

Print the help message to check the image works:

```
docker run --rm hasindu2008/slorado:0.6.0-rocm slorado basecaller --help
```

To basecall on an AMD GPU, expose the GPU devices to the container and bind mount the directory holding your BLOW5 file:

```
docker run -v "$PWD":/data  \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video \
  hasindu2008/slorado:0.6.0-rocm \
  slorado basecaller /slorado/models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 /data/reads.blow5 -o /data/reads.fastq -x cuda:all
```

Note that `-x cuda:all` selects all GPUs for the rocm build too. If `--group-add video` fails with `Unable to find group video`, pass the numeric gid instead (`$(getent group video | cut -d: -f3)`).


## For NVIDIA (cuda)

```
docker pull hasindu2008/slorado:0.6.0-cuda
```

Print the help message to check the image works:
```
docker run --rm hasindu2008/slorado:0.6.0-cuda slorado basecaller --help
```

To basecall on an NVIDIA GPU, expose the GPU devices to the container and bind mount the directory holding your BLOW5 file:

```
docker run -v "$PWD":/data --gpus all \
  hasindu2008/slorado:0.6.0-cuda \
  slorado basecaller /slorado/models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 /data/reads.blow5 -o /data/reads.fastq -x cuda:all
```
