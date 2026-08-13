
# Docker

Pre-built images are available on [Docker Hub](https://hub.docker.com/r/hasindu2008/slorado). Builds for both AMD (rocm) and NVIDIA (cuda) are available:

## For AMD (rocm)

```
docker pull hasindu2008/slorado:0.3.0-beta-rocm
```

The image bundles the `slorado` binary (on `PATH`, at `/slorado/bin`) along with the FAST/HAC/SUP v4.2.0 and v5.0.0 DNA models under `/slorado/models`. Print the help message to check the image works:

```
docker run --rm hasindu2008/slorado:0.3.0-beta-rocm slorado basecaller --help
```

To basecall on an AMD GPU, expose the GPU devices to the container and bind mount the directory holding your BLOW5 file:

```
# gid of the render group on the host, needed to access /dev/dri/renderD*
export RENDER_GID=$(stat -c '%g' /dev/dri/renderD128)

docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video --group-add $RENDER_GID \
  -v "$PWD":/data -w /data \
  hasindu2008/slorado:0.3.0-beta-rocm \
  slorado basecaller /slorado/models/dna_r10.4.1_e8.2_400bps_hac@v5.0.0 reads.blow5 -o reads.fastq -x cuda:all
```

Note that `-x cuda:all` selects all GPUs for the rocm build too. If `--group-add video` fails with `Unable to find group video`, pass the numeric gid instead (`$(getent group video | cut -d: -f3)`).


## For NVIDIA (cuda)

Todo