FROM ubuntu:22.04
WORKDIR /
RUN apt-get update && apt-get install wget tar xz-utils  -y
RUN VERSION=v0.3.0-beta && GPU=rocm && wget "https://github.com/BonsonW/slorado/releases/download/$VERSION/slorado-$VERSION-x86_64-$GPU-linux-binaries.tar.xz" && ls && tar xvf slorado-$VERSION-x86_64-$GPU-linux-binaries.tar.xz && mv slorado-$VERSION slorado && rm slorado-$VERSION-x86_64-$GPU-linux-binaries.tar.xz
ENV PATH="$PATH:/slorado/bin"
WORKDIR /slorado
CMD bin/slorado basecaller --help

