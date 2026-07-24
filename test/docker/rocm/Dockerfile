FROM ubuntu:22.04
WORKDIR /
RUN apt-get update && apt-get install wget tar xz-utils  -y
RUN VERSION=v0.5.0-beta && GPU=rocm && wget "https://cdn.bioinf.science/slorado/slorado-$VERSION-x86_64-$GPU-linux-binaries.tar.xz" && ls && tar xvf slorado-$VERSION-x86_64-$GPU-linux-binaries.tar.xz && mv slorado-$VERSION slorado && rm slorado-$VERSION-x86_64-$GPU-linux-binaries.tar.xz
RUN VERSION=v1.4.0 && wget "https://github.com/hasindu2008/slow5tools/releases/download/$VERSION/slow5tools-$VERSION-x86_64-linux-binaries.tar.gz" && ls && tar xvf slow5tools-$VERSION-x86_64-linux-binaries.tar.gz && mv slow5tools-$VERSION slow5tools && rm slow5tools-$VERSION-x86_64-linux-binaries.tar.gz
RUN VERSION=v0.5.0 && wget "https://github.com/Psy-Fer/blue-crab/releases/download/$VERSION/blue-crab-$VERSION-x86_64-linux-binaries.tar.gz" && ls && tar xvf blue-crab-$VERSION-x86_64-linux-binaries.tar.gz && mv blue-crab-$VERSION blue-crab && rm blue-crab-$VERSION-x86_64-linux-binaries.tar.gz
ENV PATH="$PATH:/slorado/bin:/slow5tools/:/blue-crab"
WORKDIR /slorado
CMD slow5tools --version && blue-crab --version && slorado basecaller --version
CMD slorado basecaller --help

