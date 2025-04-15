#!/bin/bash

die() {
    echo "$1"
    exit 1
}

git submodule update || die "Update failed"

VERSION=`git describe --tags`
[ -z "$VERSION" ] && VERSION=`git rev-parse --short HEAD`

rm -rf slorado-$VERSION slorado-*.tar.gz models/
scripts/download-models.sh || die "Download models failed"
scripts/install-zstd.sh || die "Install zstd failed"

test -d thirdparty/torch/libtorch || scripts/install-torch2.sh rocm || die "Install torch failed"

mkdir -p slorado-$VERSION
mkdir -p slorado-$VERSION/bin slorado-$VERSION/lib slorado-$VERSION/share

# roc-obj-ls libtorch_hip.so  | awk '{print $2}' | sort -u
make clean
make rocm=1 -j zstd_local=../zstd/lib ROCM_ARCH='"--offload-arch=gfx1030 --offload-arch=gfx1100 --offload-arch=gfx900 --offload-arch=gfx906 --offload-arch=gfx908 --offload-arch=gfx90a"' || die "Build failed"
rm -f thirdparty/torch/libtorch/lib/libamdhip64.so.5
ln -s libamdhip64.so thirdparty/torch/libtorch/lib/libamdhip64.so.5 || die "Link failed"
cp -r thirdparty/torch/libtorch/* slorado-$VERSION/|| die "Copy failed"
mv  ./slorado slorado-$VERSION/bin/ || die "Copy failed"
cp -r models/ slorado-$VERSION/ || die "Copy failed"

# clean up
mv slorado-$VERSION/build-hash slorado-$VERSION/build-version  slorado-$VERSION/share/ || die "mv failed"
rm -r slorado-$VERSION/include
rm -r slorado-$VERSION/share/cmake
rm -r slorado-$VERSION/lib/*.a

./slorado-$VERSION/bin/slorado --version || die "Test failed"
tar cJf slorado-$VERSION-x86_64-rocm-linux-binaries.tar.xz slorado-$VERSION || die "Tar failed"
rm -rf slorado-$VERSION
