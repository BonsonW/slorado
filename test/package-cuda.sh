#!/bin/bash

die() {
    echo "$1"
    exit 1
}

source /opt/rh/devtoolset-8/enable || die "Enable devtoolset-8 failed"

git submodule update || die "Update failed"

VERSION=`git describe --tags`
[ -z "$VERSION" ] && VERSION=`git rev-parse --short HEAD`

rm -rf slorado-$VERSION slorado-*.tar.gz models/
scripts/download-models.sh || die "Download models failed"
scripts/install-zstd.sh || die "Install zstd failed"

test -d thirdparty/torch/libtorch || scripts/install-torch2.sh cuda || die "Install torch failed"

mkdir -p slorado-$VERSION
mkdir -p slorado-$VERSION/bin slorado-$VERSION/lib slorado-$VERSION/share

# roc-obj-ls libtorch_hip.so  | awk '{print $2}' | sort -u
make clean
make cuda=1 zstd_local=../zstd/lib || die "Build failed"
cp -r thirdparty/torch/libtorch/* slorado-$VERSION/|| die "Copy failed"
mv  ./slorado slorado-$VERSION/bin/ || die "Copy failed"
cp -r models/ slorado-$VERSION/ || die "Copy failed"

# clean up
mv slorado-$VERSION/build-hash slorado-$VERSION/build-version  slorado-$VERSION/share/ || die "mv failed"
rm -r slorado-$VERSION/include
rm -r slorado-$VERSION/share/cmake
rm -r slorado-$VERSION/lib/*.a
rm slorado-$VERSION/lib/

./slorado-$VERSION/bin/slorado --version || die "Test failed"
tar zcf slorado-$VERSION-x86_64-cuda-linux-binaries.tar.gz slorado-$VERSION || die "Tar failed"
rm -rf slorado-$VERSION
