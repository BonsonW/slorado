#!/bin/bash

die() {
    echo "$1"
    exit 1
}

VERSION=`git describe --tags`
[ -z "$VERSION" ] && VERSION=`git rev-parse --short HEAD`

rm -rf slorado-$VERSION slorado-$VERSION.tar.gz
mkdir -p slorado-$VERSION
mkdir -p slorado-$VERSION/bin slorado-$VERSION/lib
make clean
make rocm=1 -j  || die "Build failed"

mv  ./slorado slorado-$VERSION/bin/ || die "Copy failed"
cp -r thirdparty/torch/libtorch/lib/lib*.so* slorado-$VERSION/lib/ || die "Copy failed"
./slorado-$VERSION/bin/slorado --version || die "Test failed"
tar zcf slorado-$VERSION.tar.gz slorado-$VERSION || die "Tar failed"
rm -rf slorado-$VERSION
