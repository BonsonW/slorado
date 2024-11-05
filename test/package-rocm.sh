#!/bin/bash

die() {
    echo "$1"
    exit 1
}

VERSION=`git describe --tags`
[ -z "$VERSION" ] && VERSION=`git rev-parse --short HEAD`

rm -rf slorado-$VERSION slorado-*.tar.gz models/
test/download-models.sh || die "Download models failed"

mkdir -p slorado-$VERSION
mkdir -p slorado-$VERSION/bin slorado-$VERSION/lib slorado-$VERSION/share
make clean
make rocm=1 -j  || die "Build failed"

cp -r thirdparty/torch/libtorch/* slorado-$VERSION/|| die "Copy failed"
mv  ./slorado slorado-$VERSION/bin/ || die "Copy failed"
cp -r models/ slorado-$VERSION/ || die "Copy failed"

./slorado-$VERSION/bin/slorado --version || die "Test failed"
tar zcf slorado-$VERSION.tar.gz slorado-$VERSION || die "Tar failed"
rm -rf slorado-$VERSION
