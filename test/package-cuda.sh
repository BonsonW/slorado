#!/bin/bash

die() {
    echo "$1"
    exit 1
}

source /opt/rh/devtoolset-9/enable || die "Enable devtoolset-9 failed"

git submodule update || die "Update failed"

VERSION=`git describe --tags`
[ -z "$VERSION" ] && VERSION=`git rev-parse --short HEAD`
# VERSION=v0.4.0-beta

rm -rf slorado-$VERSION slorado-*.tar.gz models/
scripts/download-models.sh || die "Download models failed"
scripts/install-zstd.sh || die "Install zstd failed"

test -d thirdparty/torch/libtorch || scripts/install-torch2.sh cuda || die "Install torch failed"

mkdir -p slorado-$VERSION
mkdir -p slorado-$VERSION/bin slorado-$VERSION/lib slorado-$VERSION/share

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
rm -f slorado-$VERSION/lib/libtorch_cuda_linalg.so slorado-$VERSION/lib/libnvrtc-builtins-*
rm -f slorado-$VERSION/lib/libcudnn_cnn_train.so.8 slorado-$VERSION/lib/libcudnn_ops_train.so.8 slorado-$VERSION/lib/libcudnn_adv_train.so.8

for f in slorado-$VERSION/lib/*.so*; do
   patchelf --force-rpath  --set-rpath '$ORIGIN' $f || die "Failed to patchelf $f"
done

./slorado-$VERSION/bin/slorado --version || die "Test failed"
tar cJf slorado-$VERSION-x86_64-cuda-linux-binaries.tar.xz slorado-$VERSION || die "Tar failed"
rm -rf slorado-$VERSION

#ldd ./sloraodo-$VERSION/bin/slorado | grep -v hasindu
        # linux-vdso.so.1 =>  (0x00007fffed6f5000)
        # libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f628ac10000)
        # libz.so.1 => /lib/x86_64-linux-gnu/libz.so.1 (0x00007f622ff30000)
        # librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007f622fd20000)
        # libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f622fb00000)
        # libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f622f770000)
        # libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f622f460000)
        # libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f622f230000)
        # libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f622ee60000)
        # /lib64/ld-linux-x86-64.so.2 (0x00007f62a3800000)




