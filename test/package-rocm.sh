#!/bin/bash

die() {
    echo "$1"
    exit 1
}

scl enable gcc-toolset-9 bash || die "Enable devtoolset-9 failed"

git submodule update || die "Update failed"

VERSION=`git describe --tags`
[ -z "$VERSION" ] && VERSION=`git rev-parse --short HEAD`

rm -rf slorado-$VERSION slorado-*.tar.gz models/
scripts/download-models.sh || die "Download models failed"
scripts/install-zstd.sh || die "Install zstd failed"

test -d thirdparty/torch/libtorch || scripts/install-torch2.sh rocm || die "Install torch failed"

mkdir -p slorado-$VERSION
mkdir -p slorado-$VERSION/bin slorado-$VERSION/lib slorado-$VERSION/share

# roc-obj-ls thirdparty/torch/libtorch/lib/libtorch_hip.so  | awk '{print $2}' | sort -u
# https://github.com/pytorch/pytorch/blob/89a1835a782b4053422a37274e0e873069cc3cbd/.ci/docker/manywheel/build.sh#L94

make clean
make cxx11_abi=1 rocm=1 -j zstd_local=../zstd/lib ROCM_ARCH='"--offload-arch=gfx900 --offload-arch=gfx906 --offload-arch=gfx908 --offload-arch=gfx90a --offload-arch=gfx942 --offload-arch=gfx1030 --offload-arch=gfx1100 --offload-arch=gfx1101 --offload-arch=gfx1102 --offload-arch=gfx1200 --offload-arch=gfx1201 --offload-arch=gfx950 --offload-arch=gfx1150 --offload-arch=gfx1151"' || die "Build failed"


rm -f thirdparty/torch/libtorch/lib/libamdhip64.so.6
ln -s libamdhip64.so thirdparty/torch/libtorch/lib/libamdhip64.so.6 || die "Link failed"
cp -r thirdparty/torch/libtorch/lib/* slorado-$VERSION/lib/ || die "Copy failed"
cp -r thirdparty/torch/libtorch/share/* slorado-$VERSION/share/ || die "Copy failed"
mv  ./slorado slorado-$VERSION/bin/ || die "Copy failed"
cp -r models/ slorado-$VERSION/ || die "Copy failed"


# clean up
# mv slorado-$VERSION/build-hash slorado-$VERSION/build-version  slorado-$VERSION/share/ || die "mv failed"
cp -r thirdparty/torch/torch-2.9.0+rocm6.4.dist-info/ slorado-$VERSION/share/ || die "Copy failed"

#rm -r slorado-$VERSION/include
rm -r slorado-$VERSION/share/cmake
#rm -r slorado-$VERSION/lib/*.a

for f in slorado-$VERSION/lib/*.so*; do
   patchelf --force-rpath  --set-rpath '$ORIGIN' $f || die "Failed to patchelf $f"
done


# ldd ./slorado-$VERSION/bin/slorado | grep -v hasindu
cp /lib64/libzstd.so.1 slorado-$VERSION/lib/libzstd.so.1
cp /lib64/liblzma.so.5 slorado-$VERSION/lib/liblzma.so.5
cp /lib64/libbz2.so.1 slorado-$VERSION/lib/libbz2.so.1

# #
#         linux-vdso.so.1 (0x00007ffe3ffeb000)
#         libpthread.so.0 => /lib64/libpthread.so.0 (0x00007fae45f18000)
#         libz.so.1 => /lib64/libz.so.1 (0x00007fae311e9000)
#         libstdc++.so.6 => /lib64/libstdc++.so.6 (0x00007fae2f4f1000)
#         libm.so.6 => /lib64/libm.so.6 (0x00007fae2f16f000)
#         libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007fae2ef57000)
#         libc.so.6 => /lib64/libc.so.6 (0x00007fae2eb92000)
#         /lib64/ld-linux-x86-64.so.2 (0x00007fae5b09a000)
#         librt.so.1 => /lib64/librt.so.1 (0x00007fae2e98a000)
#         libdl.so.2 => /lib64/libdl.so.2 (0x00007fae2e786000)


./slorado-$VERSION/bin/slorado --version || die "Test failed"
tar cJf slorado-$VERSION-x86_64-rocm-linux-binaries.tar.xz slorado-$VERSION || die "Tar failed"
rm -rf slorado-$VERSION
