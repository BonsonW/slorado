#!/bin/bash

# MIT License
# Copyright (c) 2019 Hasindu Gamaarachchi (hasindu@unsw.edu.au)
# Copyright (c) 2023 Bonson Wong (bonson.ym@gmail.com)

die () {
    echo "$@" >&2
    exit 1
}

DEV=cuda
WHL=0
test -z $1 || DEV=$1
if [ ${DEV} == "cpu" ]; then
    VER=2.0.0
elif [ ${DEV} == "cuda" ]; then
    VER=2.4.0
elif [ ${DEV} == "rocm" ]; then
    VER=2.9.0
else
    die "Unknown DEV option ${DEV}. Supported options are: cpu, cuda, rocm"
fi


test -z $2 || VER=$2


test -e torch.zip && rm torch.zip
test -d torch && rm -r torch
mkdir thirdparty/torch || die "Could not create directory thirdparty/torch"

if [ ${DEV} = "cpu" ]; then
    if [ ${VER} != "2.0.0" ] ; then
        die "Only Torch 2.0.0 is tested for slorado CPU version"
    fi
    LINK="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.0%2Bcpu.zip"
    MAKE_CMD="make -j"
elif [ ${DEV} = "cuda" ]; then
    if [ ${VER} = "2.0.0" ] ; then
        LINK="https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.0.0%2Bcu118.zip"
        MAKE_CMD="make -j cuda=1"
    elif [ ${VER} = "2.4.0" ] ; then
        LINK="https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.4.0%2Bcu118.zip"
        MAKE_CMD="make -j cuda=1"
    elif [ ${VER} = "2.9.0" ] ; then
        LINK="https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.9.0%2Bcu128.zip"
        MAKE_CMD="make -j cuda=1 cxx11_abi=1"
    else
        die "Untested Torch version ${VER} for CUDA. Tested versions are: 2.0.0, 2.4.0, 2.9.0"
    fi
elif [ ${DEV} = "rocm" ]; then
    if [ ${VER} == "2.2.0" ] ; then
        LINK="https://download.pytorch.org/libtorch/rocm5.7/libtorch-shared-with-deps-2.2.0%2Brocm5.7.zip"
        MAKE_CMD="make -j rocm=1"
    elif [ ${VER} == "2.9.0" ] ; then
        LINK="https://download.pytorch.org/whl/rocm6.4/torch-2.9.0%2Brocm6.4-cp310-cp310-manylinux_2_28_x86_64.whl"
        WHL=1
        MAKE_CMD="make -j rocm=1 cxx11_abi=1"
    else
        die "Untested Torch version ${VER} for ROCm. Tested versions are: 2.2.0, 2.9.0"
    fi
fi

echo "Downloading Torch ${VER} for ${DEV}. Please wait ..."
echo "Download link: ${LINK}"
wget ${LINK} -O torch.zip || die "Could not download torch"
unzip torch.zip -d thirdparty/torch/ || die "Could not extract torch"
if [ ${WHL} -eq 1 ] ; then
    mv thirdparty/torch/torch thirdparty/torch/libtorch
    #would need the  make rocm=1  cxx11_abi=1
fi
rm torch.zip
echo ""
echo "Torch ${VER} for ${DEV} installed successfully from ${LINK} to thirdparty/torch/"
echo "Now invoke '${MAKE_CMD}' to build slorado"
