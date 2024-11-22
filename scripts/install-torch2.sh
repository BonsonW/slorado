#!/bin/bash

# MIT License
# Copyright (c) 2019 Hasindu Gamaarachchi (hasindu@unsw.edu.au)
# Copyright (c) 2023 Bonson Wong (bonson.ym@gmail.com)

DEV=cuda
test -z $1 || DEV=$1

die () {
    echo "$@" >&2
    exit 1
}

test -e torch.zip && rm torch.zip
test -d torch && rm -r torch
mkdir thirdparty/torch || die "Could not create directory thirdparty/torch"

if [ ${DEV} = cuda ]; then
    LINK="https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.0.0%2Bcu118.zip"
elif [ ${DEV} = cpu ]; then
    LINK="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.0%2Bcpu.zip"
elif [ ${DEV} = rocm ]; then
    LINK="https://download.pytorch.org/libtorch/rocm5.7/libtorch-shared-with-deps-2.2.0%2Brocm5.7.zip"
fi

wget ${LINK} -O torch.zip || die "Could not download torch"
unzip torch.zip -d thirdparty/torch/ || die "Could not extract torch"
rm torch.zip

