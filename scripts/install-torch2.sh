#!/bin/bash

# MIT License
# Copyright (c) 2019 Hasindu Gamaarachchi (hasindu@unsw.edu.au)
# Copyright (c) 2023 Bonson Wong (bonson.ym@gmail.com)

DEV=cuda
test -z $1 || DEV=$1

set -e
set -x
test -e torch.zip && rm torch.zip
test -d torch && rm -r torch
mkdir thirdparty/torch

if [ ${DEV} = cuda ]; then
    LINK="https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.0.0%2Bcu118.zip"
elif [ ${DEV} = cpu ]; then
    LINK="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.0%2Bcpu.zip"
elif [ ${DEV} = rocm ]; then
    LINK="https://download.pytorch.org/libtorch/rocm5.4.2/libtorch-shared-with-deps-2.0.0%2Brocm5.4.2.zip"
fi

wget ${LINK} -O torch.zip
unzip torch.zip -d thirdparty/torch/
rm torch.zip

