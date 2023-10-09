#!/bin/bash

# MIT License
# Copyright (c) 2019 Hasindu Gamaarachchi (hasindu@unsw.edu.au)
# Copyright (c) 2023 Bonson Wong (bonson.ym@gmail.com)

set -e
set -x
test -e torch.zip && rm torch.zip
test -d torch && rm -r torch
mkdir thirdparty/torch
wget "https://download.pytorch.org/libtorch/cu118/libtorch-shared-with-deps-2.0.0%2Bcu118.zip" -O torch.zip
unzip torch.zip -d thirdparty/torch/
rm torch.zip

