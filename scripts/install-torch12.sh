#!/bin/bash

# MIT License
# Copyright (c) 2019 Hasindu Gamaarachchi (hasindu@unsw.edu.au)
# Copyright (c) 2023 Bonson Wong (bonson.ym@gmail.com)

set -e
set -x
test -e torch.zip && rm torch.zip
test -d torch && rm -r torch
mkdir thirdparty/torch
wget "https://download.pytorch.org/libtorch/cu113/libtorch-shared-with-deps-1.12.1%2Bcu113.zip" -O torch.zip
unzip torch.zip -d thirdparty/torch/
rm torch.zip

