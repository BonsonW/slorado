#!/bin/bash

set -e
set -x
test -e torch.zip && rm torch.zip
test -d torch && rm -r torch
mkdir torch/
wget "https://download.pytorch.org/libtorch/cu102/libtorch-shared-with-deps-1.10.2%2Bcu102.zip" -O torch.zip
unzip torch.zip -d thirdparty/torch/
rm torch.zip

