set -e

KOI_VERSION=0.4.5
CMAKE_SYSTEM_NAME=Linux
CMAKE_SYSTEM_PROCESSOR=x86_64
KOI_CUDA=11.8

KOI_DIR=libkoi-${KOI_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-cuda-${KOI_CUDA}

test -d koi && rm -r koi
test -e koi.tar.gz && rm  koi.tar.gz
test -d thirdparty/koi_lib && rm -r thirdparty/koi_lib
wget https://cdn.oxfordnanoportal.com/software/analysis/${KOI_DIR}.tar.gz -O koi.tar.gz
mkdir koi
mkdir thirdparty/koi_lib
tar xf koi.tar.gz -C koi/
mv koi/*/* thirdparty/koi_lib
rm -r koi
rm  koi.tar.gz


