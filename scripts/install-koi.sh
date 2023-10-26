set -e

test -d koi && rm -r koi
test -e koi.tar.gz && rm  koi.tar.gz
test -d thirdparty/koi_lib && rm -r thirdparty/koi_lib
wget https://cdn.oxfordnanoportal.com/software/analysis/libkoi-0.3.5-Linux-x86_64-cuda-11.8.tar.gz -O koi.tar.gz
mkdir koi
mkdir thirdparty/koi_lib
tar xf koi.tar.gz -C koi/
mv koi/*/* thirdparty/koi_lib
rm -r koi
rm  koi.tar.gz


