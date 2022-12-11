set -e

test -d koi && rm -r koi
test -e koi.tar.gz && rm  koi.tar.gz
wget https://nanoporetech.box.com/shared/static/qbasibmplodr2ixztz97v53vmkttxia1.gz -O koi.tar.gz
mkdir thirdparty/koi_lib
tar xf koi.tar.gz -C thirdparty/koi_lib
#mv koi/*/* koi/
rm -r  koi.tar.gz


