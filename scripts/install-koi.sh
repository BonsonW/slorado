set -e

test -d koi && rm -r koi
test -e koi.tar.gz && rm  koi.tar.gz
test -d thirdparty/koi_lib && rm -r thirdparty/koi_lib
wget https://nanoporetech.box.com/shared/static/tphtmha3qoyd6j2amyr40qq0ebed4b4d.gz -O koi.tar.gz
mkdir koi
mkdir thirdparty/koi_lib
tar xf koi.tar.gz -C koi/
mv koi/*/* thirdparty/koi_lib
rm -r koi
rm  koi.tar.gz


