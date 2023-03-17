set -e

test -d toml11 && rm -r toml11
test -e toml11.tar.gz && rm  toml11.tar.gz
test -d thirdparty/toml11 && rm -r thirdparty/toml11
wget https://github.com/ToruNiina/toml11/archive/refs/tags/v3.7.1.tar.gz -O toml11.tar.gz
mkdir toml11
mkdir thirdparty/toml11
tar xf toml11.tar.gz -C toml11/
mv toml11/*/* thirdparty/toml11
rm -r toml11
rm  toml11.tar.gz


