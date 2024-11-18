
# Buiding for ARM64-based Jetson devices

1. Install zlib development files needed for slow5lib:

```
On Debian/Ubuntu : sudo apt-get install zlib1g-dev
```

2. Install and activate python venv (need for getting pytorch that includes torchlib, as as of 1811/2024 no binary libtorch download is available)

    ```
    sudo apt install python3.8-venv
    python3 -m venv pytorch_venv
    source pytorch_venv/bin/activate
    ```

3. Update pip and install pytorch for your specific Nvidia Jetpack version. You can find this by running `sudo apt-cache show nvidia-jetpack | grep "Version"`, or browse https://developer.download.nvidia.com/compute/redist/jp/ to find a suitable version of pytorch. We tested on a Jetson Xavier board with Jetpack 5.0 installed and the commands used were:

    ```
    pip3 install --upgrade pip
    pip3 install --no-cache  https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/torch-2.0.0a0+8aa34602.nv23.03-cp38-cp38-linux_aarch64.whl
    ```

4. Clone and build.

    ```
    git clone --recursive https://github.com/BonsonW/slorado.git
    cd slorado
    make -j cuda=1 cxx11_abi=1 LIBTORCH_DIR=/path/to/pytorch_venv/lib64/python3.8/site-packages/torch/
    ```

## Advanced building options

- You can optionally enable zstd support for built-in slow5lib when building slorado by invoking `make zstd=1`. This requires zstd 1.3 development libraries installed on your system (libzstd1-dev package for apt, libzstd-devel for yum/dnf).
