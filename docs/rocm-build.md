# Building ROCM version of slorado on x84_64

1. To build for the AMD GPU, you need to have the ROCM SDK installed. As of 18.11.2024, as we are using libtorch v2.2.0 with rocm 5.7, you need [rocm 5.7.x](https://rocm.docs.amd.com/en/docs-5.7.1/deploy/linux/os-native/install.html).

2. A minimum g++ version of 9 (available on Ubuntu 20.04 or higher) is required as of 18.11.2024 due to libtorch v2.2.0 we use currently.

3. Install zlib development files needed for slow5lib:

    ```
    On Debian/Ubuntu : sudo apt-get install zlib1g-dev
    On Fedora/CentOS : sudo dnf/yum install zlib-devel
    ```

4. Clone the slorado repository recursively

    ```
    git clone --recursive https://github.com/BonsonW/slorado
    cd slorado
    ```

5. Run the script that downloads and extracts torchlib.

    ```
    scripts/install-torch2.sh rocm
    ```

5. Call make

    ```
    make cuda=1 -j
    ```

6. See the slorado version

    ```
    ./slorado --version
    ```

## Advanced building options

- By default it is assumed that you have ROCM on the standard location (`/opt/rocm`). Otherwise, you can specify the path manually as:
   ```
   make rocm=1 ROCM_LIB=/path/to/rocm/library/
   ```
   Make sure you have `/path/to/curocmda/library/bin/hipcc` and `/path/to/cuda/library/lib`.

- For ROCM you can provide the architecture as `make rocm=1 ROCM_ARCH=--offload-arch=gfxnnn``

- Custom libtorch path:
    ```
    make LIBTORCH_DIR=/path/to/libtorch
    ```

- C++11 ABI (if you are using torch version with C++11 ABI):
    ```
    make cxx11_abi=1
    ```

- You can optionally enable zstd support for built-in slow5lib when building slorado by invoking `make zstd=1`. This requires zstd 1.3 development libraries installed on your system (libzstd1-dev package for apt, libzstd-devel for yum/dnf).

