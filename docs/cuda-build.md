# Building CUDA version of slorado on x84_64

1. To build for the NVIDIA GPU, you need to have the CUDA toolkit installed.

2. A minimum g++ version of 5.4 (available on Ubuntu 16.04 or higher) is required as of 18.11.2024 due to libtorch v2.0.0 we use currently.

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
    scripts/install-torch2.sh cuda
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

- By default it is assumed that you have CUDA on the standard location (`/usr/local/cuda/`). Otherwise, you can specify the path manually as:
   ```
   make cuda=1 CUDA_LIB=/path/to/cuda/library/
   ```
   Make sure you have `/path/to/cuda/library/bin/nvcc` and `/path/to/cuda/library/lib64`.

- For CUDA you can provide the architecture as `make cuda=1 CUDA_ARCH=-arch=sm_xy`

- Custom libtorch path:
    ```
    make LIBTORCH_DIR=/path/to/libtorch
    ```

- C++11 ABI (if you are using torch version with C++11 ABI):
    ```
    make cxx11_abi=1
    ```

- You can optionally enable zstd support for built-in slow5lib when building slorado by invoking `make zstd=1`. This requires zstd 1.3 development libraries installed on your system (libzstd1-dev package for apt, libzstd-devel for yum/dnf).

