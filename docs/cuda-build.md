# Building CUDA version of slorado on x84_64

1. To build for the NVIDIA GPU, you need to have the CUDA toolkit installed. See [below](#tested-versions-and-requirements) for the CUDA toolkit versions we have tested.

2. A minimum g++ version as listed [below](#tested-versions-and-requirements) is required due to libtorch. You check check your g++ version by invoking `g++ --version`.

3. Install zlib development files needed for slow5lib:

    ```
    On Debian/Ubuntu : sudo apt-get install zlib1g-dev
    On Fedora/CentOS : sudo dnf/yum install zlib-devel
    ```

4. Clone the slorado repository **recursively**

    ```
    git clone --recursive https://github.com/BonsonW/slorado
    cd slorado
    ```

5. Run the script that downloads and extracts torchlib.

    ```
    scripts/install-torch2.sh cuda
    ```

5. Invoke make

    ```
    make cuda=1 -j
    ```

6. See the slorado version

    ```
    ./slorado --version
    ```

## Advanced building options

- By default it is assumed that you have CUDA on the standard system location (`/usr/local/cuda/`). There should be the `nvcc` compiler at `/usr/local/cuda/bin/nvcc`. The library files (.a files) should be present under `/usr/local/cuda/lib64/`. If the CUDA location is different, you can specify the path manually as:
   ```
   make cuda=1 CUDA_ROOT=/path/to/cuda/
   ```
   Make sure you have `nvcc` at `/path/to/cuda/bin/nvcc` and library files at `/path/to/cuda/lib64`.

- You can provide the CUDA architecture as `make cuda=1 CUDA_ARCH=-arch=sm_xy`

- Custom libtorch path:
    ```
    make LIBTORCH_DIR=/path/to/libtorch
    ```

- C++11 ABI (if you are using torch version with C++11 ABI):
    ```
    make cxx11_abi=1
    ```

- You can optionally enable zstd support for built-in slow5lib when building slorado by invoking `make zstd=1`. This requires zstd 1.3 development libraries installed on your system (libzstd1-dev package for apt, libzstd-devel for yum/dnf).


## Tested versions and requirements

Note that we have tested compilation on a limited number of combinations and the minimum g++ requirements are as below.

| Slorado version | tested libtorch | tested CUDA | minimum g++ | comments |
|---              | ---             | ---         | ---         | ---      |
| 0.4.0           | 2.0.0           | 11.8        | 5.4         | Flash Attention unsupported          |
| 0.4.0           | 2.4.0           | 11.8        | 9           | Fused RMSNorm unsupported          |
| 0.4.0           | 2.9.0           | 12.6        | xx         |           |
||||
| 0.2.0-0.3.0     | 2.0.0           | 10,11,12    | 5.4         | |

