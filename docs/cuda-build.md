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

    This will download the torch version marked default in table [below](#tested-versions-and-requirements).
    You can specify a different torch version in this table, for example `scripts/install-torch2.sh cuda 2.0.0`.

5. Invoke make

    ```
    make cuda=1 -j
    ```

    Note that if the particular torch version in table [below](#tested-versions-and-requirements) states "yes" for cxx11_abi, you should invoke as `make cuda=1 -j cxx11_abi=1`

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

Note that we have tested compilation on a limited number of combinations and the approximate minimum g++ requirements are as below.

| Slorado version | tested libtorch | tested CUDA | minimum g++ (approximate) | cxx11_abi | comments |
|---              | ---             | ---         | ---          | ---      | ---      |
| 0.4.0-beta           | 2.0.0           | 11        | 5.4          | no | Flash Attention unsupported for SUP >= v5.0.0         |
| 0.4.0-beta           | 2.4.0 (default)           | 11        | 9            | no | Fused RMSNorm unsupported for SUP >= v5.0.0           |
| 0.4.0-beta           | 2.9.0           | 12        | 9            | yes |           |
||||
| 0.2.0-beta,0.3.0-beta     | 2.0.0           | 10,11,12    | 5.4         | no |  |

