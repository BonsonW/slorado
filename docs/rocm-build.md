# Building ROCM version of slorado on x84_64

1. To build for the AMD GPU, you need to have the ROCM SDK installed. See [below](#tested-versions-and-requirements) for the versions that we have tested.

2. A minimum g++ version as listed [below](#tested-versions-and-requirements) is required due to libtorch. You check check your g++ version as `g++ --version`.

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
    scripts/install-torch2.sh rocm
    ```

5. Call make

    ```
    make rocm=1 -j
    ```

6. See the slorado version

    ```
    ./slorado --version
    ```

## Advanced building options

- By default it is assumed that you have ROCM on the standard location (`/opt/rocm`). You should have `hipcc` at `/opt/rocm/bin/hipcc` and the library files (.so files) at `/opt/rocm/lib`. If your ROCM is installed elsewhere,  you can specify the path manually as:
   ```
   make rocm=1 ROCM_ROOT=/path/to/rocm/
   ```
   Make sure you have `hipcc` at `/path/to/rocm/bin/hipcc` and the library files at `/path/to/rocm/lib`.

- You can provide the [ROCM architecture](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html) as `make rocm=1 ROCM_ARCH=--offload-arch=gfxnnn`

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

| Slorado version | tested libtorch | tested ROCM | minimum g++ (approximate) | cxx11_abi | comments |
|---              | ---             | ---         | ---         | ---      | ---      |
| 0.4.0           | 2.0.0           | 5.7.0        | 5.4         | no | Flash Attention and Fused RMSNorm layer unsupported for SUP >= v5.0.0         |
| 0.4.0           | 2.9.0           | 6.3.0        | x.x         | yes | |
||||
| 0.3.0-beta,0.2.0-beta     | 2.2.0           | [5.7.x](https://rocm.docs.amd.com/en/docs-5.7.1/deploy/linux/os-native/install.html)   | 9  | no         |

