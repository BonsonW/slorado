# Building CPU-only version of slorado on x84_64

1. For slorado versions 0.2.0 to 0.4.0, a minimum g++ version of 5.4 is required due to libtorch v2.0.0 we use currently. This is by default available on Ubuntu 16.04 or higher. You can check your g++ version as `g++ --version`.

2. Install zlib development files needed for slow5lib:

    ```
    On Debian/Ubuntu : sudo apt-get install zlib1g-dev
    On Fedora/CentOS : sudo dnf/yum install zlib-devel
    ```

3. Clone the slorado repository **recursively**

    ```
    git clone --recursive https://github.com/BonsonW/slorado
    cd slorado
    ```

4. Run the script that downloads and extracts torchlib.

    ```
    scripts/install-torch2.sh cpu
    ```

5. Invoke make

    ```
    make -j
    ```

6. Check the compiled slorado version

    ```
    ./slorado --version
    ```

## Advanced building options

- Custom libtorch path:
    ```
    make LIBTORCH_DIR=/path/to/libtorch
    ```

- C++11 ABI (if you are using torch version with C++11 ABI):
    ```
    make cxx11_abi=1
    ```

- You can optionally enable zstd support for built-in slow5lib when building slorado by invoking `make zstd=1`. This requires zstd 1.3 development libraries installed on your system (libzstd1-dev package for apt, libzstd-devel for yum/dnf).

