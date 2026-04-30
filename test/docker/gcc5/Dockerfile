FROM ubuntu:16.04

# Fix broken repositories
RUN printf "deb http://old-releases.ubuntu.com/ubuntu xenial main restricted universe multiverse\n\
deb http://old-releases.ubuntu.com/ubuntu xenial-updates main restricted universe multiverse\n\
deb http://old-releases.ubuntu.com/ubuntu xenial-security main restricted universe multiverse\n" > /etc/apt/sources.list

# Disable validity checks (required for EOL repos)
RUN apt-get -o Acquire::Check-Valid-Until=false update

# Install toolchain
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    g++-5 \
    gcc-5 \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set gcc/g++ 5 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 100

WORKDIR /workspace