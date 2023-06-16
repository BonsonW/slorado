#!/bin/bash

# terminate script
die() {
	echo "$1" >&2
	echo
	exit 1
}

if [ "$1" = 'mem' ]; then
    mem=1
else
    mem=0
fi

ex() {
    if [ $mem -eq 1 ]; then
        valgrind --leak-check=full --error-exitcode=1 "$@"
    else
        "$@"
    fi
}

test -z "$DEVICE" && DEVICE=cpu

download_model () {
    test -e dna_r10.4.1_e8.2_400bps_fast.zip && rm dna_r10.4.1_e8.2_400bps_fast.zip
    test -d dna_r10.4.1_e8.2_400bps_fast@v4.0.0 && rm -r dna_r10.4.1_e8.2_400bps_fast@v4.0.0
    wget https://nanoporetech.box.com/shared/static/6xmmoltxeo8budtsxlak4qi0130m3opx.zip -O dna_r10.4.1_e8.2_400bps_fast.zip || die "Downloading the model failed"
    unzip dna_r10.4.1_e8.2_400bps_fast.zip || die "Unzipping the model failed"
    test -d models || mkdir models || die "Creating the models directory failed"
    mv dna_r10.4.1_e8.2_400bps_fast@v4.0.0 models/ || die "Moving the model failed"
    rm -f dna_r10.4.1_e8.2_400bps_fast.zip || die "Removing the model failed"
}

download_minimap2 () {

    uname -m || die "Could not determine the architecture. "
    ARCH=$(uname -m)

    if [ ${ARCH} = "x86_64" ];
    then
        wget https://github.com/lh3/minimap2/releases/download/v2.24/minimap2-2.24_x64-linux.tar.bz2
        tar xf minimap2-2.24_x64-linux.tar.bz2
        mv minimap2-2.24_x64-linux minimap2
        rm minimap2-2.24_x64-linux.tar.bz2
    elif [ ${ARCH} = "aarch64" ];
    then
        wget https://github.com/lh3/minimap2/releases/download/v2.24/minimap2-2.24.tar.bz2
        tar xf minimap2-2.24.tar.bz2
        mv minimap2-2.24 minimap2
        rm minimap2-2.24.tar.bz2
        cd minimap2
        make arm_neon=1 aarch64=1
        cd ..
    else
        die "Unsupported architecture"
    fi
}

check_accuracy () {
    if (( $(echo "$1 >= 0.8" | bc -l) ));
    then
        return 0
    fi

    die "Failed accuracy test with value of $2"
}

test -d models/dna_r10.4.1_e8.2_400bps_fast@v4.0.0 || download_model
test -e minimap2/minimap2 || download_minimap2

make clean && make -j cuda=1 koi=1 CUDA_ROOT=/data/install/cuda-11.3

# echo "Test 1"
ex  ./slorado basecaller models/dna_r10.4.1_e8.2_400bps_fast@v4.0.0 test/oneread_r10.blow5 --device "$DEVICE" > test/tmp.fastq  || die "Running the tool failed"
minimap2/minimap2 -cx map-ont test/chr4_90700000_90900000.fa test/tmp.fastq --secondary=no > test/tmp.paf || die "minimap2 failed"
check_accuracy $(awk '{print $10/$11}' test/tmp.paf | datamash median 1)

echo "Tests passed"