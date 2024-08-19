#!/bin/bash

FAST="dna_r10.4.1_e8.2_400bps_fast@v4.2.0"

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
    test -e $1.zip && rm $1.zip
    test -d $1 && rm -r $1
    wget https://cdn.oxfordnanoportal.com/software/analysis/dorado/${1}.zip -O $1.zip || die "Downloading the model failed"
    unzip $1.zip || die "Unzipping the model failed"
    test -d models || mkdir models || die "Creating the models directory failed"
    mv $1 models/ || die "Moving the model failed"
    rm -f $1.zip || die "Removing the model failed"
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

test -d models/$FAST || download_model $FAST
test -e minimap2/minimap2 || download_minimap2

ex  ./slorado basecaller models/$FAST test/5khz_r10/one_5khz.blow5 -c 1000 -C 100 --device $DEVICE -v 6 > test/tmp.fastq  || die "Running the tool failed"
minimap2/minimap2 -cx map-ont test/chr3_34011000_34012000.fa test/tmp.fastq --secondary=no > test/tmp.paf || die "minimap2 failed"
check_accuracy $(awk '{print $10/$11}' test/tmp.paf | datamash median 1)

echo "Tests passed"
