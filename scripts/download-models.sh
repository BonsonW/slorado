#!/bin/bash

# MIT License
# Copyright (c) 2019 Hasindu Gamaarachchi (hasindu@unsw.edu.au)
# Copyright (c) 2023 Bonson Wong (bonson.ym@gmail.com)

die () {
    echo "$@" >&2
    exit 1
}

download_model () {
    [ -d models/$1 ] && echo "$1 already downloaded" && return 0
    wget https://cdn.oxfordnanoportal.com/software/analysis/dorado/$1.zip -O $1.zip || die "Downloading the model failed"
    unzip $1.zip || die "Unzipping the model failed"
    test -d models || mkdir models || die "Creating the models directory failed"
    mv $1 models/ || die "Moving the model failed"
    rm -f $1.zip || die "Removing the model failed"
}

download_model dna_r10.4.1_e8.2_400bps_fast@v4.2.0
download_model dna_r10.4.1_e8.2_400bps_hac@v4.2.0
download_model dna_r10.4.1_e8.2_400bps_sup@v4.2.0