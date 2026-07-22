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

download_model dna_r10.4.1_e8.2_400bps_fast@v5.2.0
download_model dna_r10.4.1_e8.2_400bps_hac@v5.2.0
download_model dna_r10.4.1_e8.2_400bps_sup@v5.2.0

download_model dna_r10.4.1_e8.2_400bps_fast@v5.0.0
download_model dna_r10.4.1_e8.2_400bps_hac@v5.0.0
download_model dna_r10.4.1_e8.2_400bps_sup@v5.0.0

# v6.0.0: only HAC exists for DNA (FLSTM architecture)
download_model dna_r10.4.1_e8.2_400bps_hac@v6.0.0

# download_model dna_r10.4.1_e8.2_400bps_fast@v5.0.0_5mCG_5hmCG@v3
download_model dna_r10.4.1_e8.2_400bps_hac@v5.0.0_5mCG_5hmCG@v3
download_model dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mCG_5hmCG@v3

# latest DNA 6mA (all-context) modification models -- conv_lstm_v3 (chunked)
#   hac: pairs with hac@v6.0.0 (latest DNA base). sup: latest is v5.2.0 (no v6 sup DNA base exists),
#   so its v5.2.0 base is fetched here too.
download_model dna_r10.4.1_e8.2_400bps_hac@v6.0.0_6mA@v1
download_model dna_r10.4.1_e8.2_400bps_sup@v5.2.0_6mA@v1

# v6.0.0 RNA models (note: no "130bps" in name)
download_model rna004_fast@v6.0.0
download_model rna004_hac@v6.0.0
download_model rna004_sup@v6.0.0

# latest RNA m6A (DRACH-context) modification models -- conv_lstm_v3 (chunked), pair with v6.0.0 bases
download_model rna004_hac@v6.0.0_m6A_DRACH@v1
download_model rna004_sup@v6.0.0_m6A_DRACH@v1
