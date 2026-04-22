#!/bin/bash

# =========================================================================================================

# run like: ./test/extensive.sh <cuda|rocm> <custom|bin|build>

# =========================================================================================================
# change these

SLORADO="slorado"

SLORADO_CUDA_URL="https://unsw-my.sharepoint.com/:u:/g/personal/z5136909_ad_unsw_edu_au/IQBksuPWnfpaTbnkIC9k05I3AY_Dhe1uz_Rj5sO9JxT2n64?download=1"
SLORADO_ROCM_URL="https://unsw-my.sharepoint.com/:u:/g/personal/z5136909_ad_unsw_edu_au/IQC0nnJ4s3foSJDd8Qaw7124AX_STnHRsg0bZaZ7zSeEcgk?download=1"

BUILD_FROM_SOURCE=0 # run only if in slorado repo, for asan memory checks
RUN_500K=0 # run 500k DNA dataset for HAC

# batch sizes for each model
FAST_BATCH=512
HAC_BATCH=256
SUP_BATCH=256

# basecaller options
CHUNKSIZE=10000
READ_MEM=512M
READ_BATCH=2048

# models
FAST="dna_r10.4.1_e8.2_400bps_fast@v5.0.0"
HAC="dna_r10.4.1_e8.2_400bps_hac@v5.0.0"
SUP="dna_r10.4.1_e8.2_400bps_sup@v5.0.0"

FAST_RNA="rna004_130bps_fast@v5.1.0"
HAC_RNA="rna004_130bps_hac@v5.1.0"
SUP_RNA="rna004_130bps_sup@v5.1.0"

# mod
METH=5mCG_5hmCG@v3

# =========================================================================================================
# tools (will be automatically downloaded if not present)

TOOLS_DIR=test/tools
SAMTOOLS_VERSION=1.20
DATAMASH_VERSION=1.8
MINIMAP2_VERSION=2.24
MINIMOD_VERSION=0.5.0

export SAMTOOLS=${SAMTOOLS:-${TOOLS_DIR}/bin/samtools}
export DATAMASH=${DATAMASH:-${TOOLS_DIR}/bin/datamash}
export MINIMAP2=${MINIMAP2:-${TOOLS_DIR}/bin/minimap2}
export MINIMOD=${MINIMOD:-${TOOLS_DIR}/bin/minimod}

# =========================================================================================================
# test data - this will be automatically downloaded if not present

DATA_URL="https://unsw-my.sharepoint.com/:u:/g/personal/z5136909_ad_unsw_edu_au/IQAECzbEx9P0SIYFBFOS7whhAe6TcLp2YIZlBELlKwCpL9I?download=1"
DATA_DIR=test/slorado_test_ext_dat

REF_DNA="${DATA_DIR}/genome/hg38noAlt.idx"
REF_RNA="${DATA_DIR}/genome/gencode.v40.transcripts.fa"
REF_DNA_FA="${DATA_DIR}/genome/hg38noAlt.fa"

SUBSUBSAMPLE="${DATA_DIR}/PGXXXX230339_reads_20k.blow5"
SUBSUBSAMPLE_RNA="${DATA_DIR}/PNXRXX240011_reads_20k.blow5"

CHR22="${DATA_DIR}/PGXXXX230339_reads_chr22.blow5"
CHR22_METH_BED=test/bisulphite_chr22.tsv

SINGLE_READ="test/PGXXXX230339/reads_1.blow5"

# optional only checks if $RUN_500K = 1
SUBSAMPLE="/data/slow5-testdata/hg2_prom_lsk114_5khz_subsample/PGXXXX230339_reads_500k.blow5"

# =========================================================================================================

die() {
    echo "Error: $@" >&2
    exit 1
}

ex() {
    "$@"
}

check_accuracy() {
    case $1 in
    $FAST )
        if (( $(echo "$2 >= 0.93" | bc -l) ));
        then
            return 0
        fi
        ;;
    $HAC )
        if (( $(echo "$2 >= 0.97" | bc -l) ));
        then
            return 0
        fi
        ;;
    $SUP )
        if (( $(echo "$2 >= 0.98" | bc -l) ));
        then
            return 0
        fi
        ;;
    $FAST_RNA )
        if (( $(echo "$2 >= 0.91" | bc -l) ));
        then
            return 0
        fi
        ;;
    $HAC_RNA )
        if (( $(echo "$2 >= 0.95" | bc -l) ));
        then
            return 0
        fi
        ;;
    $SUP_RNA )
        if (( $(echo "$2 >= 0.97" | bc -l) ));
        then
            return 0
        fi
        ;;
    *)
        die "Invalid model provided"
        ;;
    esac

    die "$1 failed accuracy test with value of $2"
}

check_corr() {
    case $1 in
    $HAC )
        if (( $(echo "$2 >= 0.91" | bc -l) ));
        then
            return 0
        fi
        ;;
    $SUP )
        if (( $(echo "$2 >= 0.91" | bc -l) ));
        then
            return 0
        fi
        ;;
    *)
        die "Invalid model provided"
        ;;
    esac
    die "$1 failed mod freq correlation test with value of $2"
}

download_test_data() {
	if [ -d ${DATA_DIR} ]; then
		return
	fi

	tar_path=test/data.tgz
	wget -O $tar_path ${DATA_URL} || rm -rf $tar_path ${DATA_DIR}
	echo "Extracting. Please wait."
	tar -xf $tar_path || rm -rf $tar_path ${DATA_DIR}
	rm -f $tar_path
}

download_minimap2() {
    test -d ${TOOLS_DIR}/src || mkdir -p ${TOOLS_DIR}/src || die "Creating ${TOOLS_DIR}/src failed"
    test -d ${TOOLS_DIR}/bin || mkdir -p ${TOOLS_DIR}/bin || die "Creating ${TOOLS_DIR}/bin failed"

    tarball=${TOOLS_DIR}/src/minimap2-${MINIMAP2_VERSION}_x64-linux.tar.bz2
    src_dir=${TOOLS_DIR}/src/minimap2-${MINIMAP2_VERSION}_x64-linux

    test -e $tarball && rm -f $tarball
    test -d $src_dir && rm -rf $src_dir

    wget https://github.com/lh3/minimap2/releases/download/v${MINIMAP2_VERSION}/minimap2-${MINIMAP2_VERSION}_x64-linux.tar.bz2 -O $tarball || die "Downloading minimap2 failed"
    tar -xf $tarball -C ${TOOLS_DIR}/src || die "Extracting minimap2 failed"
    cp ${src_dir}/minimap2 ${TOOLS_DIR}/bin/minimap2 || die "Installing minimap2 failed"
    chmod +x ${TOOLS_DIR}/bin/minimap2 || die "Setting minimap2 permissions failed"
}

download_minimod() {
    test -d ${TOOLS_DIR}/src || mkdir -p ${TOOLS_DIR}/src || die "Creating ${TOOLS_DIR}/src failed"
    test -d ${TOOLS_DIR}/bin || mkdir -p ${TOOLS_DIR}/bin || die "Creating ${TOOLS_DIR}/bin failed"

    tarball=${TOOLS_DIR}/src/minimod-v${MINIMOD_VERSION}-x86_64-linux-binaries.tar.gz
    src_dir=${TOOLS_DIR}/src/minimod-v${MINIMOD_VERSION}

    test -e $tarball && rm -f $tarball
    test -d $src_dir && rm -rf $src_dir

    wget https://github.com/warp9seq/minimod/releases/download/v${MINIMOD_VERSION}/minimod-v${MINIMOD_VERSION}-x86_64-linux-binaries.tar.gz -O $tarball || die "Downloading minimod failed"
    tar -xf $tarball -C ${TOOLS_DIR}/src || die "Extracting minimod failed"
    cp ${src_dir}/minimod ${TOOLS_DIR}/bin/minimod || die "Installing minimod failed"
    chmod +x ${TOOLS_DIR}/bin/minimod || die "Setting minimod permissions failed"
}

download_samtools() {
    test -d ${TOOLS_DIR}/src || mkdir -p ${TOOLS_DIR}/src || die "Creating ${TOOLS_DIR}/src failed"
    tarball=${TOOLS_DIR}/src/samtools-${SAMTOOLS_VERSION}.tar.bz2
    src_dir=${TOOLS_DIR}/src/samtools-${SAMTOOLS_VERSION}

    test -e $tarball && rm -f $tarball
    test -d $src_dir && rm -rf $src_dir

    wget https://github.com/samtools/samtools/releases/download/${SAMTOOLS_VERSION}/samtools-${SAMTOOLS_VERSION}.tar.bz2 -O $tarball || die "Downloading samtools failed"
    tar -xf $tarball -C ${TOOLS_DIR}/src || die "Extracting samtools failed"
    (
        cd $src_dir || exit 1
        ./configure --without-curses || exit 1
        make -j || exit 1
    ) || die "Building samtools failed"

    test -d ${TOOLS_DIR}/bin || mkdir -p ${TOOLS_DIR}/bin || die "Creating ${TOOLS_DIR}/bin failed"
    cp ${src_dir}/samtools ${TOOLS_DIR}/bin/samtools || die "Installing samtools failed"
    chmod +x ${TOOLS_DIR}/bin/samtools || die "Setting samtools permissions failed"
}

download_datamash() {
    test -d ${TOOLS_DIR}/src || mkdir -p ${TOOLS_DIR}/src || die "Creating ${TOOLS_DIR}/src failed"
    tarball=${TOOLS_DIR}/src/datamash-${DATAMASH_VERSION}.tar.gz
    src_dir=${TOOLS_DIR}/src/datamash-${DATAMASH_VERSION}

    test -e $tarball && rm -f $tarball
    test -d $src_dir && rm -rf $src_dir

    wget https://ftp.gnu.org/gnu/datamash/datamash-${DATAMASH_VERSION}.tar.gz -O $tarball || die "Downloading datamash failed"
    tar -xf $tarball -C ${TOOLS_DIR}/src || die "Extracting datamash failed"
    (
        cd $src_dir || exit 1
        ./configure --prefix=$(pwd)/../../ || exit 1
        make -j || exit 1
    ) || die "Building datamash failed"

    test -d ${TOOLS_DIR}/bin || mkdir -p ${TOOLS_DIR}/bin || die "Creating ${TOOLS_DIR}/bin failed"
    if [ -x ${src_dir}/datamash ]; then
        cp ${src_dir}/datamash ${TOOLS_DIR}/bin/datamash || die "Installing datamash failed"
    elif [ -x ${src_dir}/src/datamash ]; then
        cp ${src_dir}/src/datamash ${TOOLS_DIR}/bin/datamash || die "Installing datamash failed"
    else
        die "Installing datamash failed (built binary not found)"
    fi
    chmod +x ${TOOLS_DIR}/bin/datamash || die "Setting datamash permissions failed"
}

download_model() {
    test -e $1.zip && rm $1.zip
    test -d $1 && rm -r $1
    wget https://cdn.oxfordnanoportal.com/software/analysis/dorado/${1}.zip -O $1.zip || die "Downloading the model failed"
    unzip $1.zip || die "Unzipping the model failed"
    test -d models || mkdir models || die "Creating the models directory failed"
    mv $1 models/ || die "Moving the model failed"
    rm -f $1.zip || die "Removing the model failed"
}

download_slorado_binary() {
    archive_path=test/slorado_${DEV}.binpkg
    extract_dir=test/slorado_${DEV}

    # Reuse an existing extracted binary to avoid downloading every run.
    downloaded_slorado=$(find "$extract_dir" -type f -path "*/bin/slorado" -perm -u+x 2>/dev/null | head -n1)
    if [ -n "$downloaded_slorado" ]; then
        SLORADO="$downloaded_slorado"
        return
    fi

    test -e "$archive_path" && rm -f "$archive_path"
    test -d "$extract_dir" && rm -rf "$extract_dir"
    mkdir -p "$extract_dir" || die "Creating $extract_dir failed"

    if [ "$DEV" = "cuda" ]; then
        wget -O "$archive_path" "$SLORADO_CUDA_URL" || die "Downloading slorado CUDA binary failed"
    elif [ "$DEV" = "rocm" ]; then
        wget -O "$archive_path" "$SLORADO_ROCM_URL" || die "Downloading slorado ROCm binary failed"
    else
        die "Unknown DEV option ${DEV}. Supported options are: cuda, rocm"
    fi

    if ! tar -xf "$archive_path" -C "$extract_dir"; then
        if ! unzip -q "$archive_path" -d "$extract_dir"; then
            die "Extracting slorado binary package failed"
        fi
    fi

    downloaded_slorado=$(find "$extract_dir" -type f -path "*/bin/slorado" -perm -u+x | head -n1)
    test -n "$downloaded_slorado" || die "slorado executable not found at /bin/slorado in downloaded package"
    SLORADO="$downloaded_slorado"
}

check_acc_dna() {
    $MINIMAP2 -cx map-ont $REF_DNA -t $NTHREADS tmp.fastq --secondary=no > tmp.paf || die "minimap2 failed"
    MEDIAN=$(awk '{print $10/$11}' tmp.paf | $DATAMASH median 1)
    check_accuracy $1 $MEDIAN
}

check_acc_rna() {
    $MINIMAP2 -cx splice -uf -k14 $REF_RNA -t $NTHREADS --secondary=no tmp.fastq > tmp.paf || die "minimap2 failed"
    MEDIAN=$(awk '{print $10/$11}' tmp.paf | $DATAMASH median 1)
    check_accuracy $1 $MEDIAN
}

check_corr_mod() {
    ./scripts/get_meth_freq.sh $REF_DNA_FA tmp.sam > tmp.mm.bedmethyl || die "Getting methylation frequency failed"
    corr=$(python3 scripts/corr_meth.py $CHR22_METH_BED tmp.mm.bedmethyl)
    check_corr $1 $corr
}

if [ $# -ne 2 ]; then
    die "Usage: $0 <cuda|rocm> <custom|bin|build>"
fi

DEV=$1
SLORADO_MODE=$2

if [ "$DEV" = "cuda" ]; then
    GPU_BUILD_FLAG=cuda=1
elif [ "$DEV" = "rocm" ]; then
    GPU_BUILD_FLAG=rocm=1
else
    die "Unknown DEV option ${DEV}. Supported options are: cuda, rocm"
fi

if [ "$SLORADO_MODE" = "custom" ]; then
    :
elif [ "$SLORADO_MODE" = "bin" ]; then
    download_slorado_binary
elif [ "$SLORADO_MODE" = "build" ]; then
    SLORADO="slorado"
else
    die "Unknown slorado mode ${SLORADO_MODE}. Supported modes are: custom, bin, build"
fi

echo "Using slorado executable at: $SLORADO"

# check tools
test -x $MINIMOD || download_minimod
test -x $MINIMAP2 || download_minimap2
test -x $SAMTOOLS || download_samtools
test -x $DATAMASH || download_datamash

$MINIMOD --version > /dev/null || die "minimod is missing"
$MINIMAP2 --version > /dev/null || die "minimap2 is missing"
$SAMTOOLS --version > /dev/null || die "samtools is missing"
$DATAMASH --version > /dev/null || die "datamash is missing"

# set number of threads
export NTHREADS=$(getconf _NPROCESSORS_ONLN)

if ! [[ "$NTHREADS" =~ ^[0-9]+$ ]] || [ "$NTHREADS" -le 0 ]; then
    die "NTHREADS must be a positive integer, got '$NTHREADS'"
fi

download_test_data

# check files
test -e $REF_DNA || die "missing DNA reference genome $REF_DNA"
test -e $REF_DNA_FA || die "missing DNA reference genome $REF_DNA_FA"
test -e $REF_RNA || die "missing RNA reference genome $REF_RNA"
test -e $SUBSUBSAMPLE || die "missing DNA BLOW5 subsubsample $SUBSUBSAMPLE"
test -e $SUBSUBSAMPLE_RNA || die "missing RNA BLOW5 subsubsample $SUBSUBSAMPLE_RNA"
test -e $CHR22 || die "missing chr22 BLOW5 subsubsample $CHR22"

if [ $RUN_500K -eq 1 ]; then
    test -e $SUBSAMPLE || die "missing DNA BLOW5 subsample"
fi

$SLORADO --version > /dev/null || die "slorado is missing"

# download models
test -d models/$FAST || download_model $FAST
test -d models/$HAC || download_model $HAC
test -d models/$SUP || download_model $SUP

test -d models/$FAST_RNA || download_model $FAST_RNA
test -d models/$HAC_RNA || download_model $HAC_RNA
test -d models/$SUP_RNA || download_model $SUP_RNA

test -d models/${HAC}_${METH} || download_model ${HAC}_${METH}
test -d models/${SUP}_${METH} || download_model ${SUP}_${METH}

# memory check with asan if building from source

if [ $BUILD_FROM_SOURCE -eq 1 ]; then
    make clean && make -j asan=1 cxx11_abi=1

    # basecalling

    echo "Memory Check - CPU - FAST model - 1 5khz reads"
    ex $SLORADO basecaller models/$FAST $SINGLE_READ -xcpu -c200 -K10 > test/tmp.fastq  || die "Running the tool failed"

    echo "Memory Check - CPU - FAST model - 2 batch 1 thread"
    ex $SLORADO basecaller models/$FAST $SINGLE_READ -xcpu -c200 -K5 -t1 > test/tmp.fastq  || die "Running the tool failed"

    echo "Memory Check - CPU - FAST model - incomplete batch 1 thread"
    ex $SLORADO basecaller models/$FAST $SINGLE_READ -xcpu -c200 -K6 -t1 > test/tmp.fastq  || die "Running the tool failed"

    echo "Memory Check - CPU - FAST model - 2 batch 2 thread"
    ex $SLORADO basecaller models/$FAST $SINGLE_READ -xcpu -c200 -K5 -t2 > test/tmp.fastq  || die "Running the tool failed"

    echo "Memory Check - CPU - FAST model - incomplete batch 2 thread"
    ex $SLORADO basecaller models/$FAST $SINGLE_READ -xcpu -c200 -K6 -t2 > test/tmp.fastq  || die "Running the tool failed"

    echo "Memory Check - CPU - FAST model - 2 batch 3 thread"
    ex $SLORADO basecaller models/$FAST $SINGLE_READ -xcpu -c200 -K5 -t3 > test/tmp.fastq  || die "Running the tool failed"

    echo "Memory Check - CPU - FAST model - incomplete batch 3 thread"
    ex $SLORADO basecaller models/$FAST $SINGLE_READ -xcpu -c200 -K6 -t3 > test/tmp.fastq  || die "Running the tool failed"

    # modcalling

    echo "Memory Check - CPU - HAC model - $MOD - 1 5khz reads"
    ex $SLORADO basecaller models/$HAC $SINGLE_READ --mod $METH -xcpu -c200 -K10 > test/tmp.fastq  || die "Running the tool failed"

    echo "Memory Check - CPU - HAC model - $MOD - 2 batch 2 thread"
    ex $SLORADO basecaller models/$HAC $SINGLE_READ --mod $METH -xcpu -c200 -K5 -t2 > test/tmp.fastq  || die "Running the tool failed"
fi

# GPU tests
if [ $BUILD_FROM_SOURCE -eq 1 ]; then
    echo "Using requested GPU backend: $DEV ($GPU_BUILD_FLAG)"
    make clean && make -j $GPU_BUILD_FLAG cxx11_abi=1
fi

# correlation check modified basecalling with 5mCG_5hmCG

echo "GPU - HAC meth model - chr22"
ex $SLORADO basecaller models/$HAC $CHR22 --mod $METH -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $HAC_BATCH > tmp.sam || die "Running the tool failed"
check_corr_mod $HAC
echo ""
echo "********************************************************************"

echo "GPU - SUP meth model - chr22"
ex $SLORADO basecaller models/$SUP $CHR22 --mod $METH -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $SUP_BATCH > tmp.sam || die "Running the tool failed"
check_corr_mod $SUP
echo ""
echo "********************************************************************"

# accuracy check DNA

if [ $RUN_500K -eq 1 ]; then
    echo "GPU - HAC model - 500k reads"
    ex $SLORADO basecaller models/$HAC $SUBSAMPLE -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $HAC_BATCH > tmp.fastq || die "Running the tool failed"
    check_accuracy_dna $HAC
    echo ""
    echo "********************************************************************"
fi

echo "GPU - FAST model - 20k reads"
ex $SLORADO basecaller models/$FAST $SUBSUBSAMPLE -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $FAST_BATCH > tmp.fastq || die "Running the tool failed"
check_acc_dna $FAST
echo ""
echo "********************************************************************"

echo "GPU - HAC model - 20k reads"
ex $SLORADO basecaller models/$HAC $SUBSUBSAMPLE -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $HAC_BATCH > tmp.fastq || die "Running the tool failed"
check_acc_dna $HAC
echo ""
echo "********************************************************************"

echo "GPU - SUP model - 20k reads"
ex $SLORADO basecaller models/$SUP $SUBSUBSAMPLE -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $SUP_BATCH > tmp.fastq || die "Running the tool failed"
check_acc_dna $SUP
echo ""
echo "********************************************************************"

# accuracy check RNA
echo "GPU - FAST RNA model - 20k reads"
ex $SLORADO basecaller models/$FAST_RNA $SUBSUBSAMPLE_RNA -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $FAST_BATCH > tmp.fastq || die "Running the tool failed"
check_acc_rna $FAST_RNA
echo ""
echo "********************************************************************"

echo "GPU - HAC RNA model - 20k reads"
ex $SLORADO basecaller models/$HAC_RNA $SUBSUBSAMPLE_RNA -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $HAC_BATCH > tmp.fastq || die "Running the tool failed"
check_acc_rna $HAC_RNA
echo ""
echo "********************************************************************"

echo "GPU - SUP RNA model - 20k reads"
ex $SLORADO basecaller models/$SUP_RNA $SUBSUBSAMPLE_RNA -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $SUP_BATCH > tmp.fastq || die "Running the tool failed"
check_acc_rna $SUP_RNA
echo ""
echo "********************************************************************"

# check flash support by trying to run it
FLASH_SUPPORTED=1

echo "GPU - SUP model (flash) - 20k reads"
if ex $SLORADO basecaller models/$SUP $SUBSUBSAMPLE --flash yes -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $SUP_BATCH > tmp.fastq; then
    if ! (check_acc_dna $SUP); then
        FLASH_SUPPORTED=0
    fi
else
    FLASH_SUPPORTED=0
fi
echo ""
echo "********************************************************************"

echo "GPU - SUP RNA model (flash) - 20k reads"
if ex $SLORADO basecaller models/$SUP_RNA $SUBSUBSAMPLE_RNA --flash yes -xcuda:all -t $NTHREADS -B $READ_MEM -K $READ_BATCH -c $CHUNKSIZE -C $SUP_BATCH > tmp.fastq; then
    if ! (check_acc_rna $SUP_RNA); then
        FLASH_SUPPORTED=0
    fi
else
    FLASH_SUPPORTED=0
fi
echo ""
echo "********************************************************************"

echo "all tests passed!"
if [ $FLASH_SUPPORTED -eq 0 ]; then
    echo "...but flash attention tests have failed on this system/device"
fi