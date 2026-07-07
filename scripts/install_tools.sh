#!/bin/bash
# Build/install the thirdparty CLI tools used by the test + bench scripts into a shared TOOLS_DIR
# (built from source, like the rest of the repo's tooling). Both test/extensive.sh and
# scripts/bench/bench.sh use this so the tools are defined + built in exactly one place.
#
#   Source it   (. install_tools.sh)      -> get $MINIMAP2/$DATAMASH/$SAMTOOLS/$MINIMOD + install_tools()
#   Run it      (./install_tools.sh [tool...]) -> install the given tools (default: all)
#
# install_tools [tool ...]  ensures each named tool (default: minimap2 datamash samtools minimod)
# exists, building it if missing. Override any tool by exporting its var (e.g. MINIMAP2=/usr/bin/minimap2).

_ITS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_ITS_DIR}/.." && pwd)}"   # scripts/ -> repo root
TOOLS_DIR="${TOOLS_DIR:-${REPO_ROOT}/test/tools}"

SAMTOOLS_VERSION="${SAMTOOLS_VERSION:-1.20}"
DATAMASH_VERSION="${DATAMASH_VERSION:-1.8}"
MINIMAP2_VERSION="${MINIMAP2_VERSION:-2.24}"
MINIMOD_VERSION="${MINIMOD_VERSION:-0.5.0}"

export SAMTOOLS="${SAMTOOLS:-${TOOLS_DIR}/bin/samtools}"
export DATAMASH="${DATAMASH:-${TOOLS_DIR}/bin/datamash}"
export MINIMAP2="${MINIMAP2:-${TOOLS_DIR}/bin/minimap2}"
export MINIMOD="${MINIMOD:-${TOOLS_DIR}/bin/minimod}"

NTHREADS="${NTHREADS:-$(getconf _NPROCESSORS_ONLN)}"

die() { echo "Error: $*" >&2; exit 1; }

download_minimap2() {
    test -d ${TOOLS_DIR}/src || mkdir -p ${TOOLS_DIR}/src || die "Creating ${TOOLS_DIR}/src failed"
    test -d ${TOOLS_DIR}/bin || mkdir -p ${TOOLS_DIR}/bin || die "Creating ${TOOLS_DIR}/bin failed"

    tarball=${TOOLS_DIR}/src/minimap2-${MINIMAP2_VERSION}.tar.bz2
    src_dir=${TOOLS_DIR}/src/minimap2-${MINIMAP2_VERSION}

    test -e $tarball && rm -f $tarball
    test -d $src_dir && rm -rf $src_dir

    wget https://github.com/lh3/minimap2/releases/download/v${MINIMAP2_VERSION}/minimap2-${MINIMAP2_VERSION}.tar.bz2 -O $tarball || die "Downloading minimap2 failed"
    tar -xf $tarball -C ${TOOLS_DIR}/src || die "Extracting minimap2 failed"
    (
        cd $src_dir || exit 1
        arch=$(uname -m)
        case "$arch" in
            aarch64|arm64) make -j $NTHREADS arm_neon=1 aarch64=1 || exit 1 ;;
            armv7l|armv8l) make -j $NTHREADS arm_neon=1 || exit 1 ;;
            *)             make -j $NTHREADS || exit 1 ;;
        esac
    ) || die "Building minimap2 failed"
    cp ${src_dir}/minimap2 ${TOOLS_DIR}/bin/minimap2 || die "Installing minimap2 failed"
    chmod +x ${TOOLS_DIR}/bin/minimap2 || die "Setting minimap2 permissions failed"
}

download_minimod() {
    test -d ${TOOLS_DIR}/src || mkdir -p ${TOOLS_DIR}/src || die "Creating ${TOOLS_DIR}/src failed"
    test -d ${TOOLS_DIR}/bin || mkdir -p ${TOOLS_DIR}/bin || die "Creating ${TOOLS_DIR}/bin failed"

    tarball=${TOOLS_DIR}/src/minimod-v${MINIMOD_VERSION}-release.tar.gz
    src_dir=${TOOLS_DIR}/src/minimod-v${MINIMOD_VERSION}

    test -e $tarball && rm -f $tarball
    test -d $src_dir && rm -rf $src_dir

    wget https://github.com/warp9seq/minimod/releases/download/v${MINIMOD_VERSION}/minimod-v${MINIMOD_VERSION}-release.tar.gz -O $tarball || die "Downloading minimod failed"
    tar -xf $tarball -C ${TOOLS_DIR}/src || die "Extracting minimod failed"
    (
        cd $src_dir || exit 1
        scripts/install-hts.sh || exit 1
        make -j $NTHREADS || exit 1
    ) || die "Building minimod failed"
    cp ${src_dir}/minimod ${TOOLS_DIR}/bin/minimod || die "Installing minimod failed"
    chmod +x ${TOOLS_DIR}/bin/minimod || die "Setting minimod permissions failed"
}

download_samtools() {
    if test -x ${TOOLS_DIR}/bin/samtools; then export SAMTOOLS=${TOOLS_DIR}/bin/samtools; return; fi

    test -d ${TOOLS_DIR}/src || mkdir -p ${TOOLS_DIR}/src || die "Creating ${TOOLS_DIR}/src failed"
    tarball=${TOOLS_DIR}/src/samtools-${SAMTOOLS_VERSION}.tar.bz2
    src_dir=${TOOLS_DIR}/src/samtools-${SAMTOOLS_VERSION}

    test -e $tarball && rm -f $tarball
    test -d $src_dir && rm -rf $src_dir

    wget https://github.com/samtools/samtools/releases/download/${SAMTOOLS_VERSION}/samtools-${SAMTOOLS_VERSION}.tar.bz2 -O $tarball || die "Downloading samtools failed"
    tar -xf $tarball -C ${TOOLS_DIR}/src || die "Extracting samtools failed"
    (
        cd $src_dir || exit 1
        ./configure --without-curses --disable-bz2 --disable-lzma --disable-libcurl --disable-plugins || exit 1
        make -j $NTHREADS || exit 1
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
        make -j $NTHREADS || exit 1
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

# ensure each named tool exists (build if missing). default: all.
install_tools() {
    local tools=("$@"); [ "$#" -eq 0 ] && tools=(minimap2 datamash samtools minimod)
    local t
    for t in "${tools[@]}"; do
        case "$t" in
            minimap2) test -x "$MINIMAP2" || download_minimap2 ;;
            datamash) test -x "$DATAMASH" || download_datamash ;;
            samtools) test -x "$SAMTOOLS" || download_samtools ;;
            minimod)  test -x "$MINIMOD"  || download_minimod  ;;
            *) die "unknown tool: $t (known: minimap2 datamash samtools minimod)" ;;
        esac
    done
}

# when executed directly (not sourced), install the requested tools then report
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    install_tools "$@"
    echo "tools ready in ${TOOLS_DIR}/bin"
fi
