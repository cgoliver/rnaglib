#!/bin/bash

INSTALL_DIR=$1

# Check if $1 is empty
if [ -z "$1" ]; then
    echo "Error: Missing install path."
    exit 1
fi

# Detect OS
OS=$(uname -s)

# Set shell RC file based on shell type
if [ -n "$ZSH_VERSION" ]; then
    SHELLRC="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    SHELLRC="$HOME/.bashrc"
else
    SHELLRC="$HOME/.profile"
fi

echo "Detected OS: $OS"
echo "Using shell config: $SHELLRC"

# Load compiler only on Linux with module system
if [ "$OS" = "Linux" ] && command -v module &> /dev/null; then
    module load gcc/12
    export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
fi

# Set compiler flags based on OS
if [ "$OS" = "Darwin" ]; then
    # macOS - check for homebrew gcc
    if command -v gcc-14 &> /dev/null; then
        CC=gcc-14
        CXX=g++-14
        OPENMP_FLAGS="-fopenmp"
        USE_OPENMP=1
        echo "Using Homebrew GCC-14"
    elif command -v gcc-13 &> /dev/null; then
        CC=gcc-13
        CXX=g++-13
        OPENMP_FLAGS="-fopenmp"
        USE_OPENMP=1
        echo "Using Homebrew GCC-13"
    elif command -v gcc-12 &> /dev/null; then
        CC=gcc-12
        CXX=g++-12
        OPENMP_FLAGS="-fopenmp"
        USE_OPENMP=1
        echo "Using Homebrew GCC-12"
    else
        CC=clang
        CXX=clang++
        OPENMP_FLAGS=""
        USE_OPENMP=0
        echo "Warning: OpenMP not available. Using clang without OpenMP."
        echo "For better performance, install GCC via Homebrew: brew install gcc"
    fi
    LIB_FLAGS=""
else
    # Linux
    CC=gcc
    CXX=g++
    OPENMP_FLAGS="-fopenmp"
    USE_OPENMP=1
    LIB_FLAGS="-L/usr/lib64 -B/usr/lib64"
fi

echo "Using compiler: $CXX"

#=============================================================================
# CD-HIT Install
#=============================================================================
echo ""
echo "Installing CD-HIT..."
TEMPDIR=$(mktemp -d)
cd "$TEMPDIR"
wget https://github.com/weizhongli/cdhit/releases/download/V4.8.1/cd-hit-v4.8.1-2019-0228.tar.gz
tar -xzvf cd-hit-v4.8.1-2019-0228.tar.gz
cd cd-hit-v4.8.1-2019-0228

# Patch for macOS without OpenMP
if [ $USE_OPENMP -eq 0 ]; then
    echo "Patching CD-HIT to disable OpenMP..."
    
    # Comment out omp.h includes and OpenMP pragmas
    find . -name "*.c++" -o -name "*.h" | while read file; do
        sed -i.bak 's/#include<omp.h>/\/\/#include<omp.h>/' "$file"
        sed -i.bak 's/#include <omp.h>/\/\/#include <omp.h>/' "$file"
        sed -i.bak 's/#pragma omp/\/\/#pragma omp/' "$file"
        sed -i.bak 's/omp_set_num_threads/\/\/omp_set_num_threads/' "$file"
        sed -i.bak 's/omp_get_thread_num()/0/' "$file"
    done
    
    # Modify Makefile to remove OpenMP flags
    sed -i.bak 's/-fopenmp//' Makefile
    sed -i.bak 's/LDFLAGS =/LDFLAGS = -Wno-unused-command-line-argument/' Makefile
else
    # Modify Makefile for proper OpenMP support
    sed -i.bak "s/^CC = g++/CC = $CXX/" Makefile
fi

# Remove zlib if not available on macOS
if [ "$OS" = "Darwin" ]; then
    if ! [ -f /usr/local/include/zlib.h ] && ! [ -f /opt/homebrew/include/zlib.h ]; then
        sed -i.bak "s/-DWITH_ZLIB//" Makefile
        echo "Warning: zlib not found, building without zlib support"
    fi
fi

make CC="$CXX" || {
    echo "Error: CD-HIT compilation failed"
    cd /
    rm -rf "$TEMPDIR"
    exit 1
}

chmod u+x cd-hit
mkdir -p "${INSTALL_DIR%/}/cd_hit"
cp cd-hit "${INSTALL_DIR%/}/cd_hit"

EXPORT_CMD="export PATH=\$PATH:${INSTALL_DIR%/}/cd_hit"
grep -qxF "$EXPORT_CMD" "$SHELLRC" || echo "$EXPORT_CMD" >> "$SHELLRC"

cd /
rm -rf "$TEMPDIR"
echo "✓ CD-HIT installed successfully"

#=============================================================================
# RNAalign Install
#=============================================================================
echo ""
echo "Installing RNAalign..."
TEMPDIR=$(mktemp -d)
cd "$TEMPDIR"
wget https://zhanggroup.org/RNA-align/bin/RNAalign.tar.bz2
tar -xvjf RNAalign.tar.bz2
cd RNAalign

# Fix malloc.h issue on macOS
if [ "$OS" = "Darwin" ]; then
    # Replace malloc.h with stdlib.h in all files
    find . -name "*.h" -o -name "*.cpp" | xargs sed -i.bak 's/#include <malloc.h>/#include <stdlib.h>/'
fi

$CXX -O3 -ffast-math -o RNAalign RNAalign.cpp $LIB_FLAGS || {
    echo "Error: RNAalign compilation failed"
    cd /
    rm -rf "$TEMPDIR"
    exit 1
}

chmod u+x ./RNAalign
mkdir -p "${INSTALL_DIR%/}/RNAalign"
cp RNAalign "${INSTALL_DIR%/}/RNAalign"

EXPORT_CMD="export PATH=\$PATH:${INSTALL_DIR%/}/RNAalign"
grep -qxF "$EXPORT_CMD" "$SHELLRC" || echo "$EXPORT_CMD" >> "$SHELLRC"

cd /
rm -rf "$TEMPDIR"
echo "✓ RNAalign installed successfully"

#=============================================================================
# US-Align Install
#=============================================================================
echo ""
echo "Installing US-Align..."
TEMPDIR=$(mktemp -d)
cd "$TEMPDIR"
wget https://zhanggroup.org/US-align/bin/module/USalign.cpp

$CXX -O3 -ffast-math -o USalign USalign.cpp $LIB_FLAGS || {
    echo "Error: USalign compilation failed"
    cd /
    rm -rf "$TEMPDIR"
    exit 1
}

chmod u+x ./USalign
mkdir -p "${INSTALL_DIR%/}/USalign"
cp USalign "${INSTALL_DIR%/}/USalign"

EXPORT_CMD="export PATH=\$PATH:${INSTALL_DIR%/}/USalign"
grep -qxF "$EXPORT_CMD" "$SHELLRC" || echo "$EXPORT_CMD" >> "$SHELLRC"

cd /
rm -rf "$TEMPDIR"
echo "✓ USalign installed successfully"

#=============================================================================
# Finish
#=============================================================================
echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "The following tools have been installed to: $INSTALL_DIR"
echo "  - cd-hit"
echo "  - RNAalign"
echo "  - USalign"
echo ""
echo "PATH updated in: $SHELLRC"
echo ""
echo "To use the tools in your current shell, run:"
echo "  source $SHELLRC"
echo ""
if [ $USE_OPENMP -eq 0 ]; then
    echo "NOTE: CD-HIT was compiled without OpenMP (single-threaded)."
    echo "For better performance, install GCC via Homebrew:"
    echo "  brew install gcc"
    echo "Then re-run this script."
    echo ""
fi
