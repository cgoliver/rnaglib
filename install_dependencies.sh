INSTALL_DIR=$1

# Check if $1 is empty
if [ -z "$1" ]; then
    echo "Error: Missing install path."
    exit 1
fi

TEMPDIR=$(mktemp -d)
BASHRC=$HOME/.bashrc

# CD-HIT Install

cd $TEMPDIR

wget https://github.com/weizhongli/cdhit/releases/download/V4.8.1/cd-hit-v4.8.1-2019-0228.tar.gz

tar -xzvf cd-hit-v4.8.1-2019-0228.tar.gz
cd cd-hit-v4.8.1-2019-0228
make
chmod u+x cd-hit
mkdir ${INSTALL_DIR%/}/cd_hit
cp cd-hit ${INSTALL_DIR%/}/cd_hit

EXPORT_CMD="export PATH=\$PATH:${INSTALL_DIR%/}/cd_hit"
echo "$EXPORT_CMD" >> "$BASHRC"

rm -rf "$TEMPDIR"

# RNAalign Install

TEMPDIR=$(mktemp -d)
cd $TEMPDIR
wget https://zhanggroup.org/RNA-align/bin/RNAalign.tar.bz2
tar -xvjf RNAalign.tar.bz2
cd RNAalign
g++ -O3 -ffast-math -o RNAalign RNAalign.cpp -static
chmod u+x ./RNAalign
mkdir ${INSTALL_DIR%/}/RNAalign
cp RNAalign $INSTALL_DIR/RNAalign
EXPORT_CMD="export PATH=\$PATH:${INSTALL_DIR%/}/RNAalign"
echo "$EXPORT_CMD" >> "$BASHRC"

rm -rf "$TEMPDIR"

source $BASHRC
