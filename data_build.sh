# Master script for building a data release
# This is normally called by crontab
# Usage:
# rnaglib_build <build_name> <number of workers>

# get latest rnaglib
rm -rf .venv
python -m venv .venv
source .venv/bin/activate 
cd ~/rnaglib
git pull
pip install -e ~/rnaglib/.venv/bin/activate

mkdir rnaglib_data_builds
mkdir rnaglib_data_builds/$1

# update structure DB
rnaglib_prepare -S rnaglib_data_builds/structures -O rnaglib_data_builds/$1 -u
rnaglib_prepare -S rnaglib_data_builds/structures -O rnaglib_data_builds/$1 -nw $2 
