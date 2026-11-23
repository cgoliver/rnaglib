#!/usr/bin/env bash
# Build an RNAglib database release and package it for Zenodo.
#
# Usage: ./data_build.sh <version> [num_workers] [build_root]
#   e.g. ./data_build.sh 3.1.0 128 ~/rnaglib_build
#
# Produces in <build_root>/release-<version>/:
#   rnaglib-all-<version>.tar.gz    every RNA-containing PDB entry
#   rnaglib-nr-<version>.tar.gz     subset in the BGSU non-redundant set (4.0A)
#   rnaglib-debug-<version>.tar.gz  DEBUG_PDBIDS from data_release.py, for tests / debug=True
# Then upload with: python scripts/zenodo_upload.py <build_root>/release-<version>/*.tar.gz
# and set DATA_VERSION / ZENODO_RECORD in src/rnaglib/config/data_release.py.
set -euo pipefail

VERSION=${1:?usage: data_build.sh <version> [num_workers] [build_root]}
NW=${2:-$(nproc)}
ROOT=${3:-$HOME/rnaglib_build}
PY=${PYTHON:-python}

STRUCTS=$ROOT/structures
BUILDS=$ROOT/builds
RELEASE=$ROOT/release-$VERSION
ALL=rnaglib-all-$VERSION
NR=rnaglib-nr-$VERSION
DEBUG=rnaglib-debug-$VERSION
mkdir -p "$STRUCTS" "$BUILDS" "$RELEASE"

# 1. Update the local PDB mirror and build graphs for every RNA structure
#    (-c resumes an interrupted build).
rnaglib_prepare_data -s "$STRUCTS" -o "$BUILDS" -t "$ALL" -nw "$NW" -c

# 2. Cut the nr and debug variants out of the full build.
"$PY" - "$BUILDS" "$ALL" "$NR" "$DEBUG" <<'EOF'
import os, shutil, sys
from pathlib import Path
from rnaglib.config.data_release import DEBUG_PDBIDS
from rnaglib.utils.graph_io import get_rna_list

builds, all_tag, nr_tag, debug_tag = sys.argv[1:]
src = Path(builds) / all_tag / "graphs"
built = sorted(p.stem for p in src.glob("*.json"))
nr = set(get_rna_list(nr_only=True))
missing = sorted(set(DEBUG_PDBIDS) - set(built))
if missing:
    sys.exit(f"Debug structures missing from the build: {missing}")
for tag, ids in [(nr_tag, [i for i in built if i in nr]), (debug_tag, sorted(DEBUG_PDBIDS))]:
    dst = Path(builds) / tag / "graphs"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for i in ids:
        shutil.copy2(src / f"{i}.json", dst / f"{i}.json")
    print(f"{tag}: {len(ids)} graphs")
print(f"{all_tag}: {len(built)} graphs")
EOF

# 3. Package: archives contain <tag>/graphs/*.json, as download_graphs() expects.
for tag in "$ALL" "$NR" "$DEBUG"; do
    tar -C "$BUILDS" -czf "$RELEASE/$tag.tar.gz" "$tag/graphs"
done
ls -la "$RELEASE"
