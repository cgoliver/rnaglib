"""Various wrappers for external tools"""

import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from rnaglib.utils import cif_remove_residues


def US_align_wrapper(
    cif_path_1: str | os.PathLike,
    cif_path_2: str | os.PathLike,
    flags: tuple = ("-mm", "1", "-ter", "1"),
    reslist_1: list | None = None,
    reslist_2: list | None = None,
    file_type: str = "cif",
):
    """Calls USalign on two mmCIF files and returns the output.

    Must have USalign (https://zhanggroup.org/US-align/bin/module/USalign.cpp) in your executable path.
    """
    assert Path(cif_path_1).exists(), f"{cif_path_1} missing"
    assert Path(cif_path_2).exists(), f"{cif_path_2} missing"

    with tempfile.TemporaryDirectory() as tmpdir:
        if reslist_1 is not None:
            new_cif_1 = Path(tmpdir) / "rna_1." + str(file_type)
            cif_remove_residues(cif_path_1, reslist_1, new_cif_1, file_type)
            cif_path_1 = new_cif_1
        if reslist_2 is not None:
            new_cif_2 = Path(tmpdir) / "rna_2." + str(file_type)
            cif_remove_residues(cif_path_2, reslist_2, new_cif_2, file_type)
            cif_path_2 = new_cif_2

        command = [
            "USalign",
            # "/home/vincent/bin/USAlign/USalign",
            *flags,
            cif_path_1,
            cif_path_2,
        ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    # print(result.stdout)
    # uncomment above for debugging
    # Regular expression to find all TM-scores
    tm_scores = re.findall(r"TM-score=\s*([\d.]+)", result.stdout)

    # Convert to float if needed
    tm_scores = [float(score) for score in tm_scores]

    try:
        return max(tm_scores)
    except ValueError:
        return None


def rna_align_wrapper(
    cif_path_1: str | os.PathLike,
    cif_path_2: str | os.PathLike,
    flags: tuple = ("-a", "T"),
    reslist_1: list | None = None,
    reslist_2: list | None = None,
):
    """Calls RNAalign on two mmCIF files and returns the output.

    Must have RNAalign (https://zhanggroup.org/RNA-align/download.html) in your executable path.
    """
    # assert shutil.which('RNAalign') is not None,\
    # "RNAalign installation not found. Go here https://zhanggroup.org/RNA-align/"

    assert Path(cif_path_1).exists(), f"{cif_path_1} missing"
    assert Path(cif_path_2).exists(), f"{cif_path_2} missing"

    with tempfile.TemporaryDirectory() as tmpdir:
        if reslist_1 is not None:
            new_cif_1 = Path(tmpdir) / "rna_1.cif"
            cif_remove_residues(cif_path_1, reslist_1, new_cif_1)
            cif_path_1 = new_cif_1
        if reslist_2 is not None:
            new_cif_2 = Path(tmpdir) / "rna_1.cif"
            cif_remove_residues(cif_path_2, reslist_2, new_cif_2)
            cif_path_2 = new_cif_2

        command = [
            "RNAalign",
            *flags,
            cif_path_1,
            cif_path_2,
        ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if "-a" in flags:
        pattern = r"TM-score=\s*([\d.]+)\s*\(if normalized by average length of two structures"
        match = re.search(pattern, result.stdout)

        if match:
            tm = float(match.group(1))
            return tm
        print(result.stderr)
        print(result.stdout)
        return None


def locarna_wrapper(seq_1: str, seq_2: str):
    """Calls LocaRNA on two RNA sequences to perform sequence-2d-structure aligment"""
    with tempfile.TemporaryDirectory as tdir:
        seq_1_file = Path(tdir) / "seq_1.fa"
        seq_2_file = Path(tdir) / "seq_2.fa"

        with open(seq_1_file, "w") as s1:
            seq_1_file.write(f"> s1\n {seq_1}")

        with open(seq_2_file, "w") as s2:
            seq_2_file.write(f"> s2\n {seq_2}")

        command = ["locarna", seq_1_file, seq_2_file]
        result = subprocess.run(command, capture_output=True, text=True, check=False)


def cdhit_wrapper(ids, sequences, sim_thresh=0.6, n_jobs=1):
    """Cluster sequences using CD-hit. Adapted from ProteinShake.

    Choose of word size:
    -n 5 for thresholds 0.7 ~ 1.0
    -n 4 for thresholds 0.6 ~ 0.7
    -n 3 for thresholds 0.5 ~ 0.6
    -n 2 for thresholds 0.4 ~ 0.5

    Parameters
    -----------
    sequences: list
        List of protein sequences to cluster.

    Returns:
    --------
    representatives: list
        List of sequence indices to preserve as representatives.
    """
    assert sim_thresh >= 0.4 and sim_thresh <= 1, "Similarity threshold not in [0.4, 1]"

    if sim_thresh >= 0.4 and sim_thresh < 0.5:
        word_size = 2
    elif sim_thresh >= 0.5 and sim_thresh < 0.6:
        word_size = 3
    elif sim_thresh >= 0.6 and sim_thresh < 0.7:
        word_size = 4
    else:
        word_size = 5

    """
    assert shutil.which('cd-hit') is not None,\
    "CD-HIT installation not found. Go here https://github.com/weizhongli/cdhit to install"
    """

    n_jobs = max(n_jobs, 0)

    with tempfile.TemporaryDirectory() as tmpdir:
        in_file = Path(tmpdir) / "in.fasta"
        out_file = Path(tmpdir) / "out.fasta"
        with open(in_file, "w") as inp:
            for id, s in zip(ids, sequences, strict=False):
                inp.write(f">{id}\n")
                inp.write(s + "\n")
        cmd = [
            "cd-hit",
            # "/home/vincent/bin/cd_hit/cd-hit",
            "-c",
            str(sim_thresh),
            "-i",
            in_file,
            "-n",
            str(word_size),
            "-l",
            "5",
            "-o",
            out_file,
            "-T",
            str(n_jobs),
            "-M",
            "0",  # unlimited memory
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)

        # parse cluster assignments
        pdb2cluster = {}
        cluster2pdb = defaultdict(list)
        with open(str(out_file) + ".clstr") as out:
            for line in out:
                if line.startswith(">"):
                    clust_id = int(line.split()[1])
                    continue
                pdb_id = line.split(">")[1].split(".")[0]
                pdb2cluster[pdb_id] = clust_id
                cluster2pdb[clust_id].append(pdb_id)
        return pdb2cluster, cluster2pdb
