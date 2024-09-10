"""
Prepare graphs from PDB cif files using DSSR annotation software.
Ligand and Ion annotations are manually annotated by neighbor search
on PDB structures.
Write graphs in JSON format to output directory
Run with -u to update the PDB atleast once a week
"""

import argparse
import os
from joblib import Parallel, delayed

from tqdm import tqdm

from rnaglib.utils import update_RNApdb

from rnaglib.prepare_data import cif_to_graph
from rnaglib.prepare_data import chop_all
from rnaglib.prepare_data import annotate_all


def dir_setup(args):
    """Create all necessary folders"""

    build_dir = os.path.join(args.output_dir, args.tag)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(os.path.join(build_dir, "graphs"), exist_ok=True)
    if args.annotate:
        os.makedirs(os.path.join(build_dir, "chops"), exist_ok=True)
        os.makedirs(os.path.join(build_dir, "annot"), exist_ok=True)
    elif args.chop:
        os.makedirs(os.path.join(build_dir, "chops"), exist_ok=True)
    else:
        pass

    return build_dir


def cline():
    parser = argparse.ArgumentParser()
    # Input/Output Directories
    parser.add_argument(
        "-s",
        "--structures_dir",
        required=True,
        help="directory containing RNA structures from the PDB",
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="directory to output graphs"
    )
    # For just one output
    parser.add_argument(
        "--one_mmcif",
        help="If one wants to do only one structure, path to the mmcif file",
    )
    # Optional arguments
    parser.add_argument(
        "--rna-source",
        type=str,
        default="rcsb",
        help='Source of RNA structures. If "rcsb" RNAs taken from up to date list of\
                              RNA-containing PDBs in RCSB. If "local" take all cifs in --structures_dir',
        choices=["rcsb", "local"],
    )
    parser.add_argument(
        "-nw",
        "--num-workers",
        type=int,
        help="number of workers for multiprocessing",
        default=1,
    )
    parser.add_argument(
        "-u", "--update", action="store_true", help="update the structures dir"
    )
    parser.add_argument(
        "-t", "--tag", default="default", help="Version tag to assign to the dataset."
    )
    parser.add_argument(
        "-a",
        "--annotate",
        action="store_true",
        default=False,
        help="Whether to build graphlet annotations.",
    )
    parser.add_argument(
        "-ch",
        "--chop",
        action="store_true",
        default=False,
        help="Whether to build chopped graphs.",
    )
    parser.add_argument(
        "-c",
        "--continu",
        action="store_true",
        help="Continue previously paused execution",
    )
    parser.add_argument(
        "-nr",
        "--nr",
        "--non-redundant",
        action="store_true",
        default=False,
        help="If true, build non-redundant dataset",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="runs only on --n-debug structures for debug.",
    )
    parser.add_argument(
        "--n-debug", type=int, default=10, help="set number of debug structures."
    )
    return parser.parse_args()


def prepare_data_main():
    """Master function for building an annotated RNA dataset.
    Results in a folder with the following sub-directories: 'graphs', 'chops', annots'.
    The last two are only present if args.annotate are set. The 'graphs' folder
    contains JSON objects which can be loaded by networkx into graphs.
    """

    args = cline()

    if args.one_mmcif is not None:
        cif_to_graph(cif=args.one_mmcif, output_dir=args.output_dir)
        return
    else:
        build_dir = dir_setup(args)
        graphs_dir = os.path.join(build_dir, "graphs")

    # Update PDB and get Todo list
    # list of rnas to process
    if args.rna_source == "rcsb":
        print(">>> Updating local PDB mirror")
        rna_list = update_RNApdb(args.structures_dir, nr_only=args.nr)
    if args.rna_source == "local":
        print(f">>> Using structures in {args.structures_dir}")
        rna_list = [f.split(".")[0] for f in os.listdir(args.structures_dir)]

    done = []
    if args.continu:
        done = set([os.path.splitext(g)[0] for g in os.listdir(graphs_dir)])

    todo = [
        (os.path.join(args.structures_dir, pdbid + ".cif"), build_dir)
        for pdbid in rna_list
        if pdbid not in done
    ]
    if args.debug:
        print(">>> Using debug mode. Preparing only 10 structures.")
        todo = [item for i, item in enumerate(todo) if i < args.n_debug]

    # Build Graphs
    total = len(todo)
    print(f">>> Processing {total} RNAs.")
    errors = Parallel(n_jobs=args.num_workers)(
        delayed(cif_to_graph)(*t)
        for t in tqdm(todo, total=total, desc="Building RNA graphs.")
    )
    with open(os.path.join(build_dir, "errors.csv"), "w") as err:
        for pdbid, error in errors:
            err.write(f"{pdbid},{error}\n")

    chop_dir = os.path.join(build_dir, "chops")
    annot_dir = os.path.join(build_dir, "annot")
    if args.annotate:
        print(">>> Chopping")
        chop_all(graphs_dir, chop_dir, n_jobs=args.num_workers)
        print(">>> Annotating")
        annotate_all(graph_path=chop_dir, dump_path=annot_dir)
        print("Done annotating graphs")
    elif args.chop:
        print(">>> Chopping")
        chop_all(graphs_dir, chop_dir, n_jobs=args.num_workers)
    else:
        pass


if __name__ == "__main__":
    prepare_data_main()
