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
from pathlib import Path

from tqdm import tqdm

from rnaglib.utils import update_RNApdb
from rnaglib.utils import dump_json
from rnaglib.transforms import Compose
from rnaglib.transforms import CifMetadata
from rnaglib.transforms import SmallMoleculeBindingTransform
from rnaglib.transforms import SecondaryStructureTransform
from rnaglib.transforms import RBPTransform
from rnaglib.prepare_data import fr3d_to_graph
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
        default="./structures",
        help="directory containing RNA structures from the PDB",
    )
    parser.add_argument("-o", "--output_dir", default="./graphs", help="directory to output graphs")
    # For just one output
    parser.add_argument(
        "--one_mmcif",
        default=None,
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
    parser.add_argument("-u", "--update", default=True, action="store_true", help="update the structures dir")
    parser.add_argument("-t", "--tag", default="default", help="Version tag to assign to the dataset.")
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
        default=False,
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
        default=False,
        help="runs only on --n-debug structures for debug.",
    )
    parser.add_argument("--n-debug", type=int, default=10, help="set number of debug structures.")
    return parser.parse_args()


def build_graph_from_cif(cif_path, dump_dir):
    """Takes a cif file and builds the full RNAglib graph.

    :param cif_path: path to source mmCif file
    :param dump_dir: where to save the resulting networkx graph. If None just returns graph.

    :return: graph if `dump_dir` is None, else return path to saved graph.
    """
    structures_dir = Path(cif_path).parent
    graph = fr3d_to_graph(cif_path)
    if graph is None:
        return None

    transforms = [
        CifMetadata(structures_dir=structures_dir),
        SmallMoleculeBindingTransform(structures_dir=structures_dir),
        RBPTransform(structures_dir=structures_dir, protein_number_annotations=True, distances=[4.0, 6.0, 8.0]),
        SecondaryStructureTransform(structures_dir=structures_dir),
    ]

    T = Compose(transforms)

    rna_dict = {"rna": graph}
    rna_dict = T(rna_dict)
    if dump_dir is None:
        return rna_dict
    else:
        save_path = Path(dump_dir) / f"{graph.graph['pdbid']}.json"
        dump_json(save_path, rna_dict["rna"])
        return save_path


def prepare_data_main(args):
    """Master function for building an annotated RNA dataset.
    Results in a folder with the following sub-directories: 'graphs', 'chops', annots'.
    The last two are only present if args.annotate are set. The 'graphs' folder
    contains JSON objects which can be loaded by networkx into graphs.
    """

    if args.one_mmcif is not None:
        cif_to_graph(cif=args.one_mmcif, output_dir=args.output_dir)
        return
    else:
        build_dir = dir_setup(args)
        graphs_dir = os.path.join(build_dir, "graphs")

    # Update PDB and get Todo list
    # list of rnas to process
    Path(args.structures_dir).mkdir(parents=True, exist_ok=True)
    if args.rna_source == "rcsb":
        print(f">>> Updating local PDB mirror in {args.structures_dir}")
        rna_list = update_RNApdb(args.structures_dir, nr_only=args.nr, debug=args.debug)
    if args.rna_source == "local":
        print(f">>> Using structures in {args.structures_dir}")
        rna_list = [f.split(".")[0] for f in os.listdir(args.structures_dir)]

    done = []
    if args.continu:
        done = set([os.path.splitext(g)[0] for g in os.listdir(graphs_dir)])

    todo = [Path(args.structures_dir) / f"{pdbid}.cif" for pdbid in rna_list if pdbid not in done]
    if args.debug:
        print(">>> Using debug mode. Preparing only 10 structures.")
        todo = todo[: int(args.n_debug)]

    # Build Graphs
    total = len(todo)
    print(f">>> Processing {total} RNAs.")
    job = Parallel(n_jobs=args.num_workers)(
        delayed(build_graph_from_cif)(t, graphs_dir) for t in tqdm(todo, total=total, desc="Building RNA graphs.")
    )

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
    args = cline()
    prepare_data_main(args)
