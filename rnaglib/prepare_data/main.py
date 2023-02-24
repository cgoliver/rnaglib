"""
Prepare graphs from PDB cif files using DSSR annotation software.
Ligand and Ion annotations are manually annotated by neighbor search on PDB structures.
Write graphs in JSON format to output directory
Run with -u to update the PDB atleast once a week
"""
import argparse
import os
import sys
import traceback
from joblib import Parallel, delayed
import json
import requests

from Bio.PDB.PDBList import PDBList
from collections import defaultdict


from Bio.PDB.PDBList import PDBList
from tqdm import tqdm

from rnaglib.utils import reorder_nodes
from rnaglib.utils import dump_json

from rnaglib.prepare_data.dssr_2_graphs import build_one
from rnaglib.prepare_data.annotations import add_graph_annotations
from rnaglib.prepare_data.filters import filter_dot_edges, filter_all
from rnaglib.prepare_data.chopper import chop_all
from rnaglib.prepare_data.khop_annotate import annotate_all


def get_rna_list(nr_only=False):
    """
    Fetch a list of PDBs containing RNA from RCSB API.

    """
    payload = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_RNA", "operator": "greater", "value": 0}
        },
        "request_options": {
            "results_verbosity": "compact",
            "return_all_hits": True
        },
        "return_type": "entry"
    }

    r = requests.get(f'https://search.rcsb.org/rcsbsearch/v2/query?json={json.dumps(payload)}')
    try:
        response_dict = json.loads(r.text)
        ids = response_dict['result_set']
        if nr_only:
            nr_chains = getNRchains("4.0A")
            ids = [pdbid for pdbid in ids if pdbid in nr_chains]
    except:
        print('An error occured when querying RCSB.')
        print(r.text)
        exit()
    return ids


def update_RNApdb(pdir, nr_only=True):
    """
    Download a list of RNA containing structures from the PDB
    overwrite exising files

    :param pdbdir: path containing downloaded PDBs

    :returns rna: list of PDBIDs that were fetched.
    """
    print(f'Updating PDB mirror in {pdir}')
    # Get a list of PDBs containing RNA
    rna = set(get_rna_list(nr_only=nr_only))

    pl = PDBList()

    # If not fully downloaded before, download all structures
    if len(os.listdir(pdir)) < 2000:
        pl.download_pdb_files(rna, pdir=pdir, overwrite=True)
    else:
        added, mod, obsolete = pl.get_recent_changes()
        # Download new and modded entries
        new_rna = rna.intersection(set(added).union(set(mod)))
        pl.download_pdb_files(new_rna, pdir=pdir, overwrite=True)

        # Remove Obsolete entries
        obsolete_dir = os.path.join(pdir, 'obsolete')
        if not os.path.exists(obsolete_dir):
            os.mkdir(obsolete_dir)
        for cif in os.listdir(pdir):
            if cif[-8:-4].upper() in set(obsolete):
                os.rename(os.path.join(pdir, cif), os.path.join(obsolete_dir, cif))

    return rna


def cif_to_graph(cif, output_dir=None, min_nodes=20, return_graph=False):
    """
    Build DDSR graphs for one mmCIF. Requires x3dna-dssr to be in PATH.

    :param cif: path to CIF
    :param output_dir: where to dump
    :param min_nodes: smallest RNA (number of residue nodes)
    :param return_graph: Boolean to include the graph in the output
    :return: networkx graph of structure.
    """

    if '.cif' not in cif:
        # print("Incorrect format")
        return os.path.basename(cif), 'format'
    pdbid = cif[-8:-4]
    # print('Computing Graph for ', pdbid)

    # Build graph with DSSR
    error_type = 'OK'
    try:
        dssr_failed = False
        g = build_one(cif)
        dssr_failed = g is None
        filter_dot_edges(g)
    except Exception as e:
        # print("ERROR: Could not construct DSSR graph for ", cif)
        if dssr_failed:
            # print("Annotation using x3dna-dssr failed, please ensure you have the executable in your PATH")
            # print("This requires a license.")
            error_type = 'DSSR_error'
        else:
            # print(traceback.print_exc())
            error_type = 'Filtering error after DSSR building'
        return pdbid, error_type

    if len(g.nodes()) < min_nodes:
        # print(f'Excluding {pdbid} from output, less than 20 nodes')
        error_type = 'tooSmall'
        return pdbid, error_type
    if len(g.edges()) < len(g.nodes()) - 3:
        # print(f'Excluding {pdbid} from output, edges < nodes -3')
        error_type = 'edges<nodes-3'
        return pdbid, error_type

    # Find ligand and ion annotations from the PDB cif
    try:
        add_graph_annotations(g=g, cif=cif)
    except Exception as e:
        # print('ERROR: Could not compute interfaces for ', cif)
        # print(e)
        # print(traceback.print_exc())
        error_type = 'interfaces_error'
    # Order the nodes
    g = reorder_nodes(g)

    # Write graph to outputdir in JSON format
    if output_dir is not None:
        dump_json(os.path.join(output_dir, 'graphs', pdbid + '.json'), g)
    if return_graph:
        return pdbid, error_type, g
    return pdbid, error_type


def dir_setup(args):
    """ Create all necessary folders"""

    build_dir = os.path.join(args.output_dir, args.tag)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(os.path.join(build_dir, 'graphs'), exist_ok=True)
    if args.annotate:
        os.makedirs(os.path.join(build_dir, 'chops'), exist_ok=True)
        os.makedirs(os.path.join(build_dir, 'annot'), exist_ok=True)
    elif args.chop:
        os.makedirs(os.path.join(build_dir, 'chops'), exist_ok=True)
    else:
        pass
    
    return build_dir

def cline():
    parser = argparse.ArgumentParser()
    # Input/Output Directories
    parser.add_argument('-s',
                        '--structures_dir',
                        required=True,
                        help='directory containing RNA structures from the PDB')
    parser.add_argument('-o',
                        '--output_dir',
                        required=True,
                        help='directory to output graphs')
    # For just one output
    parser.add_argument('--one_mmcif',
                        help='If one wants to do only one structure, path to the mmcif file')
    # Optional arguments
    parser.add_argument('--rna-source',
                        type=str,
                        default='rcsb',
                        help='Source of RNA structures. If "rcsb" RNAs taken from up to date list of\
                              RNA-containing PDBs in RCSB. If "local" take all cifs in --structures_dir',
                        choices=['rcsb', 'local'])
    parser.add_argument('-nw', '--num_workers',
                        type=int,
                        help='number of workers for multiprocessing',
                        default=1)
    parser.add_argument('-u', '--update', action='store_true',
                        help='update the structures dir')
    parser.add_argument('-t',
                        '--tag',
                        default='default',
                        help='Version tag to assign to the dataset.')
    parser.add_argument('-a', '--annotate', action='store_true', default=False,
                        help='Whether to build graphlet annotations.')
    parser.add_argument('-ch', '--chop', action='store_true', default=False,
                        help='Whether to build chopped graphs.')
    parser.add_argument('-c', '--continu', action='store_true',
                        help='Continue previously paused execution')
    parser.add_argument('-nr', "--nr", '--non-redundant', action='store_true', default=False,
                        help='If true, build non-redundant dataset')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='runs only on 10 structures for debug.')
    return parser.parse_args()


def prepare_data_main():
    """ Master function for building an annotated RNA dataset.
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
        graphs_dir = os.path.join(build_dir, 'graphs')

    # Update PDB and get Todo list
    # list of rnas to process
    if args.rna_source == 'rcsb':
        print(">>> Updating local PDB mirror")
        rna_list = update_RNApdb(args.structures_dir, nr_only=args.nr)
    if args.rna_source == 'local':
        print(f">>> Using structures in {args.structures_dir}")
        rna_list = [f.split(".")[0] for f in os.listdir(args.structures_dir)]

    done = []
    if args.continu:
        done = set([os.path.splitext(g)[0] for g in os.listdir(graphs_dir)])

    todo = [(os.path.join(args.structures_dir, pdbid + ".cif"), build_dir)\
            for pdbid in rna_list if pdbid not in done]
    if args.debug:
        print(">>> Using debug mode. Preparing only 10 structures.")
        todo = [item for i, item in enumerate(todo) if i < 10]

    # Build Graphs
    total = len(todo)
    errors = Parallel(n_jobs=args.num_workers)(delayed(cif_to_graph)(*t) for t in tqdm(todo, total=total, desc='Building RNA graphs.'))
    with open(os.path.join(args.build_dir, "errors.csv"), 'w') as err:
        for pdbid, error in errors:
            err.write(f"{pdbid},{error}\n")

    chop_dir = os.path.join(build_dir, "chops")
    annot_dir = os.path.join(build_dir, "annot")
    if args.annotate:
        print(">>> Chopping")
        chop_all(graphs_dir, chop_dir, n_jobs=args.num_workers)
        print(">>> Annotating")
        annotate_all(graph_path=chop_dir, dump_path=annot_dir)
        print('Done annotating graphs')
    elif args.chop:
        print(">>> Chopping")
        chop_all(graphs_dir, chop_dir, n_jobs=args.num_workers)
    else:
        pass

if __name__ == '__main__':
    prepare_data_main()
