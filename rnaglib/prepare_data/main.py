"""
Prepare graphs from PDB cif files using DSSR annotation software.
Ligand and Ion annotations are manually annotated by neighbor search on PDB structures.
Write graphs in JSON format to output directory
Run with -u to update the PDB atleast once a week
EXTERNAL PACKAGES:
    rcsbsearch : `pip install rcsbsearch`
"""
import multiprocessing as mp
import os
import sys
import traceback
import argparse
from Bio.PDB.PDBList import PDBList
# from rcsbsearch import TextQuery, Attr
import json
from collections import defaultdict

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..', '..'))

from rnaglib.prepare_data.dssr_2_graphs import build_one
from rnaglib.prepare_data.interfaces import get_interfaces
from rnaglib.prepare_data.annotations import *
from rnaglib.prepare_data.filters import filter_all
from rnaglib.prepare_data.filters import has_no_dots
from rnaglib.prepare_data.filters import filter_dot_edges
from rnaglib.prepare_data.chopper import chop_all
from rnaglib.prepare_data.khop_annotate import annotate_all

FILTERS = ['NR']


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def update_RNApdb(pdir):
    """
    Download a list of RNA containing structures from the PDB
    overwrite exising files

    :param pdbdir: path containing downloaded PDBs
    """
    print('Updating PDB...')
    # Get a list of PDBs containing RNA
    query = Attr('rcsb_entry_info.polymer_entity_count_RNA') >= 1
    rna = set(query())

    pl = PDBList()

    # If not fully downloaded before, download all structures
    if len(os.listdir(pdir)) < 2000:
        pl.download_pdb_files(rna, pdir=pdir, overwrite=True)
        return rna
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

    return new_rna


def cif_to_graph(cif):
    """Build DDSR graphs for one mmCIF. Requires x3dna-dssr to be in PATH.

    :param cif: path to CIF

    :return: networkx graph of structure.
    """

    if '.cif' not in cif:
        print("Incorrect format")
        return

    # Build graph with DSSR
    error_type = ''
    try:
        g = build_one(cif)
        filter_dot_edges(g)
        assert has_no_dots(g)

    except Exception as e:
        print("ERROR: Could not construct DSSR graph for ", cif)
        print(traceback.print_exc())
        return
    else:
        # Find ligand and ion annotations from the PDB cif
        try:
            interfaces, _ = get_interfaces(cif, cutoff=5)
            annotations = parse_interfaces(interfaces)
            g = annotate_graph(g, annotations)
        except Exception as e:
            print('ERROR: Could not compute interfaces for ', cif)
            print(e)
            print(traceback.print_exc())
            error_type = 'interfaces_error'

        if error_type in ['interfaces_error', 'OK']:
            # Order the nodes
            g = reorder_nodes(g)

    return g


def do_one(cif, output_dir, min_nodes=20):
    """Build DDSR graphs for one mmCIF.

    :param cif: path to CIF
    :param output_dir: where to dump
    :param min_nodes: smallest RNA (number of residue nodes)
    """

    if '.cif' not in cif: return
    pdbid = cif[-8:-4]
    error_type = 'OK'

    # Build graph with DSSR
    print('Computing Graph for ', pdbid)
    try:
        g = build_one(cif)
        filter_dot_edges(g)
        assert has_no_dots(g)

    except Exception as e:
        print("ERROR: Could not construct DSSR graph for ", pdbid)
        print(traceback.print_exc())
        error_type = 'DSSR_error'
    else:
        if g is None:
            print(f'Excluding {pdbid} from output')
            error_type = 'noBasePairs'
            return pdbid, error_type
        if len(g.nodes()) < min_nodes:
            print(f'Excluding {pdbid} from output, less than 20 nodes')
            error_type = 'tooSmall'
            return pdbid, error_type
        if len(g.edges()) < len(g.nodes()) - 3:
            print(f'Excluding {pdbid} from output, edges < nodes -3')
            error_type = 'edges<nodes-3'
            return pdbid, error_type

        # Find ligand and ion annotations from the PDB cif
        try:
            interfaces, _ = get_interfaces(cif, cutoff=5)
            annotations = parse_interfaces(interfaces)
            g = annotate_graph(g, annotations)
        except Exception as e:
            print('ERROR: Could not compute interfaces for ', pdbid)
            print(e)
            print(traceback.print_exc())
            error_type = 'interfaces_error'

        if error_type in ['interfaces_error', 'OK']:
            # Order the nodes
            g = reorder_nodes(g)

            # Write graph to outputdir in JSON format
            write_graph(g, os.path.join(output_dir, 'all_graphs', pdbid + '.json'))
            print('>>> SUCCESS: graph written: ', pdbid)

    return pdbid, error_type


def prepare_data_main():
    parser = argparse.ArgumentParser()
    # Input/Output Directories
    parser.add_argument('-S', '--structures_dir',
                        help='directory containing structures from the PDB')
    parser.add_argument('-O', '--output_dir',
                        help='directory to output graphs')
    # Optional Flags
    parser.add_argument('-nw', '--num_workers',
                        type=int,
                        help='number of workers for multiprocessing',
                        default=1)
    parser.add_argument('-u', '--update', action='store_true',
                        help='update the structures dir')
    parser.add_argument('-c', '--continu', action='store_true',
                        help='Continue previously paused execution')
    parser.add_argument('-f', '--filter', action='store_true',
                        help='build filtered datasets')
    args = parser.parse_args()

    try:
        os.mkdir(os.path.join(args.output_dir))
        os.mkdir(os.path.join(args.output_dir, 'all_graphs'))
    except FileExistsError:
        pass

    # Update PDB and get Todo list
    cifs = listdir_fullpath(args.structures_dir)
    if args.update:
        new_rna = update_RNApdb(args.structures_dir)
        todo = ((cif, args.output_dir) for cif in cifs \
                if cif[-8:-4].upper() in new_rna)
    elif args.continu:
        done = [graph[:4] for graph in os.listdir(args.output_dir)]
        todo = ((cif, args.output_dir, fltr) for cif in cifs \
                if cif[-8:-4] not in done)
    else:
        todo = ((cif, args.output_dir) for cif in cifs)

    # Build Graphs
    pool = mp.Pool(args.num_workers)
    errors = pool.starmap(do_one, todo)

    # Filters
    if args.filter:
        filter_all(os.path.join(args.output_dir, 'all_graphs'),
                   args.output_dir,
                   filters=FILTERS
                   )

    for filter in FILTERS + ['all_graphs', 'NR']:
        filter_dest = os.path.join(args.output_dir, filter)
        chop_all(graph_path=filter_dest,
                 pdb_path=args.structures_dir,
                 dest=filter_dest + "_chops"
                 )

        print('Done producing graphs')

        annotate_all(graph_path=filter_dest + "_chops", dump_path=filter_dest + "_annot")

        print('Done annotating graphs')

    # Error Logging
    errors = [e for e in errors if e is not None]
    if len(errors) == 0:
        print("DONE\nNo Errors found")
        return

    errors_dict = defaultdict(list)
    for pbid, err in errors: errors_dict[err].append(pbid)
    with open('log.json', 'w') as f:
        json.dump(errors_dict, f, indent=2)

    print("DONE\nErrors found: (recorded in log.json)")
    print(errors_dict)

    return


if __name__ == '__main__':
    prepare_data_main()
