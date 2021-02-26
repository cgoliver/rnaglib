"""
Prepare graphs from PDB cif files using DSSR annotation software.
Ligand and Ion annotations are manually annotated by neighbor search on PDB structures.
Write graphs in JSON format to output directory
"""
import multiprocessing as mp
import os
import sys
import argparse
from Bio.PDB.PDBList import update_pdb

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from prepare_data.dssr_2_graphs import build_one
from prepare_data.interfaces import get_interfaces
from prepare_data.annotations import parse_interfaces, annotate_graph, write_graph

def listdir_fullpath(d):
        return [os.path.join(d, f) for f in os.listdir(d)]

def do_one(cif, output_dir):

    if '.cif' not in cif: return
    pbid = cif[-8:-4]

    # Build graph with DSSR
    g = build_one(cif)

    # Find ligand and ion annotations from the PDB cif
    interfaces, _ = get_interfaces(cif)
    annotations = parse_interfaces(interfaces)
    h = annotate_graph(g, annotations)

    # Write graph to outputdir in JSON format
    write_graph(h, os.path.join(output_dir, pbid+'.json'))

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--structures_dir',
                        help='directory containing structures from the PDB')
    parser.add_argument('-O', '--output_dir',
                        help='directory to output graphs')
    parser.add_argument('-nw', '--num_workers',
                        type=int,
                        help='number of workers for multiprocessing',
                        default = 1)
    args = parser.parse_args()

    # args.structures_dir = '../data/structures/test_structures/'
    # args.output_dir = '../data/graphs/DSSR/annotated/'
    # g, _ = load_graph('../examples/1aju.json')

    # Update PDB
    pl = PDBList(pdb=args.structures_dir)
    pl.update_pdb()

    cifs = listdir_fullpath(args.structures_dir)
    todo = ((cif, args.output_dir) for cif in cifs)

    pool = mp.Pool(args.num_workers)

    pool.starmap(do_one, todo)

    return

if __name__ == '__main__':
    main()

