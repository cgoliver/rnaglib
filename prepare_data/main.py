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
import argparse
from Bio.PDB.PDBList import PDBList
from rcsbsearch import TextQuery, rcsb_attributes

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

def update_RNApdb(pdir):
    """
    Download a list of RNA containing structures from the PDB
    overwrite exising files
    """
    print('Updating PDB...')
    # Get a list of PDBs containing RNA
    query = rcsb_attributes.rcsb_entry_info.polymer_entity_count_RNA >= 1
    rna = set(query())

    pl = PDBList()

    # If not fully downloaded before, download all structures
    if len(os.listdir(pdir)) < 500:
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
    parser.add_argument('-u', '--update', action='store_true',
                        help='update the structures dir')
    args = parser.parse_args()

    # args.structures_dir = '/Users/jonbroad/OneDrive - McGill University/School/McGill/Honours Project/data/structures/test_structures'
    # args.output_dir = '../examples'

    # Update PDB
    if args.update: update_RNApdb(args.structures_dir)

    cifs = listdir_fullpath(args.structures_dir)
    todo = ((cif, args.output_dir) for cif in cifs)

    pool = mp.Pool(args.num_workers)

    pool.starmap(do_one, todo)

    return

if __name__ == '__main__':
    main()

