import argparse
from Bio.PDB import *
import os
import sys
import csv

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

def slice_structures(repr_set):
    """
    Slice RNA structures in output_dir so that they only include chains in the repr_set
    :param repr_set: Set of representative RNAs output (see load_csv())
    :param output_dir: directory containing PDB cif files (already downloaded)
    :return:
    """
    parser = MMCIFParser(QUIET = True)

    for line in repr_set:
        pbid = line[:4]

        # Parse structure
        # TODO: error handling and download structure if it does not exist
        # TODO: Can speed up by grouping repr_set by pbid
        structure = parser.get_structure(pbid, os.path.join(args.output_dir, f'{pbid}.cif'))

        # split into each IFE (Integrated Functional Element)
        items = line.split('+')
        for entry in items:
            pbid2, model, chain = entry.split('|')
            try:
                repr_chain = structure[int(model) - 1][chain]
            except KeyError:
                if not quiet:
                    print("WARNING: Chain not found: \n", entry, "\n")
                    continue
        # TODO: Make a new structure containing the chains found
    return

def load_csv(input_file, quiet=False):
    """
    load a csv of from rna.bgsu.edu of representative set
    :param input_file: path to csv file
    :param quiet: set to true to turn off warnings
    :return repr_set: list of equivalence class RNAs
    """
    repr_set = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                repr_set.append(row[1])
            except csv.Error as e:
                if not quiet:
                    print(f'Warning error {e} found when trying to parse row: \n {row}')

    return repr_set

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        help = 'csv of RNA representative set \
                                can be downloaded at rna.bgsu.edu')
    parser.add_argument('output_dir',
                        help = 'directory to store output structures')
    args = parser.parse_args()

    repr_set = load_csv(args.input_file)

    # Download PDB files
    pdbl = PDBList()
    for line in repr_set:
        pbid = line[:4]
        pdbl.retrieve_pdb_file(pbid, pdir=args.output_dir)


    # Slice the structures into only the representative chains
    # slice_structures(repr_set)

if __name__ == '__main__':
    main()
