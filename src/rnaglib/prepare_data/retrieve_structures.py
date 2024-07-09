"""
Script to retrieve only PBID structures containing non-redundant chains
Alternatively all RNA containing chains can be downloaded from the PDB website at
https://www.rcsb.org/
"""

import argparse
from Bio.PDB import *
import os
import sys
import csv

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

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


    pdbl = PDBList()
    # Download non redundant PDBs
    if 'nrlist' in args.input_file.lower():
        for line in repr_set:
            pbid = line[:4]
            pdbl.retrieve_pdb_file(pbid, pdir=args.output_dir)
    # Download from a comma seperate list text file output from the pdb
    else:
        with open(args.input_file, 'r') as f:
            pbids = f.readline().split(',')
        for pbid in pbids:
            pdbl.retrieve_pdb_file(pbid, pdir=args.output_dir)




if __name__ == '__main__':
    main()
