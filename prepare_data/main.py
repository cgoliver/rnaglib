"""
Module for detecting RNA-protein, RNA-RNA and RNA-smallMolecule interface residues.
Interface residues decided on Euclidean distance cutoff.
"""

import sys
import argparse
import os
import numpy as np
from Bio.PDB import *
import json

scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(scriptdir, '..'))

from prepare_data.interfaces import *
from prepare_data.slice import *

def write_interfaces(output_file, interface_residues):
    """
    Write interface residues found to a csv with format:
        pbid, position, chain, interaction type
    i.e x12t, 12      , Q    ,  protein

    :param output_file:
    :param interface_residues: list of tuples
    :return:
    """
    with open(output_file, 'w') as f:
        # Header:
        f.write('pbid,chain_pos,chain,type,binding_molecule,PDB_pos\n')
        # Data
        for line in interface_residues:
            f.write(f'{line[0]},{line[1]},{line[2]},{line[3]},{line[4]},{line[5]}\n')

def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--pbd_dir',
                        help='Input directory of PBD structures (format=mmCIF)',
                        default= os.path.join(scriptdir,'..','data','structures'))
    parser.add_argument('-interface_list_output',
                        help='Output CSV file of interface residue list',
                        default = os.path.join(scriptdir, '..', 'data',
                            'interface_residue_list.csv'))
    parser.add_argument( '-c', '--cutoff',
                        help='Cutoff (in angstroms) of the distance \
                                between interacting residues',
                                default=10)
    parser.add_argument('-l', '--ligands',
                        help='list of ligands',
                        default= os.path.join(scriptdir, '..', 'data', 'ligand_list.txt'))
    parser.add_argument('-native_graphs_dir',
                        help='directory containing native RNA graphs',
                        default = os.path.join(scriptdir, '..', 'data', 'graphs', 'native'))
    parser.add_argument('-interface_list_input',
                        help='CSV file of list of interface residues\
                                call with this option if prepare_data/main has\
                                been called before to speed up execution')
    parser.add_argument('interface_graphs_dir',
                        help='output directory to store interface graphs')
    parser.add_argument('-t','--type',
                        help='RNA interface interaction partner\
                                can be any of {protein, ion, rna, ligand, all}\
                                (input multiple options in space seperated string)',
                        default='all')
    args = parser.parse_args()

    args = parser.parse_args()
    interaction_type = args.type.split(' ')
    cutoff = int(args.cutoff)
    files_not_found=[]

    if args.interface_list_input == None:
        # Get ligands list
        ligands = []
        with open(args.ligands, 'r') as f:
            for line in f.readlines():
                ligands.append(line.strip())

        # initialize empty residue lists
        interface_residues = []
        files_not_found = []

        # find interfaces
        i = 0
        for cif_file in os.listdir(args.pbd_dir):
            # DEBUGGING: Loop control
            # if i == 30: break
            # i+=1
            path = os.path.join(args.pbd_dir, cif_file)
            try:
                residues, _ = get_interfaces(path, ligands = ligands, cutoff = cutoff)
            except TypeError:
                files_not_found.append(path)
            interface_residues = interface_residues + residues

        # Write interfaces to csv and parse csv
        write_interfaces(args.interface_list_output, interface_residues)
        interface_list_input = args.interface_list_output

    else:
        interface_list_input = args.interface_list_input

    # Parse interfaces into dictionaries
    subset = parse_subset_fromcsv(interface_list_input, interaction_type)

    # with open(os.path.join(scriptdir, '..', 'junk', 'subset.json'), 'w') as f:
        # json.dump(subset, f, indent=4)

    # slice graphs from interface residues list 
    slice_all(args.native_graphs_dir, args.interface_graphs_dir, subset)

    print("DONE")

    if len(files_not_found) > 0:
        print('\n\n', 'NOTE: \t The following files were not found:\n')
        for line in files_not_found:
            print(line)

if __name__ == "__main__":
    main()



