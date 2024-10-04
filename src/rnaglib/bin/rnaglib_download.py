#!/usr/bin/env python3

import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..', '..'))

import argparse

from rnaglib.utils import graph_io

"""
A downloading script
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--redundancy", default='nr', type=str,
                        help="Do we want all RNA or a non redundant subset ? Options are 'all' or 'nr'")
    parser.add_argument("-c", "--chop", default=False, action='store_true', help="Shall we chop the big rna into "
                                                                                 "smaller spatially coherent graphs")
    parser.add_argument("-a", "--annotated", default=False, action='store_true',
                        help="To include graphlets annotations in "
                             "the graphs for kernel computations.")
    parser.add_argument("-o", "--overwrite", default=False, action='store_true', help='To overwrite existing data.')
    parser.add_argument("-d", "--debug", default=False, action='store_true', help='To download the debug dataset.')
    parser.add_argument("-vs", "--version", default='1.0.0', type=str, help='Which data release to fetch.')
    parser.add_argument("-pdb", "--store_pdbs", default=False, action="store_true", help='Whether to store associated PDB files.')
    parser.add_argument("-dl", "--download_dir", default=None, type=str, help='Where to store the downloaded data.')
    args, _ = parser.parse_known_args()

    graph_io.download_graphs(redundancy=args.redundancy,
                             chop=args.chop,
                             annotated=args.annotated,
                             version=args.version,
                             overwrite=args.overwrite,
                             data_root=args.download_dir,
                             verbose=True,
                             get_pdbs=args.store_pdbs,
                             debug=args.debug)
