#!/usr/bin/env python3

import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..', '..'))

import argparse

from rnaglib.data_loading import rna_loader
from rnaglib.data_loading.feature_maps import build_node_feature_parser

"""
A downloading script
"""

def cline():

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default='0.0.1', help="Which version of the data to process.")
    parser.add_argument("-output", default=".", type=str, help='File to dump token maps.')
    args, _ = parser.parse_known_args()
    return args

def main():
    args = cline()
    dataset = graphloader.GraphDataset(version=args.version, redundancy='NR')
    node_features = ['nt_code',
                     'dbn']
    node_features_parser = build_node_feature_parser(node_features)
    for feature in node_features:
        mapping = node_features_parser[feature].mapping
        with open(os.path.join(args.output, f"{feature}.txt"), "w") as f:
            for k, v in mapping.items():
                f.write(f"{k} {v}\n")
    with open(os.path.join(args.output, f"edge_map.txt"), "w") as f:
        for k, v in dataset.edge_map.items():
            f.write(f"{k} {v}\n")

if __name__ == "__main__":
    main(cline())
    pass
