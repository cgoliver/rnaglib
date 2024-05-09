#!/usr/bin/env python3

import os
import sys
import json
script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..', '..'))

import argparse
from rnaglib.data_loading import get_graph_indexes

def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnaglib_root", default=f"{os.path.expanduser('~')}/.rnaglib")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--redundancy", default="nr")

    return parser.parse_args()


def main():
    args = cline()
    index = get_graph_indexes(f"{args.rnaglib_root}/datasets/rnaglib-{args.redundancy}-{args.version}/graphs",
                              dump_name=f'graph_index_{args.redundancy}_{args.version}.json',
                              possible_supervisions=
                              {'node_binding_small-molecule', 'node_binding_protein',
                               'node_binding_ion', "node_is_modified"}
                              )

    try:
        os.mkdir(f"{args.rnaglib_root}/indexes")
    except FileExistsError:
        pass

    with open(f"{args.rnaglib_root}/indexes/rnaglib-{args.redundancy}-{args.version}.json", "w") as js:
        json.dump(index, js)
    pass
