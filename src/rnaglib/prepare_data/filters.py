"""
Produce filters of the data set
"""

import os
import sys
import traceback
import json
import argparse
import networkx as nx
import csv
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import shutil

from rnaglib.utils import dump_json, load_graph
from rnaglib.utils import listdir_fullpath


def filter_graph(g, fltr):
    """
    Filter graph for redundant structures identified by BGSU
    full list of non-redundant IFE (Integrated Functional Elements) is available at
    rna.bgsu.edu/rna3dhub/nrlist

    :param g: nx graph
    :param fltr: Dictionary, keys=PDB IDs, values=(set) Chain IDs

    :return h: subgraph or None if does not exist

    """

    NR_nodes = []
    for node in g.nodes():
        pbid, chain, pos = node.split(".")
        try:
            if (chain in fltr[pbid]) or fltr[pbid] == "all":
                NR_nodes.append(node)
        except KeyError:
            continue

    if len(NR_nodes) == 0:
        return None

    h = g.subgraph(NR_nodes).copy()

    return h


# fltrs = ['NR', 'Ribo', 'NonRibo'],
def filter_all(graph_dir, output_dir, filters=["NR"], min_nodes=20):
    """Apply filters to a graph dataset.

    :param graph_dir: where to read graphs from
    :param output_dir: where to dump the graphs
    :param filters: list of which filters to apply ('NR', 'Ribo', 'NonRibo')
    :param min_nodes: skip graphs with fewer than `min_nodes` nodes (default=20)

    """

    for fltr in filters:
        fltr_set = get_fltr(fltr)
        fltr_dir = os.path.join(output_dir, fltr)
        try:
            os.mkdir(fltr_dir)
        except FileExistsError:
            pass
        print(f"Filtering for {fltr}")
        fails = 0
        for graph_file in tqdm(listdir_fullpath(graph_dir)):
            try:
                output_file = os.path.join(fltr_dir, graph_file[-9:])
                if fltr == "NR":
                    g = load_graph(graph_file)
                    g = filter_graph(g, fltr_set)
                    if g is None:
                        continue
                    if len(g.nodes) < min_nodes:
                        continue
                    dump_json(output_file, g)
                else:
                    pbid = graph_file[-9:-5]
                    if pbid in fltr_set:
                        shutil.copy(graph_file, output_file)

            except Exception as e:
                print(e)
                traceback.print_exc()
                fails += 1
                continue

    print(f"Fails: {fails}")


def get_fltr(fltr):
    """Fetch the filter object for a given filter ID.

    :param fltr: Filter ID ('NR', 'Ribo', 'NonRibo')
    """

    if fltr == "NR":
        return get_NRchains("4.0A")

    if fltr == "Ribo":
        return get_Ribochains()

    if fltr == "NonRibo":
        return get_NonRibochains()

    return get_Custom(fltr)


def main():
    filter_all("data/graphs_vernal", "data/graphs_vernal_filters")


if __name__ == "__main__":
    main()
