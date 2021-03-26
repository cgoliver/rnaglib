"""
Produce filters of the data set
"""
import os
import sys
import json
import argparse
import networkx as nx
import csv
import pandas as pd
from collections import defaultdict
from rcsbsearch import TextQuery, Attr
from tqdm import tqdm

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from prepare_data.annotations import load_graph, write_graph

def listdir_fullpath(d):
        return [os.path.join(d, f) for f in os.listdir(d)]
def get_NRlist(resolution):
    """
    Get non-redudant RNA list from the BGSU website
    """

    base_url = 'http://rna.bgsu.edu/rna3dhub/nrlist/download'
    release = 'current' # can be replaced with a specific release id, e.g. 0.70
    url = '/'.join([base_url, release, resolution])

    df = pd.read_csv(url, header=None)

    repr_set = []
    for ife in df[1]:
        repr_set.append(ife)

    return repr_set

def load_csv(input_file, quiet=False):
    """
    load a csv of from rna.bgsu.edu of representative set
    :param input_file: path to csv file
    :param quiet: set to true to turn off warnings
    :return repr_set: list of equivalence class RNAs
    """
    NRlist = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                NRlist.append(row[1])
            except csv.Error as e:
                if not quiet:
                    print(f'Warning error {e} found when trying to parse row: \n {row}')

    return NRlist

def parse_NRlist(NRlist):
    """
    Parse NR BGSU csv file for a list of non-redundant RNA chains
    list can be downloaded from:
        http://rna.bgsu.edu/rna3dhub/nrlist
    :param NRlist: Set of representative RNAs output (see load_csv())
    :return: set of non-redundant RNA chains (tuples of (structure, model, chain))
    """

    NRchains = defaultdict(set)

    # split into each IFE (Integrated Functional Element)
    for representative in NRlist:
        items = representative.split('+')
        for entry in items:
            pbid, model, chain = entry.split('|')
            NRchains[pbid.lower()].add(chain)

    return NRchains


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
        pbid, chain, pos = node.split('.')
        try:
            if (chain in fltr[pbid]) \
                    or fltr[pbid] == 'all':
                NR_nodes.append(node)
        except KeyError:
            continue

    if len(NR_nodes) == 0:
        return None

    h = g.subgraph(NR_nodes).copy()

    return h

def get_NRchains(resolution):
    """
    Get a map of non redundant IFEs (integrated functional elements) from
    rna.bgsu.edu/rna3dhub/nrlist

    :param resolution: (string) one of
    [1.0A, 1.5A, 2.0A, 2.5A, 3.0A, 3.5A, 4.0A, 20.0A]
    :return NRchains: (Dictionary) keys=PDB IDs, Values=(set) Chain IDs
    """

    NR_list = get_NRlist(resolution)
    return parse_NRlist(NR_list)


def get_Ribochains():
    """
    Get a list of all PDB structures containing RNA and have the text 'ribosome'

    :return: (dictionary) keys=pbid, value='all'
    """
    q1 = Attr('rcsb_entry_info.polymer_entity_count_RNA') >= 1
    q2 = TextQuery("ribosome")

    query = q1 & q2

    results = set(query())

    # print("textquery len: ", len(set(q2())))
    # print("RNA query len: ", len(set(q1())))
    # print("intersection len: ", len(results))

    fltr = {}
    for pbid in results:
        fltr[pbid.lower()] = 'all'

    return fltr

def get_NonRibochains():
    """
    Get a list of all PDB structures containing RNA
    and do not have the text 'ribosome'

    :return: (dictionary) keys=pbid, value='all'
    """
    q1 = Attr('rcsb_entry_info.polymer_entity_count_RNA') >= 1
    q2 = TextQuery("ribosome")


    results = set(q1()).difference(set(q2()))

    # print("textquery len: ", len(set(q2())))
    # print("RNA query len: ", len(set(q1())))
    # print("intersection len: ", len(results))

    fltr = {}
    for pbid in results:
        fltr[pbid.lower()] = 'all'

    return fltr

def filter_all(graph_dir, output_dir,
        fltrs = ['NR', 'Ribo', 'NonRibo'],
        min_nodes = 20):


    for fltr in fltrs:
        fltr_set = get_fltr(fltr)
        fltr_dir = os.path.join(output_dir, fltr + '_graphs')
        try:
            os.mkdir(fltr_dir)
        except FileExistsError:
            pass
        print(f'Filtering for {fltr}')
        for graph_file in tqdm(listdir_fullpath(graph_dir)):
            g = load_graph(graph_file)
            g = filter_graph(g, fltr_set)
            if g is None: continue
            if len(g.nodes) < min_nodes: continue
            write_graph(g, os.path.join(fltr_dir, graph_file[-9:]))

def get_fltr(fltr):

    if fltr == 'NR':
        return get_NRchains('4.0A')

    if fltr == 'Ribo':
        return get_Ribochains()

    if fltr == 'NonRibo':
        return get_NonRibochains()

def main():

    filter_all('data/output', 'data')


if __name__ == '__main__':
    main()
