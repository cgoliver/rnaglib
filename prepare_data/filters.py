import os
import sys
import json
import networkx as nx

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

def get_NRlist(save):
    """
    Get non-redudant RNA list from the BGSU website
    """
    pass

def parse_NRlist(NR_csv):
    """
    Parse NR BGSU csv file for a list of non-redundant RNA chains
    list can be downloaded from:
        http://rna.bgsu.edu/rna3dhub/nrlist
    :param repr_set: Set of representative RNAs output (see load_csv())
    :return: set of non-redundant RNA chains (tuples of (structure, model, chain))
    """

    nonRedundantStrings = load_csv(NR_csv)
    nonRedundantChains = []

    # split into each IFE (Integrated Functional Element)
    for representative in nonRedundantStrings:
        items = representative.split('+')
        for entry in items:
            pbid, model, chain = entry.split('|')
            nonRedundantChains.append((pbid, model, chain))

    return set(nonRedundantChains)


def NR_filter(g):
    """
    Filter graph for redundant structures identified by BGSU
    full list of non-redundant IFE (Integrated Functional Elements) is available at
    rna.bgsu.edu/rna3dhub/nrlist
    """
    NR_csv = get_NRlist()

    NRlist = parse_NRList(NR_csv)

    NR_nodes = []
    for node in g.nodes():
        if node in NR_list: NR_nodes.append(nodes)

    h = g.subgraph(NR_nodes).copy()

    return h

