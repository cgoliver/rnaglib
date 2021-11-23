"""
Package installs:
conda install -c salilab dssp
"""

import json
import networkx as nx
import sys
import os
import csv
from collections import defaultdict
from Bio.PDB.DSSP import DSSP
from Bio.PDB import MMCIFParser

from time import perf_counter

script_dir = os.path.join(os.path.realpath(__file__), '..')
sys.path.append(os.path.join(script_dir, '..'))


def dangle_trim(G):
    """
    Recursively remove dangling nodes from graph.

    :param G: Networkx graph
    :type G: networkx.Graph
    :return: Trimmed networkx graph
    :rtype: networkx.Graph
    """
    dangles = lambda G: [n for n in G.nodes() if G.degree(n) < 2]
    while dangles(G):
        G.remove_nodes_from(dangles(G))


def reorder_nodes(g):
    """
    Reorder nodes in graph

    :param g: Pass a graph for node reordering. 
    :type g: networkx.DiGraph

    :return h: (nx DiGraph)
    """

    h = nx.DiGraph()
    h.add_nodes_from(sorted(g.nodes.data()))
    h.add_edges_from(g.edges.data())
    for key, value in g.graph.items():
        h.graph[key] = value

    return h


def annotate_proteinSSE(g, structure, pdb_file):
    """
    Annotate protein_binding node attributes with the relative SSE
    if available from DSSP

    :param g: (nx graph)
    :param structure: (PDB structure)

    :return g: (nx graph)
    """

    model = structure[0]
    tic = perf_counter()
    dssp = DSSP(model, pdb_file, dssp='mkdssp', file_type='DSSP')
    toc = perf_counter()

    print(dssp.keys())

    a_key = list(dssp.keys())[2]

    print(dssp[a_key])

    print(f'runtime = {tic - toc:0.7f} seconds')

    return g


def load_graph(json_file):
    """
    load DSSR graph from JSON

    :param json_file: path to json containing DSSR output

    :return: graph from parsed json data
    :rtype: networkx.DiGraph
    """
    pbid = json_file[-9:-5]
    with open(json_file, 'r') as f:
        d = json.load(f)

    g = nx.readwrite.json_graph.node_link_graph(d)

    return g


def write_graph(g, json_file):
    """
    Utility function to write networkx graph to JSON

    :param g: graph to dump
    :type g: networkx.Graph
    :param json_file: path to dump json
    :type json_file: str
    """
    d = nx.readwrite.json_graph.node_link_data(g)
    with open(json_file, 'w') as f:
        json.dump(d, f)

    return


def annotate_graph(g, annots):
    """
    Add node annotations to graph from annots
    nodes without a value receive None type

    :param g: RNA graph to add x3dna data annotations to.
    :type g: networkx.DiGraph
    :param annots: parsed output from x3dna
    :type annots: dict
    :return: graph with updated node and edge data
    :rtype: networkx.Graph
    """

    labels = {'binding_ion': 'ion',
              'binding_small-molecule': 'ligand'}

    for node in g.nodes():
        for label, typ in labels.items():
            try:
                annot = annots[node][typ]
            except KeyError:
                annot = None
            g.nodes[node][label] = annot

    return g


def parse_interfaces(interfaces,
                     types=['ion', 'ligand']):
    """
    Parse output from get_interfaces into a dictionary

    :param interfaces: output from dssr interface annotation
    :param types: which type of molecule to consider in the interface

    :return: dictionary containing interface annotations
    """
    annotations = defaultdict(dict)

    for pbid, chain, typ, target, PDB_pos in interfaces:
        if types:
            if typ not in types: continue
        annotations[str(pbid) + '.' + str(chain) + '.' + str(PDB_pos)][typ] = target

    return annotations


def load_csv_annot(csv_file, pbids=None, types=None):
    """
    Get annotations from csv file, parse into a dictionary

    :param csv_file: csv to read annotations from
    :type csv_file: path-like
    :param pdbids: list of PDBIs to process, if None, all are processed.
    :type pdbids: list
    :param types: only consider annotations for give molecule types ('ion', 'ligand')
    :type types: list

    :return: annotation dictionary
    """
    annotations = defaultdict(dict)
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = True
        for pbid, _, chain, typ, target, PDB_pos in reader:
            if header:
                header = False
                continue
            if pbids:
                if pbid not in pbids: continue
            if types:
                if typ not in types: continue
            annotations[pbid + '.' + chain + '.' + PDB_pos][typ] = target

    return annotations


def annotate_graphs(graph_dir, csv_file, output_dir,
                    ):
    """
    Add annotations from csv_file to all graphs in graph_dir

    :param graph_dir: where to read RNA graphs from
    :type graph_dir: path-like
    :param csv_file: csv containing annotations
    :type graph_dir: path-like
    :param output_dir: where to dump the annotated graphs
    :type output_dir: path-like
    """
    annotations = load_csv_annot(csv_file)

    for graph in os.listdir(graph_dir):
        path = os.path.join(graph_dir, graph)
        g, pbid = load_graph(path)
        h = annotate_graph(g, annotations)
        write_graph(h, os.path.join(output_dir, graph))


def main():
    # annotate_graphs('../examples/',
    # '../data/interface_list_1aju.csv',
    # '../data/graphs/DSSR/annotated')
    g = load_graph('../examples/2du5.json')
    # pdb_file = '../data/structures/4gkk.cif'
    # parser = MMCIFParser()
    # structure = parser.get_structure('4GKK', pdb_file)

    # annotate_proteinSSE(g, structure, '../data/structures/4gkk.dssp')

    h = reorder_nodes(g)

    print('after reordered:\n', '\n'.join(h.nodes()))
    print(h.nodes.data())

    return


if __name__ == '__main__':
    main()
