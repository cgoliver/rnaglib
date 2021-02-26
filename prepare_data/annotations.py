import json
import networkx as nx
import sys
import os
import csv
from collections import defaultdict

script_dir = os.path.join(os.path.realpath(__file__), '..')
sys.path.append(os.path.join(script_dir, '..'))


def load_graph(json_file):
    """
    load DSSR graph from JSON
    """
    pbid = json_file[-9:-5]
    with open(json_file, 'r') as f:
        d = json.load(f)

    g = nx.readwrite.json_graph.node_link_graph(d)

    return g, pbid

def write_graph(g, json_file):
    """
    write graph to JSON
    """
    d = nx.readwrite.json_graph.node_link_data(g)
    with open(json_file, 'w') as f:
        json.dump(d, f)

    return

def annotate_graph(g, annots, labels):
    """
    Add node annotations to graph from annots
    nodes without a value recieve None type
    """

    for node in g.nodes():
        for label, typ in labels.items():
            try:
                annot = annots[node][typ]
            except KeyError:
                annot = None
            g.nodes[node][label] = annot

    return g



def load_csv_annot(csv_file, pbids=None, types=None):
    """
    Get annotations from csv file, parse into a dictionary
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
    add annotations from csv_file to all graphs in graph_dir
    """
    labels = {'binding_ion': 'ion',
            'binding_small-molecule': 'ligand'}
    annotations = load_csv_annot(csv_file)

    for graph in os.listdir(graph_dir):
        path = os.path.join(graph_dir, graph)
        g, pbid = load_graph(path)
        h = annotate_graph(g, annotations, labels)
        write_graph(h, os.path.join(output_dir, graph))

def main():
    annotate_graphs('../examples/',
                    '../data/interface_list_1aju.csv',
                    '../data/graphs/DSSR/annotated')

if __name__ == '__main__':
    main()

