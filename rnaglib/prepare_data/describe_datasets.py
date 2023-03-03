import os
import networkx as nx
import sys
import argparse
import csv
from tqdm import tqdm

from rnaglib.utils import load_graph
from rnaglib.utils import listdir_fullpath

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir',
                        help='input_directory containing graphs and complement')
    parser.add_argument('output',
                        help='csv output file')

    args = parser.parse_args()

    stats = {}

    stats['Graphs'] = len(os.listdir(args.input_dir))

    # Compute number of nodes
    stats['Nodes'] = 0
    stats['Edges'] = 0
    stats['Protein Binding'] = 0
    stats['Small-Mol. Binding'] = 0
    stats['Ion Binding'] = 0
    for graph_file in tqdm(listdir_fullpath(args.input_dir)):
        g = load_graph(graph_file)
        stats['Nodes'] += len(g.nodes)
        stats['Edges'] += len(g.edges)
        stats['Protein Binding'] += len([n for n, d in g.nodes.data()\
                                        if d['binding_protein'] is not None])
        stats['Small-Mol. Binding'] += len([n for n, d in g.nodes.data()\
                                        if d['binding_small-molecule'] is not None])
        stats['Ion Binding'] += len([n for n, d in g.nodes.data()\
                                        if d['binding_ion'] is not None])

    stats['Avg Nodes'] = int(stats['Nodes']/stats['Graphs'])
    stats['Avg Edges'] = int(stats['Edges']/stats['Graphs'])

    if os.path.exists(args.output): header = False
    else: header = True

    name = args.input_dir.split('/')[-1]

    with open(args.output, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        # header
        if header: writer.writerow(['Dataset'] + list(stats.keys()))

        writer.writerow([name] + list(stats.values()))




if __name__ == '__main__':
    main()
