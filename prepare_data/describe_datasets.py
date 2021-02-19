import os
import networkx as nx
import sys
import argparse
import csv


script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from prepare_data.slice import get_num_nodes
def total_num_edges(directory):
    """
    get the total number of nodes for graphs in a directory
    """
    total = 0
    for graph_file in os.listdir(directory):
        if '.nx' not in graph_file: continue
        try:
            g = nx.read_gpickle(os.path.join(directory, graph_file))
        except EOFError:
            print("Warning graph file may be empty", directory, graph_file)
            continue
        total += len(g.edges)

    return total

def total_num_nodes(directory):
    """
    get the total number of nodes for graphs in a directory
    """
    total = 0
    for graph_file in os.listdir(directory):
        if '.nx' not in graph_file: continue
        try:
            g = nx.read_gpickle(os.path.join(directory, graph_file))
        except EOFError:
            print("Warning graph file may be empty", directory, graph_file)
            continue
        total += get_num_nodes(g)

    return total


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir',
                        help='input_directory containing graphs and complement')
    parser.add_argument('output',
                        help='csv output file')

    args = parser.parse_args()

    stats = {}

    for directory in os.listdir(args.input_dir):
        stats[directory] = {'graphs': None,
                            'nodes' : None,
                            'edges' : None,
                            'avg_nodes': None,
                            'avg_edges': None}
        stats[directory + ' complement'] =  {   'graphs': None,
                                                'nodes' : None,
                                                'edges' : None,
                                                'avg_nodes': None,
                                                'avg_edges': None}


    for key in stats.keys():
        if 'complement' in key:
            directory = os.path.join(args.input_dir, key.split(' ')[0], 'complement')
        else:
            directory = os.path.join(args.input_dir, key)

        # Count the number of graphs
        stats[key]['graphs'] = len([g for g in os.listdir(directory) if '.nx' in g])

        # Count the number of nodes
        stats[key]['nodes'] = total_num_nodes(directory)
        stats[key]['avg_nodes'] = round (stats[key]['nodes'] / stats[key]['graphs'], 1)

        # Count the number of edges
        stats[key]['edges'] = total_num_edges(directory)
        stats[key]['avg_edges'] = round(stats[key]['edges'] / stats[key]['graphs'], 1)

    with open(args.output, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        # header
        writer.writerow(['Dataset', 'Graphs', 'Edges', 'Nodes', 'Avg. Nodes', 'Avg. Edges'])
        for key, value in stats.items():
            writer.writerow([key] + list(value.values()))




if __name__ == '__main__':
    main()
