"""
This script is used to find all possible annotations present in dssr and dump them in a dict.

This is useful to handcraft encoding functions
"""
import os
import sys

import json
from tqdm import tqdm
from collections import defaultdict

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from utils.graph_io import load_json


def dict_one(graph):
    """
    Turn a graph dictionnaries into one dict key : set
    :param graph:
    :return:
    """

    def process_graph_dict(graph_dict, prepend=None):
        """
        Edges or node data gets extracted for each and merged into a clean dict {key:set}
        optionnaly preprend a string to the keys
        :param graph_dict:
        :return:
        """
        return_dict = defaultdict(set)
        for node, node_data in graph_dict.items():
            for key, value in node_data.items():
                if prepend is not None:
                    key = f'{prepend}_{key}'
                if type(value) == list:
                    value = tuple(value)
                    return_dict[key].add(value)
                if type(value) == dict:
                    # Just one does that : its 'node frame'
                    # print(type(value))
                    # print(key)
                    # print(value)
                    for inner_key, inner_value in value.items():
                        if type(inner_value) == list:
                            inner_value = tuple(inner_value)
                        return_dict[f'{key}_{inner_key}'].add(inner_value)
                else:
                    return_dict[key].add(value)
        return return_dict

    # Nx returns an Node/Edge view object, a list filled with (node, data)/(source, target, data) tuples
    list_nodes = graph.nodes(data=True)
    dict_nodes = {u: data for u, data in list_nodes}
    list_edges = graph.edges(data=True)
    dict_edges = {(u, v): data for u, v, data in list_edges}

    # print(dict_nodes)
    # print(dict_edges)

    node_key_dict = process_graph_dict(dict_nodes, prepend='node')
    edge_key_dict = process_graph_dict(dict_edges, prepend='edge')
    node_key_dict.update(edge_key_dict)
    return node_key_dict


def get_all_annots(graph_dir):
    dict_all = defaultdict(set)
    i = 0
    for graph_name in tqdm(sorted(os.listdir(graph_dir))):
        i += 1
        graph = os.path.join(graph_dir, graph_name)
        graph = load_json(graph)
        graph_dict = dict_one(graph)
        for key, value in graph_dict.items():
            dict_all[key] = dict_all[key].union(value)
        # if i > 3: break
    dict_all = {key: list(value) for key, value in dict_all.items()}
    json.dump(dict_all, open('all_annots.json', 'w'))


if __name__ == '__main__':
    graph_path = 'data/graphs/all_graphs'
    get_all_annots(graph_path)
