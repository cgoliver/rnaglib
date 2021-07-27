"""
This script is used to find all possible annotations present in dssr and dump them in a dict.

This is useful to handcraft encoding functions
"""
import os
import sys

import json
import pickle
from tqdm import tqdm
from collections import defaultdict, Counter

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from utils.graph_io import load_json


def dict_one(graph, counter=False):
    """
    Turn a graph dictionnaries into one dict key : set
    :param graph:
    :return:
    """

    def process_graph_dict(graph_dict, prepend=None, counter=False):
        """
        Edges or node data gets extracted for each and merged into a clean dict {key:set}
        optionnaly preprend a string to the keys
        :param graph_dict:
        :return:
        """
        if counter:
            return_dict = defaultdict(lambda: defaultdict(lambda: 0))
        else:
            return_dict = defaultdict(set)
        for node, node_data in graph_dict.items():
            for key, value in node_data.items():
                if prepend is not None:
                    key = f'{prepend}_{key}'
                if counter:
                    # Only count interesting strings
                    if type(value) == str and key not in ['node_nt_id', 'edge_nt1', 'edge_nt2']:
                        return_dict[key][value] += 1
                else:
                    if type(value) == list:
                        value = tuple(value)
                        return_dict[key].add(value)
                    if type(value) == dict:
                        # Just one does that : its 'node frame'
                        # print(type(value))
                        # print(key)
                        # print(value)
                        if value is not 'node_frame':
                            pass
                        for inner_key, inner_value in value.items():
                            if type(inner_value) == list:
                                inner_value = tuple(inner_value)
                            return_dict[f'{key}_{inner_key}'].add(inner_value)
                    elif type(value) in [str, int, float, list, bool] or value is None:
                        pass
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

    node_key_dict = process_graph_dict(dict_nodes, prepend='node', counter=counter)
    edge_key_dict = process_graph_dict(dict_edges, prepend='edge', counter=counter)
    node_key_dict.update(edge_key_dict)
    return node_key_dict


def get_all_annots(graph_dir, counter=False):
    if counter:
        dict_all = defaultdict(lambda: defaultdict(lambda: 0))
    else:
        dict_all = defaultdict(set)
    i = 0
    for graph_name in tqdm(sorted(os.listdir(graph_dir))):
        i += 1
        graph = os.path.join(graph_dir, graph_name)
        graph = load_json(graph)
        graph_dict = dict_one(graph, counter=counter)
        for key, value in graph_dict.items():
            if counter:
                for inner_key, inner_value in value.items():
                    dict_all[key][inner_key] += inner_value
            else:
                dict_all[key] = dict_all[key].union(value)
        # if i > 20: break
    if counter:
        # Cast as a dict of dict
        for key, dict_value in dict_all.items():
            dict_all[key] = dict(dict_value)
        dict_all = dict(dict_all)
        pickle.dump(dict_all, open('all_annots.json', 'wb'))
    else:
        dict_all = {key: list(value) for key, value in dict_all.items()}
        json.dump(dict_all, open('all_annots.json', 'w'))


if __name__ == '__main__':
    graph_path = '../data/graphs/all_graphs'
    get_all_annots(graph_path, counter=True)
