"""
This script is used to find all possible annotations present in dssr and dump them in a dict.

This is useful to handcraft encoding functions and to design data splits
"""

import os
import sys
import importlib

import json
import pickle
from tqdm import tqdm
from collections import defaultdict, Counter

from rnaglib.utils import load_json


def process_graph_dict(dict_to_flatten, prepend=None, counter=False, possible_supervisions=None):
    """
    Loop over a dictionary that represents either the nodes or edges data. (following networkx)
    The keys are edges or node ids and the values are dicts of features pertaining to these.
    We want to gather this information into just one clean dict {key:set}

    We optionally prepend a string to the keys or use a counter to get the count for each of those instances.
    We also optionally select certain keys to loop over (skipping the others)

    :param dict_to_flatten: The outer dict we want to gather the features of
    :param prepend: An optional prefix to add to the resulting dict. Useful to make the distinction between node
    and edge features
    :param counter: Whether to return just the items or their associated counts
    :param possible_supervisions: A list of features we want. By default, we count them all
    :return: A flattened dictionary with all the possible values of the node/edge data and optionally their counter
    For instance : {nucleotide_type : {'A', 'U', 'C', 'G', 'a', 'u', 'c', 'g'}
    """
    if counter:
        return_dict = defaultdict(lambda: defaultdict(lambda: 0))
    else:
        return_dict = defaultdict(set)
    for outer_key, outer_value in dict_to_flatten.items():
        for inner_key, inner_value in outer_value.items():
            if prepend is not None:
                inner_key = f"{prepend}_{inner_key}"
            if possible_supervisions is not None and inner_key not in possible_supervisions:
                continue
            if counter:
                # Only count interesting strings : if no supervision is provided, we only count specific string
                if possible_supervisions is not None:
                    if inner_value is not None:
                        if type(inner_value) == dict:
                            hashable_value = True
                        else:
                            hashable_value = inner_value
                        # PATCH until next release we replace small mol list with tuple
                        if inner_key == "node_binding_small-molecule" or inner_key == "node_binding_ion":
                            hashable_value = hashable_value[0]

                        return_dict[inner_key][hashable_value] += 1

                else:
                    if type(inner_value) == str and inner_key not in ["node_nt_id", "edge_nt1", "edge_nt2"]:
                        return_dict[inner_key][inner_value] += 1
            else:
                if type(inner_value) == list:
                    inner_value = tuple(inner_value)
                    return_dict[inner_key].add(inner_value)
                if type(inner_value) == dict:
                    # Just one does that : it is 'node frame'
                    if inner_value != "node_frame":
                        pass
                    for inner_key, inner_value in inner_value.items():
                        if type(inner_value) == list:
                            inner_value = tuple(inner_value)
                        return_dict[f"{inner_key}_{inner_key}"].add(inner_value)
                elif type(inner_value) in [str, int, float, list, bool] or inner_value is None:
                    pass
                else:
                    return_dict[inner_key].add(inner_value)
    return return_dict


def graph_to_dict(graph, counter=False, possible_supervisions=None):
    """
    Turn a graph dictionaries into one dict { possible node/edge keys : set of existing values }

    This is useful to list and count all possible values for the data associated with the edges and
    nodes of a set of graphs

    :param graph: The graph to count over
    :param counter: Boolean. Whether to also return counts
    :param possible_supervisions: A list of keys we want to process. By default, process all
    :return: A dictionary with node and edge data keys and their associated possible values.
    """

    # Nx returns a Node/Edge view object, a list filled with (node, data)/(source, target, data) tuples
    list_nodes = graph.nodes(data=True)
    dict_nodes = {u: data for u, data in list_nodes}
    list_edges = graph.edges(data=True)
    dict_edges = {(u, v): data for u, v, data in list_edges}

    node_key_dict = process_graph_dict(
        dict_nodes, prepend="node", counter=counter, possible_supervisions=possible_supervisions
    )
    edge_key_dict = process_graph_dict(
        dict_edges, prepend="edge", counter=counter, possible_supervisions=possible_supervisions
    )
    node_key_dict.update(edge_key_dict)
    return node_key_dict


def get_all_annots(graph_dir, counter=False, dump_name="all_annots.json"):
    """
    This function is used to investigate all possible labels in the data, all edge and node attributes...

    Loop over all the graphs in the dir, then call the above function that flattens the possible values into dicts
    and group those results into one big dict.

    :param graph_dir: The directory containing the set of graphs to loop over
    :param counter: Whether to also return the associated counts
    :param dump_name: Where to dump the results
    :return: The resulting flattened dict.
    """
    if counter:
        dict_all = defaultdict(lambda: defaultdict(lambda: 0))
    else:
        dict_all = defaultdict(set)
    i = 0
    for graph_name in tqdm(sorted(os.listdir(graph_dir))):
        i += 1
        graph = os.path.join(graph_dir, graph_name)
        graph = load_json(graph)
        graph_dict = graph_to_dict(graph, counter=counter)
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
        pickle.dump(dict_all, open(dump_name, "wb"))
    else:
        dict_all = {key: list(value) for key, value in dict_all.items()}
        json.dump(dict_all, open(dump_name, "w"))
    return dict_all


def get_graph_indexes(graph_dir, possible_supervisions=None, dump_name="graph_index_NR.json"):
    """
    This function is used to create data splits. For each graph, we want to report which fields it contains
    in one object, to avoid having to load all graphs every time

    We want to return a dict of dict of dict. {graph_name : {fields : { values of the field :number of occurences }}}

    :param graph_dir: The directory containing the graphs we want to loop over
    :param possible_supervisions: The elements or fields we want to include in the resulting dict
    :param dump_name: Where to dump the results
    :return: The resulting dict : {graph_name : {fields : { values of the field :number of occurences }}}
    """
    dict_all = dict()
    i = 0
    for graph_name in tqdm(sorted(os.listdir(graph_dir))):
        i += 1
        graph = os.path.join(graph_dir, graph_name)
        graph = load_json(graph)
        graph_dict = graph_to_dict(graph, counter=True, possible_supervisions=possible_supervisions)
        dict_all[graph_name] = graph_dict
    for key, dict_value in dict_all.items():
        for inner_key, inner_dict_value in dict_value.items():
            dict_value[inner_key] = dict(inner_dict_value)
        dict_all[key] = dict(dict_value)
    with open(dump_name, "w") as js:
        json.dump(dict_all, js)
    return dict_all


if __name__ == "__main__":
    graph_path = "../data/graphs/NR/"
    # get_all_annots(graph_path, counter=True)
    index = get_graph_indexes(
        graph_path,
        dump_name="graph_index_NR.json",
        possible_supervisions={
            "node_binding_small-molecule",
            "node_binding_protein",
            "node_binding_ion",
            "node_is_modified",
        },
    )
    # test_split = get_splits(query_attrs={'node_binding_protein_id', "node_binding_ion",
    #                                      "is_modified", 'node_binding_small-molecule'})
