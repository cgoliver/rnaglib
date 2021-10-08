import os
import json
import pickle

from networkx.readwrite import json_graph
import networkx as nx


def dump_json(filename, graph):
    """
    Just a shortcut to dump a json graph more compactly
    :param filename: The dump name
    :param graph: The graph to dump
    """
    g_json = json_graph.node_link_data(graph)
    json.dump(g_json, open(filename, 'w'), indent=2)


def load_json(filename):
    """
    Just a shortcut to dump a json graph more compactly
    :param filename: The dump name
    :return: The loaded graph
    """
    with open(filename, 'r') as f:
        js_graph = json.load(f)
    out_graph = json_graph.node_link_graph(js_graph)
    return out_graph


def load_graph(filename):
    """
    This is a utility function that supports loading from json or pickle
    Sometimes, the pickle also contains rings in the form of a node dict,
    in which case the rings are added into the graph
    :param filename: json or pickle filename
    :return: networkx DiGraph object
    """
    if filename.endswith('json'):
        return load_json(filename)
    elif filename.endswith('p'):
        pickled = pickle.load(open(filename, 'rb'))
        # Depending on the data versionning, the object contained in the pickles is
        # - a graph with noderings in the nodes
        # - a dict {graph: , rings: }
        if isinstance(pickled, dict):
            graph = pickled['graph']
            # rings is a dict of dict {ring_type : {node : ring}}
            rings = pickled['rings']
            for ring_type, noderings in rings.items():
                nx.set_node_attributes(G=graph, name=f'{ring_type}_annots', values=noderings)
        else:
            graph = pickled
        return graph

    else:
        raise NotImplementedError('We have not implemented this data format yet')

def graph_from_pdbid(pdbid, graph_dir, graph_format='json'):
    if graph_format == 'nx':
        graph_name = os.path.join(graph_dir, pdbid.lower() + '.nx')
    elif graph_format == 'json':
        graph_name = os.path.join(graph_dir, pdbid.lower() + '.json')
    else:
        raise ValueError(f"Invalid graph format {graph_format}. Use NetworkX or JSON.")
    graph = load_graph(graph_name)
    return graph

if __name__ == '__main__':
    tmp_path = '../../examples/2du5.json'
    g = load_json(tmp_path)
    print(g.nodes())
