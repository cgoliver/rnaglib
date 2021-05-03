"""
Check edge direcitonality.
"""
import os
import json

import networkx as nx
from networkx.readwrite import json_graph

GRAPH_DIR = "../data/graphs_resolution"

def load_json(filename):
    with open(filename, 'r') as f:
        js_graph = json.load(f)
    out_graph = json_graph.node_link_graph(js_graph)
    return out_graph

for g in os.listdir(GRAPH_DIR):
    G = load_json(os.path.join(GRAPH_DIR, g))
    print(g)
    print("Is it a nx directed graph?: ", nx.is_directed(G))
    for u, v, d in G.edges(data=True):
        if d['LW'] == 'B53':
            continue
        try:
            print("="*20)
            print(G[v][u], d['LW'])
            print(u, v, d['LW'])
            print("="*20)
        except KeyError:
            continue

