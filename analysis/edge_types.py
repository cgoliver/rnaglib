import os
import json
from collections import defaultdict

from tqdm import tqdm
import networkx as nx
from networkx.readwrite import json_graph

def load_json(filename):
    with open(filename, 'r') as f:
        js_graph = json.load(f)
    out_graph = json_graph.node_link_graph(js_graph)
    return out_graph

def process_all(graph_path="../data/graphs"):
    counts = defaultdict(int)
    tot = 0
    for g_name in tqdm(os.listdir(graph_path)):
        G = load_json(os.path.join(graph_path, g_name))
        for _,_,d in G.edges(data=True):
            if d['LW'] == 'B53':
                continue
            counts[d['LW']] += 1
            tot += 1

    dots = sum((c for edge, c in counts.items() if '.' in edge))
    print(f"Total edges: {tot}")
    print(f"Total dot edges {dots}")
    print(f"Frequency dot edges {dots / tot }")
    for edge, count in counts.items():
        print(f"{edge}: {count}, {count / tot}")
if __name__ == "__main__":
    process_all()
    pass
