import os
import json

from scipy.stats import pearsonr
from networkx.readwrite import json_graph

GRAPH_DIR = "../data/graphs_resolution"

def load_json(filename):
    with open(filename, 'r') as f:
        js_graph = json.load(f)
    out_graph = json_graph.node_link_graph(js_graph)
    return out_graph

resolutions, dot_freq  = [], []
for g in os.listdir(GRAPH_DIR):
    G = load_json(os.path.join(GRAPH_DIR, g))
    res = G.graph['resolutioin_high']
    if res is None:
        continue
    else:
        try:
            res = float(res[0])
        except:
            continue

    num_dots = sum(['.' in label['LW'] for _,_,label in G.edges(data=True)])
    frac_dots = num_dots / len(G.edges())
    resolutions.append(res)
    dot_freq.append(frac_dots)

    if len(resolutions) > 2:
        print(pearsonr(resolutions, dot_freq))
