import json
from networkx.readwrite import json_graph


def dump_json(filename, graph):
    g_json = json_graph.node_link_data(graph)
    json.dump(g_json, open(filename, 'w'), indent=2)


def load_json(filename):
    with open(filename, 'r') as f:
        js_graph = json.load(f)
    out_graph = json_graph.node_link_graph(js_graph)
    return out_graph


if __name__=='__main__':
    tmp_path = '../../examples/2du5.json'
    g = load_json(tmp_path)
    print(g.nodes())
