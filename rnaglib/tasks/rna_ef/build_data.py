import os
import sys

import networkx as nx
import pickle
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from rnaglib.utils import graph_from_pdbid
from rnaglib.utils import graph_io


def build_data(root, recompute=False):
    pocket_dir = os.path.join(root, 'graphs')
    os.makedirs(pocket_dir, exist_ok=True)

    script_dir = os.path.dirname(__file__)
    json_dump = os.path.join(script_dir, "../../data/tasks/rna_ef/dataset_as_json.json")

    train_groups, test_groups = pickle.load(open(json_dump, 'rb'))
    all_groups = {**train_groups, **test_groups}

    failed_set = set()
    for group in tqdm(all_groups):
        pocket_path = os.path.join(pocket_dir, group)
        if os.path.exists(pocket_path) and not recompute:
            continue
        pdb_id = group[:4].lower()
        rglib_graph = graph_from_pdbid(pdb_id, redundancy='all')
        if rglib_graph is None:
            failed_set.add(group)
            print(rglib_graph, 'not found')
            continue
        nodes = all_groups[group]['nodes']
        new_pocket_graph = rglib_graph.subgraph(nodes)
        nx.set_node_attributes(new_pocket_graph, values=nodes)
        graph_io.dump_json(pocket_path, new_pocket_graph)
    print(failed_set)
    print(f"{len(failed_set)}/{len(all_groups)} failed systems")


default_dir = "../../data/tasks/rna_ef"
build_data(root=default_dir)
