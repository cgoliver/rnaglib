import os
import sys

import networkx as nx
import pickle
from tqdm import tqdm

if __name__ == "__main__":
    sys.path = [os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../..")] + sys.path

from rnaglib.tasks.RNA_VS.ligands import MolGraphEncoder
from rnaglib.utils import graph_from_pdbid, graph_io


def build_data(root, recompute=False):
    script_dir = os.path.dirname(__file__)
    json_dump = os.path.join(script_dir, "../data/rna_vs/dataset_as_json.json")
    train_groups, test_groups = pickle.load(open(json_dump, 'rb'))
    all_groups = {**train_groups, **test_groups}

    # Check data was properly downloaded by getting one graph
    if graph_from_pdbid("1k73", redundancy='all') is None:
        raise FileNotFoundError("We could not fetch graphs, please be sure that you have downloaded all rna graphs. "
                                "If you didn't, you can run: rnaglib_download -r all")

    pocket_dir = os.path.join(root, 'graphs')
    os.makedirs(pocket_dir, exist_ok=True)

    print("Processing graphs...")
    failed_set = set()
    for group in tqdm(all_groups):
        pocket_path = os.path.join(pocket_dir, f"{group}.json")
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
        new_pocket_graph.graph['pocket_name'] = group
        nx.set_node_attributes(new_pocket_graph, values=nodes)
        graph_io.dump_json(pocket_path, new_pocket_graph)
    print(failed_set)
    print(f"{len(failed_set)}/{len(all_groups)} failed systems")

    ligand_file = os.path.join(root, 'ligands.p')
    if not os.path.exists(ligand_file) or recompute:
        print("Processing ligands")
        all_ligands = set()
        for group_rep, group in all_groups.items():
            actives = set(group['actives'])
            pdb_decoys = set(group['pdb_decoys'])
            chembl_decoys = set(group['chembl_decoys'])
            all_ligands = all_ligands.union(actives).union(pdb_decoys).union(chembl_decoys)

        mol_encoder = MolGraphEncoder(cache_path=None)
        all_mol_graphs = {}
        for sm in tqdm(list(all_ligands)):
            graph = mol_encoder.smiles_to_graph_one(sm)
            all_mol_graphs[sm] = graph
        pickle.dump(all_mol_graphs, open(ligand_file, 'wb'))


if __name__ == "__main__":
    default_dir = "../data/rna_vs"
    build_data(root=default_dir, recompute=False)
