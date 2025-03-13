import json
import os
import networkx as nx
import pickle
from tqdm import tqdm

from rnaglib.tasks.RNA_VS.ligands import MolGraphEncoder
from rnaglib.utils import graph_io
from rnaglib.data_loading import rna_from_pdbid


def compress_migos_data(in_path="data/dataset_as_json.json", out_path="data/dataset_compressed.json"):
    """
    To go from the format in migos, with list of smiles, to more lightweight lists of ids.

    We use the compressed format for distributing the data.
    :param in_path:
    :param out_path:
    :return:
    """
    whole_data = json.load(open(in_path, 'r'))
    trainval_groups, test_groups = whole_data["trainval"], whole_data["test"]
    mol_to_id = dict()
    max_mol = 0
    reduced_datasets = []
    for dataset in [trainval_groups, test_groups]:
        dataset_reduced = dataset.copy()
        for k, v in dataset.items():
            for mol_name in ['actives', 'pdb_decoys', 'chembl_decoys']:
                mol_name_mapping = []
                for mol in v[mol_name]:
                    if mol not in mol_to_id:
                        mol_to_id[mol] = max_mol
                        max_mol += 1
                    mol_name_mapping.append(mol_to_id[mol])
                dataset_reduced[k][mol_name] = mol_name_mapping
        reduced_datasets.append(dataset_reduced)

    compressed_data = {
        'compressed_trainval_groups': reduced_datasets[0],
        'compressed_test_groups': reduced_datasets[1],
        'mapper': mol_to_id
    }
    json.dump(compressed_data, open(out_path, 'w'))


def expand_compressed_data(in_path="data/dataset_compressed.json", out_path="data/dataset_as_json.json"):
    """
    To go from more lightweight lists of ids to the format in migos with list of smiles.
    :param in_path:
    :param out_path:
    :return:
    """
    compressed = json.load(open(in_path, 'r'))
    mol_to_id = compressed["mapper"]
    id_to_mol = {v: k for k, v in mol_to_id.items()}
    compressed_datasets = [compressed["compressed_trainval_groups"], compressed["compressed_test_groups"]]

    expanded_datasets = []
    for compressed_dataset in compressed_datasets:
        expanded = compressed_dataset.copy()
        for k, v in compressed_dataset.items():
            for mol_name in ['actives', 'pdb_decoys', 'chembl_decoys']:
                mol_name_mapping = [id_to_mol[mol_id] for mol_id in v[mol_name]]
                expanded[k][mol_name] = mol_name_mapping
        expanded_datasets.append(expanded)
    expanded_data = {"trainval": expanded_datasets[0], "test": expanded_datasets[1]}
    json.dump(expanded_data, open(out_path, 'w'))


def load_groups():
    script_dir = os.path.dirname(__file__)
    json_dump = os.path.join(script_dir, "data/dataset_as_json.json")
    if not os.path.exists(json_dump):
        json_dump_compressed = os.path.join(script_dir, "data/dataset_compressed.json")
        expand_compressed_data(in_path=json_dump_compressed, out_path=json_dump)
    whole_data = json.load(open(json_dump, 'r'))
    train_groups, test_groups = whole_data["trainval"], whole_data["test"]
    all_groups = {**train_groups, **test_groups}
    return all_groups


def dump_rna_jsons(root, recompute=False):
    all_groups = load_groups()
    # Check data was properly downloaded by getting one graph
    if rna_from_pdbid("1k73", redundancy='all') is None:
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
        rglib_graph = rna_from_pdbid(pdb_id, redundancy='all', verbose=False)['rna']
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


def precompute_ligand_graphs(root, recompute=False, framework="dgl"):
    all_groups = load_groups()
    # Go through all the ligands in the dataset, and dump a precomputed DGL representation
    ligand_file = os.path.join(root, f'ligands_{framework}.p')
    if not os.path.exists(ligand_file) or recompute:
        print("Processing ligands")
        all_ligands = set()
        for group_rep, group in all_groups.items():
            actives = set(group['actives'])
            pdb_decoys = set(group['pdb_decoys'])
            chembl_decoys = set(group['chembl_decoys'])
            all_ligands = all_ligands.union(actives).union(pdb_decoys).union(chembl_decoys)

        mol_encoder = MolGraphEncoder(framework=framework, cache_path=None)
        all_mol_graphs = {}
        for sm in tqdm(list(all_ligands)):
            graph = mol_encoder.smiles_to_graph_one(sm)
            all_mol_graphs[sm] = graph
        pickle.dump(all_mol_graphs, open(ligand_file, 'wb'))


if __name__ == "__main__":
    default_dir = "data"
    # compress_migos_data()
    # expand_compressed_data()
    dump_rna_jsons(root=default_dir, recompute=False)
    precompute_ligand_graphs(root=default_dir, recompute=False, framework="dgl")
    precompute_ligand_graphs(root=default_dir, recompute=False, framework="pyg")
