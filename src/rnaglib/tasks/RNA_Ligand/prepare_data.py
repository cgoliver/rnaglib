import numpy as np
from collections import defaultdict
import pandas as pd
import networkx as nx
import json
from tqdm import tqdm
from pathlib import Path
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from rnaglib.dataset import RNADataset
from rnaglib.transforms import SizeFilter, ResidueAttributeFilter, ComposeFilters
from rnaglib.algorithms import bfs

# Initialize dataset with in_memory=False to avoid loading everything at once
dataset = RNADataset(debug=False, in_memory=False, redundancy="all")

# Instantiate filters to apply
protein_content_filter = ResidueAttributeFilter(attribute="protein_content_8.0", aggregation_mode="aggfunc", value_checker=lambda x: x < 10, aggfunc=np.mean)
size_filter = SizeFilter(5, 500)
binding_pocket_filters_list = [protein_content_filter, size_filter]
binding_pocket_filters = ComposeFilters(binding_pocket_filters_list)


# Run through database, applying our filters
bp_dict = defaultdict(list)
ligands_dict = {}
bp_id = 0
dataset_size = len(dataset)
for i, rna in tqdm(enumerate(dataset),total=dataset_size):
    cif = str(Path(dataset.structures_path) / f"{rna['rna'].graph['pdbid'].lower()}.cif")
    mmcif_dict = MMCIF2Dict(cif)
    lig_to_nodes = defaultdict(list)
    lig_info = nx.get_node_attributes(rna['rna'], 'binding_small-molecule-6.0A')
    for node, ligand in lig_info.items():
        if not ligand is None:
            lig_to_nodes[tuple(ligand['id'])].append(node)
            ligands_dict[node] = ligand['name']
    for ligand_id, binding_pocket in lig_to_nodes.items():
        ligand_name = ligand_id[0][2:]
        if not mmcif_dict['_chem_comp.type'][mmcif_dict['_chem_comp.id'].index(ligand_name)] in ['RNA linking','DNA linking']:
            binding_pocket_nodes = bfs(rna['rna'], binding_pocket, label="LW")
            n_hop = 2
            while len(binding_pocket_nodes)<40 and n_hop<4:
                binding_pocket_nodes = bfs(rna['rna'], binding_pocket_nodes, label="LW")
                n_hop += 1
            binding_pocket = rna['rna'].subgraph(binding_pocket_nodes).copy()
            if protein_content_filter.forward({'rna':binding_pocket}):
                bp_dict[rna['rna'].name].append(list(binding_pocket_nodes))
                bp_id += 1    

bp_dict = {key:list(bp_dict[key]) for key in bp_dict}

with open('bp_dict_lastversion.json', 'w') as bp_dict_json:
    json.dump(bp_dict, bp_dict_json)

with open('ligands_dict_lastversion.json', 'w') as ligands_dict_json:
    json.dump(ligands_dict, ligands_dict_json)