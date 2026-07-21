from collections import defaultdict
import pandas as pd
import networkx as nx
import json
from tqdm import tqdm

from rnaglib.dataset import RNADataset
from rnaglib.transforms import SizeFilter, ResidueAttributeFilter
from rnaglib.algorithms import bfs

# contacts at 6.0A and at least 10 contacting residues, the binding site definition of HARIBOSS, which rna_site uses too
BINDING_ATTRIBUTE = "binding_small-molecule-6.0A"

# Initialize dataset with in_memory=False to avoid loading everything at once
dataset = RNADataset(debug=False, in_memory=False, redundancy="all")

# Instantiate filters to apply
size_filter = SizeFilter(5, 500)
binding_site_filter = ResidueAttributeFilter(attribute=BINDING_ATTRIBUTE,
                                             value_checker=lambda val: val is not None,
                                             aggregation_mode="min_valid",
                                             min_valid=10)


# Run through database, collecting the binding pockets
bp_dict = defaultdict(list)
ligands_dict = {}
bp_id = 0
dataset_size = len(dataset)
for i, rna in tqdm(enumerate(dataset),total=dataset_size):
    lig_to_nodes = defaultdict(list)
    lig_info = nx.get_node_attributes(rna['rna'], BINDING_ATTRIBUTE)
    for node, ligand in lig_info.items():
        if not ligand is None:
            lig_to_nodes[tuple(ligand['id'])].append(node)
    # nothing is filtered on the ligand or on the protein content of the pocket here: SmallMoleculeBindingTransform
    # already discards chain-integrated components, ligands bound to the protein rather than to the RNA and, when
    # asked, crystallization additives, all of them on the coordinates. Anything still annotated at this point is a
    # genuine free ligand sitting on the RNA
    for ligand_id, binding_pocket in lig_to_nodes.items():
        # a ligand merely grazing the surface does not make a binding site. Every node of the seed carries the
        # annotation of this ligand and of no other, so the filter counts exactly the residues in contact with it
        if not binding_site_filter.forward({'rna': rna['rna'].subgraph(binding_pocket)}):
            continue
        for node in binding_pocket:
            ligands_dict[node] = lig_info[node]['name']
        binding_pocket_nodes = bfs(rna['rna'], binding_pocket, label="LW")
        n_hop = 2
        while len(binding_pocket_nodes)<40 and n_hop<4:
            binding_pocket_nodes = bfs(rna['rna'], binding_pocket_nodes, label="LW")
            n_hop += 1
        bp_dict[rna['rna'].name].append(list(binding_pocket_nodes))
        bp_id += 1

bp_dict = {key:list(bp_dict[key]) for key in bp_dict}

with open('bp_dict_lastversion.json', 'w') as bp_dict_json:
    json.dump(bp_dict, bp_dict_json)

with open('ligands_dict_lastversion.json', 'w') as ligands_dict_json:
    json.dump(ligands_dict, ligands_dict_json)