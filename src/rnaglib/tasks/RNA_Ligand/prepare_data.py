import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from rnaglib.data_loading import RNADataset
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
binding_pockets = pd.DataFrame(columns = ['bp_id','nid','ligand','ligand_smiles','RNA'])
bp_ids_list = []
nids_list = []
ligands_list = []
ligand_smiles_list = []
rnas_list = []
bp_dict = {}
ligands_dict = {}
bp_id = 0
dataset_size = len(dataset)
for i, rna in tqdm(enumerate(dataset)):
    cif = str(Path(dataset.structures_path) / f"{rna['rna'].graph['pdbid'].lower()}.cif")
    mmcif_dict = MMCIF2Dict(cif)
    for ligand in rna['rna'].graph['ligands']:
        if not mmcif_dict['_chem_comp.type'][mmcif_dict['_chem_comp.id'].index(ligand['name'])] in ['RNA linking','DNA linking']:
            existing_nodes = set(rna['rna'].nodes())
            clean_rna_neighs = [nei for nei in ligand['rna_neighs'] if nei in existing_nodes]
            binding_pocket_nodes = bfs(rna['rna'], clean_rna_neighs, label="LW")
            n_hop = 2
            while len(binding_pocket_nodes)<40 and n_hop<4:
                binding_pocket_nodes = bfs(rna['rna'], binding_pocket_nodes, label="LW")
                n_hop += 1
            binding_pocket = rna['rna'].subgraph(binding_pocket_nodes).copy()
            if protein_content_filter.forward({'rna':binding_pocket}):
                try:
                    bp_dict[rna['rna'].name].append(binding_pocket_nodes)
                except:
                    bp_dict[rna['rna'].name] = [binding_pocket_nodes]
                bp_id += 1

                for rna_neighbor in tqdm(binding_pocket_nodes):
                    bp_ids_list.append(bp_id)
                    nids_list.append(rna_neighbor)
                    ligands_list.append(ligand['name'])
                    ligand_smiles_list.append(ligand['smiles'])
                    rnas_list.append(rna['rna'].name)


bp_dict2 = {key:list(bp_dict[key]) for key in bp_dict}
with open('bp_dict3.json', 'w') as f:
    json.dump(bp_dict2, f)

binding_pockets['bp_id'] = pd.Series(bp_ids_list)
binding_pockets['nid'] = pd.Series(nids_list)
binding_pockets['ligand'] = pd.Series(ligands_list)
binding_pockets['ligand_smiles'] = pd.Series(ligand_smiles_list)
binding_pockets['RNA'] = pd.Series(rnas_list)
binding_pockets.to_csv('binding_pockets4.csv')