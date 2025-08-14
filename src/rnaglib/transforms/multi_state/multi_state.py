import copy
from typing import Union
from pathlib import Path
from collections import defaultdict

from rnaglib.transforms import Transform
from rnaglib.utils import download_graphs, load_graph

class MultiStateTransform(Transform):
    """Transform converting datasets of single RNAs into datasets of multi-state RNAs
    In fact, each element of the dataset (RNA dictionary) corresponding to a structure of an RNA is replaced by a list of
    dictionaries, each corresponding to one state/conformation of this same RNA

    
    :param dict multi_state_groups: dictionary which keys are names of RNA structures and which values are names of other states
    of these RNAs (ex: {RNA1_name:[RNA_1_apo_structure_name,RNA_1_holo_structure_name], ...}
    :param in_memory: When loading a dataset from files, you can choose to load the data in memory by setting in memory to true
    :param dataset_path: Path to the files storing the graphs of the RNA alternative states (not contained in initial RNADataset
    object) (ex: the graphs can be stored in JSON files)
    :param structures_path: Path to the structures of the RNA alternative states (not contained in initial RNADataset object) 
    (the structures are stored in CIF files)
    :param rnas: dictionary which keys are names of alternative RNA structures and which values are networkx graphs
    of these RNA structures (alternative to setting dataset_path and structures_path)

    Chaitanya K. Joshi, Arian R. Jamasb, Ramon Viñas, Charles Harris, Simon Mathis, Alex Morehead, 
    and Pietro Liò. gRNAde: Geometric Deep Learning for 3D RNA inverse design. International 
    Conference on Learning Representations 2025."""

    def __init__(self, multi_state_groups: dict, in_memory = False, dataset_path: Union[Path, str] = None, structures_path: Union[Path, str] = None, rnas = None, **kwargs):
        self.multi_state_groups = multi_state_groups
        if rnas is None:
            if dataset_path is None:
                dataset_path, structures_path = download_graphs(redundancy="all")
            if in_memory is not None and in_memory:
                self.to_memory()
            else:
                self.rnas = None
                self.in_memory = False
        else:
            # handle default choice
            self.in_memory = in_memory if in_memory is not None else True
            assert self.in_memory, (
                "Conflicting arguments: if an RNADataset is instantiated with a list of graphs, "
                "it must use 'in_memory=True'"
            )
            self.rnas = rnas
            self.structures_path = None
        self.dataset_path = dataset_path
        self.structures_path = structures_path
        super().__init__(**kwargs)

    def get_single_structure(self, rna_name: str):
        # Initialise paths
        nx_path, cif_path = None, None

        # Setting path to default path if no other is specified.
        if getattr(self, "dataset_path", None) is not None:
            nx_path = Path(self.dataset_path) / f"{rna_name}{self.extension}"

        if getattr(self, "structures_path", None) is not None:
            cif_path = Path(self.structures_path) / f"{rna_name}.cif"
        if self.in_memory:
            rna_graph = self.rnas[rna_name]
        else:
            rna_graph = load_graph(str(nx_path), multigraph=self.multigraph)
            rna_graph.name = rna_name
        # Compute features
        rna_dict = {"rna": rna_graph, "graph_path": nx_path, "cif_path": cif_path}

        return rna_dict

    def forward(self, rna_dict: dict):
        idx = rna_dict["rna"].name
        new_rna_dict = defaultdict(list)
        for key, value in rna_dict.items():
            new_rna_dict[key].append(value)
        for rna_name in self.multi_state_groups[idx]:
            conf_dict = self.get_single_structure(rna_name)
            for key, value in conf_dict.items():
                new_rna_dict[key].append(value)
        rna_dict = new_rna_dict
        return rna_dict
    
    def to_memory(self):
        """Make in_memory=True from a dataset not in memory."""
        if not hasattr(self, "rnas"):
            all_rnas = list(self.multi_state_groups.keys())+sum(self.multi_state_groups.values(),[])
            self.rnas = {g_name: load_graph(Path(self.dataset_path) / f"{g_name}{self.extension}") for g_name in all_rnas}
            self.in_memory = True

