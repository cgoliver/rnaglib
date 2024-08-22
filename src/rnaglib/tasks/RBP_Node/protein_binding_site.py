from collections import defaultdict
import numpy as np
from networkx import set_node_attributes

from rnaglib.data_loading import RNADataset, FeaturesComputer
from rnaglib.splitters import RandomSplitter, SPLITTING_VARS, get_ribosomal_rnas, default_splitter_tr60_tr18
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.utils import load_index
from rnaglib.utils.feature_maps import BoolEncoder


class BenchmarkProteinBindingSiteDetection(ResidueClassificationTask):
    target_var = 'binding_site'  # "binding_site" needs to be replaced once dataset modifiable.
    input_var = "nt_code"
    rnaskeep = set(SPLITTING_VARS['TR60'] + SPLITTING_VARS['TE18'])

    def __init__(self, root, splitter=None, **kwargs):
        self.rna_id_to_chains = defaultdict(list)
        for pdb_chain in self.rnaskeep:
            pdb, chain = pdb_chain[:4], pdb_chain[4:]
            self.rna_id_to_chains[pdb].append(chain)
        super().__init__(root=root, splitter=splitter, **kwargs)

    def default_splitter(self):
        return default_splitter_tr60_tr18()

    def _nt_filter(self, x):
        # To get it split by chains,
        pdb_id = x.graph['pdbid'][0].lower()
        chains = self.rna_id_to_chains[pdb_id]
        for chain in chains:
            wrong_chain_nodes = [node for node in list(x) if chain != node.split('.')[1]]
            subgraph = x.copy()
            subgraph.remove_nodes_from(wrong_chain_nodes)
            subgraph.name = f'{pdb_id}_{chain}'
            yield subgraph

    def _annotator(self, x):
        binding_sites = {
            node: (not (nodedata.get("binding_small-molecule", None) is None and
                        nodedata.get("binding_ion", None) is None))
            for node, nodedata in x.nodes.items()}

        # Add RNA-FM embeddings
        sample_node = next(iter(x.nodes()))
        chain_embs = np.load(f"../../data/rnafm_chain_embs/{sample_node.rsplit('.', 1)[0]}.npz")
        # needs to be list or won't be json serialisable
        embeddings = {node: chain_embs[node].tolist() for node, nodedata in x.nodes.items()}
        set_node_attributes(x, binding_sites, 'binding_site')
        set_node_attributes(x, embeddings, 'embeddings')
        return x

    def build_dataset(self, root):
        features_computer = FeaturesComputer(nt_features=self.input_var,
                                             custom_encoders_targets={self.target_var: BoolEncoder()},
                                             extra_useful_keys=['embeddings'])
        dataset = RNADataset.from_args(features_computer=features_computer,
                                       nt_filter=self._nt_filter,
                                       annotator=self._annotator,
                                       all_rnas_db=[name[:-1] + '.json' for name in self.rnaskeep],
                                       redundancy='all',
                                       recompute=self.recompute)
        return dataset


class ProteinBindingSiteDetection(ResidueClassificationTask):
    target_var = "binding_protein"
    input_var = "nt_code"

    def __init__(self, root, splitter=None, **kwargs):
        self.ribosomal_rnas = get_ribosomal_rnas()
        super().__init__(root=root, splitter=splitter, **kwargs)

    def default_splitter(self):
        return RandomSplitter()

    def build_dataset(self, root):
        graph_index = load_index()
        rnas_keep = []
        for graph, graph_attrs in graph_index.items():
            rna_id = graph.split(".")[0]
            if "node_" + self.target_var in graph_attrs and rna_id not in self.ribosomal_rnas:
                rnas_keep.append(rna_id)

        features_computer = FeaturesComputer(nt_features=self.input_var, nt_targets=self.target_var)
        dataset = RNADataset.from_args(features_computer=features_computer,
                                       rna_filter=lambda x: x.graph['pdbid'][0].lower() in rnas_keep)
        return dataset
