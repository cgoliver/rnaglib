from rnaglib.tasks import RNAClassificationTask
from rnaglib.splitters import RandomSplitter
from rnaglib.data_loading import RNADataset
from rnaglib.utils import OneHotEncoder

import pandas as pd
from networkx import set_node_attributes
import os

class GMSM(RNAClassificationTask):
    target_var = 'ligand_code' #'ligand_code' # get annotator working, then change this back   
    input_var = 'nt_code'

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/gmsm_dataset.csv'))
    rnas_keep = set([id[0] for id in data.nid.str.split('.')])
    nodes_keep = data.nid.values

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass
    pass

    def default_splitter(self):
        return RandomSplitter()
    
    def _annotator(self, x):
        ligand_codes = {
            node: int(self.data.loc[self.data.nid == node, 'label'].values[0]) # remove .values[0] when playing with _nt_filter
            for node, nodedata in x.nodes.items()
        }
        set_node_attributes(x, ligand_codes, 'ligand_code')
        #set_node_attributes(x, {n: torch.zeroes(for n in x.nodes()}, 'custom')
        return x
    
    def _nt_filter(self, x):
        lacking_nodes = [node for node in x.nodes() if node not in self.nodes_keep]
        x.remove_nodes_from(lacking_nodes)
        return [x]

    def build_dataset(self, root):

        mapping = {i: i for i in range(len(self.data.label.unique()))}

        dataset = RNADataset(nt_targets=[self.target_var],
                             nt_features=[self.input_var],
                             annotator=self._annotator,
                             nt_filter = self._nt_filter,
                             custom_encoders = {self.target_var: OneHotEncoder(mapping=mapping)},
                             rna_filter=lambda x: x.graph['pdbid'][0].lower() in self.rnas_keep,
                             all_graphs=[name + '.json' for name in self.rnas_keep], # for testing [0:10]
                             redundancy='all'
                             )

        return dataset