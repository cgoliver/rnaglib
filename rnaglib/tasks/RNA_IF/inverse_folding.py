from rnaglib.data_loading import RNADataset
from rnaglib.tasks import ResidueClassificationTask
from rnaglib.splitters import RandomSplitter
from networkx import set_node_attributes

import pandas as pd
import ast
import os

class InverseFolding(ResidueClassificationTask):
    target_var = "nt_code" #in rna graph
    input_var = "is_modified" # should be dummy variable

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass
    pass
    
    def evaluate(self, data, predictions):
        return NotImplementedError
    
    def default_splitter(self):
        return RandomSplitter()
    
    def _annotator(self, x):
        dummy = {
            node: 1
            for node, nodedata in x.nodes.items()
        }
        set_node_attributes(x, dummy, 'dummy')
        return x

    def build_dataset(self, root):
        dataset = RNADataset(nt_targets=[self.target_var],
                             nt_features=[self.input_var],
                             rna_filter=lambda x: x.graph['pdbid'][0],
                             #annotator=self._annotator
                             )
        return dataset
