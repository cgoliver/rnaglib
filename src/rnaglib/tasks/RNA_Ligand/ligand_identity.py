import os

import pandas as pd

from rnaglib.tasks import RNAClassificationTask
from rnaglib.splitters import RandomSplitter
from rnaglib.data_loading import RNADataset
from rnaglib.encoders import OneHotEncoder, IntEncoder
from rnaglib.transforms import FeaturesComputer, LigandAnnotator, LigandNTFilter, ResidueNameFilter
from rnaglib.utils import dump_json
    

class LigandIdentification(RNAClassificationTask):
    input_var = "nt_code"
    target_var = "ligand_code"

    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/gmsm_dataset.csv"))
    rnas_keep = set([id[0] for id in data.nid.str.split(".")])
    nodes_keep = set(data.nid.values)
    mapping = {i: i for i in range(len(data.label.unique()))}

    def __init__(self, root, splitter=None, **kwargs):
        super().__init__(root=root, splitter=splitter, **kwargs)
        pass

    def process(self):
        rna_filter = ResidueNameFilter(
            value_checker = lambda name: name in self.nodes_keep,
            min_valid = 1
        )
        nt_filter = LigandNTFilter(data=self.data)
        annotator = LigandAnnotator(data=self.data)

        # Run through database, applying our filters
        dataset = RNADataset(debug=self.debug, in_memory=self.in_memory)
        all_rnas = []
        os.makedirs(self.dataset_path, exist_ok=True)
        for rna in dataset:
            if rna_filter.forward(rna):
                rna_dict = next(iter(nt_filter(rna)))
                rna = annotator(rna_dict)["rna"]
                if self.in_memory:
                    all_rnas.append(rna)
                else:
                    all_rnas.append(rna.name)
                    dump_json(os.path.join(self.dataset_path, f"{rna.name}.json"), rna)
        if self.in_memory:
            dataset = RNADataset(rnas=all_rnas)
        else:
            dataset = RNADataset(dataset_path=self.dataset_path, rna_id_subset=all_rnas)

        return dataset
    
    def get_task_vars(self) -> FeaturesComputer:
        return FeaturesComputer(
            nt_features=self.input_var, 
            rna_targets=self.target_var,
            custom_encoders={
                self.target_var: IntEncoder()
            }
        )

