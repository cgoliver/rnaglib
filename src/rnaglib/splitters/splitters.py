from rnaglib.splitters import random_split
import os
from torch import load
import pandas as pd
import ast

class Splitter:
    def __init__(self, split_train=0.7, split_valid=0.15, split_test=0.15):
        assert sum([split_train, split_valid, split_test]) == 1, "Splits don't sum to 1."
        self.split_train = split_train
        self.split_valid = split_valid
        self.split_test = split_test
        pass

    def __call__(self, dataset):
        return None, None, None


class RandomSplitter(Splitter):
    def __init__(self, seed=0, *args, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        pass

    def __call__(self, dataset):
        return random_split(dataset,
                            split_train=self.split_train,
                            split_valid=self.split_valid,
                            seed=self.seed
                            )


class BenchmarkBindingSiteSplitter(Splitter):
    def __init__(self, train_pdbs, val_pdbs, test_pdbs, seed=0, *args, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.train_pdbs = train_pdbs
        self.val_pdbs = val_pdbs
        self.test_pdbs = test_pdbs
        pass

    def __call__(self, dataset):
        dataset_map = {value['rna'].graph['pdbid'][0].lower() + '.json': idx for idx, value in enumerate(dataset)}
        train_ind = [dataset_map[item] for item in self.train_pdbs if item in dataset_map]
        val_ind = [dataset_map[item] for item in self.val_pdbs if item in dataset_map]
        test_ind = [dataset_map[item] for item in self.test_pdbs if item in dataset_map]
        return train_ind, val_ind, test_ind


class DasSplitter(Splitter):
    def __init__(self, seed=0, *args, **kwargs):
        super().__init__(**kwargs)
        print('Initialising DasSplitter')
        self.seed = seed

        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(current_dir)
        splits_path = os.path.join(parent_dir, 'tasks/data', 'das_split.pt')
        metadata_path = os.path.join(parent_dir, 'tasks/data', 'gRNAde_metadata.csv')
        
        #Note that preprocessing is needed since splits contain indices of compounds that may contain multiple pdbs. Our approach treats each pdb as an individual sample.
              
        splits = load(splits_path)
        metadata = pd.read_csv(metadata_path)
        metadata_ids = metadata['id_list'].apply(ast.literal_eval)
        train_pdbs = self._process_split(metadata_ids, splits[0])
        val_pdbs = self._process_split(metadata_ids, splits[1])
        test_pdbs = self._process_split(metadata_ids, splits[2])
        # If you want to convince yourself that this is the right order, see this notebook: https://github.com/chaitjo/geometric-rna-design/blob/deccaa0139f7f9130487858ece2fbca331100369/notebooks/split_das.ipynb 
        self.train_pdbs = train_pdbs
        self.val_pdbs = val_pdbs
        self.test_pdbs = test_pdbs
        pass


    def __call__(self, dataset):
        print('Generating split indices')
        dataset_map = {value['rna'].graph['pdbid'][0] : idx for idx, value in enumerate(dataset)}
        train_ind = [dataset_map[item] for item in self.train_pdbs if item in dataset_map]
        val_ind = [dataset_map[item] for item in self.val_pdbs if item in dataset_map]
        test_ind = [dataset_map[item] for item in self.test_pdbs if item in dataset_map]
        return train_ind, val_ind, test_ind
    
    def _process_split(self, metadata_ids, indices):
        return [x.split('_')[0] for x in sum(metadata_ids.iloc[indices].to_list(), [])]
