from rnaglib.splitters import random_split

class Splitter:
    def __init__(self, split_train=0.7, split_valid=0.85):
        self.split_train = split_train
        self.split_valid = split_valid
        self.split_test = 1 - (split_train + split_valid)
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
    def __init__(self, train_pdbs, val_pdbs, test_pdbs, seed=0,  *args, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.train_pdbs = train_pdbs 
        self.val_pdbs =  val_pdbs
        self.test_pdbs = test_pdbs        
        pass

    def __call__(self, dataset):
        dataset_map = {value['rna'].graph['pdbid'][0].lower() + '.json': idx for idx, value in enumerate(dataset)}
        train_ind = [dataset_map[item] for item in self.train_pdbs if item in dataset_map]
        val_ind = [dataset_map[item] for item in self.val_pdbs if item in dataset_map]
        test_ind = [dataset_map[item] for item in self.test_pdbs if item in dataset_map]
        return train_ind, val_ind, test_ind