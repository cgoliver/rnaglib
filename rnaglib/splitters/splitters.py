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
        return dataset.subset(self.train_pdbs), dataset.subset(self.val_pdbs), dataset.subset(self.test_pdbs)