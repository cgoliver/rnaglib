from rnaglib.dataset_transforms import DSTransform


class DistanceComputer(DSTransform):
    """Dataset transform adding to the dataset attributes a distance matrix encoding the pairwise distances between all RNAs of the dataset

    :param str name: the name identifying the distance metric
    :param bool recompute: a boolean indicating whether the distance has to be recomputed in case it had already been computed for this dataset (it can be set to True if the dataset may have been subject to modifications since the latest distance computation)
    """
    def __init__(self, name, recompute=True):
        self.name = name
        self.recompute = recompute
        pass

    def __call__(self, dataset):
        if dataset.distances is None or self.name not in dataset.distances or self.recompute:
            similarity_matrix, keep_dataset_names = self.forward(dataset)
            if len(keep_dataset_names) != len(dataset):
                print(
                    f"""WARNING: Similarity matrix computer dropped some data points.
                        Original dataset had: {len(dataset)}
                        New dataset has: {len(keep_dataset_names)}\n"""
                )
            # saving the distance matrices to the object in case we want to use them later
            dataset.remove_distance(self.name)
            dataset = dataset.subset(list_of_names=keep_dataset_names)
            dataset.add_distance(self.name, 1 - similarity_matrix)
        return dataset

    def forward(self, dataset):
        raise NotImplementedError
