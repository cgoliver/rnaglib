class Representation:
    """
    Callable object that accepts a raw RNA networkx object along with features and target vector representations
    and returns a representation of it (e.g. graph, voxel, point cloud)
    """

    def __init__(self):
        pass

    def __call__(self, rna_graph, features_dict):
        """
        Return

        :param rna_graph:
        :param features_dict:
        :return:
        """
        raise NotImplementedError

    @property
    def name(self):
        """
        Just return the name of the representation

        :return: A string
        """
        raise NotImplementedError

    def batch(self, samples):
        """
        Batch a list of voxel samples

        :param samples: A list of the output from this representation
        :return: a batched version of it.
        """
        raise NotImplementedError
