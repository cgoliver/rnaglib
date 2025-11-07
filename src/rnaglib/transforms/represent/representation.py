class Representation:
    """
    Callable object that accepts a raw RNA networkx object along with features and target vector representations
    and returns a representation of it (e.g. graph, voxel, point cloud)
    """

    def __init__(self):
        pass

    def __call__(self, rna_graph, features_dict):
        """Transform an RNA graph into a representation.

        :param rna_graph: NetworkX graph representing the RNA structure
        :param features_dict: Dictionary containing node and edge features
        :return: Representation of the RNA (e.g., graph, voxel, point cloud)
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
