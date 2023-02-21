import os
import sys

import copy

from rnaglib.utils import download_graphs
from rnaglib.utils import load_graph

class RNAGlibDataset:
    def __init__(self,
                 data_path=None,
                 version='0.0.0',
                 download_dir=None,
                 redundancy='nr',
                 all_graphs=None,
                 annotated=False,
                 verbose=False):
        """
        This class is the main object to hold the core RNA data annotations.
        The `RNAglibDataset.all_rnas` object is a generator networkx objects that hold all the annotations for each RNA in the dataset.
        You can also access individual RNAs on-disk with `RNAGlibDataset()[idx]` or `RNAGlibDataset().get_pdbid('1b23')`

        :param data_path: The path to the folder containing the graphs. If node_sim is not None, this data should be annotated
        :param version: Version of the dataset to use (default='0.0.0')
        :param redundancy: To use all graphs or just the non redundant set.
        :param all_graphs: In the given directory, one can choose to provide a list of graphs to use
        :return:
        """

        # If we don't input a data path, the right one according to redundancy, chop and annotated is fetched
        # By default, we set hashing to None and potential node sim should be specified when creating
        # the node_sim function.
        # Then if a download occurs and no hashing was provided to the loader, the hashing used is the one
        # fetched by the downloading process to ensure it matches the data we iterate over.
        self.data_path = data_path
        if data_path is None:
            self.data_path = download_graphs(redundancy=redundancy,
                                             version=version,
                                             annotated=annotated,
                                             data_root=download_dir,
                                           )

            self.graph_path = os.path.join(self.data_path, 'graphs')
        if all_graphs is not None:
            self.all_graphs = all_graphs
        else:
            self.all_graphs = sorted(os.listdir(self.graph_path))

        self.available_pdbids = [g.split(".")[0].lower() for g in self.all_graphs]
        self.all_rnas = (self.get(i) for i in range(len(self.available_pdbids)))

    def __len__(self):
        return len(self.all_graphs)

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx):
        return load_graph(os.path.join(self.graph_path, self.all_graphs[idx]))

    def get_pdbid(self, pdbid):
        """ Grab an RNA by its pdbid """
        return self.get(self.available_pdbids.index(pdbid.lower()))

    def to_graph(self):
        pass
    def to_voxel(self):
        pass
    def to_cloud(self):
        pass

