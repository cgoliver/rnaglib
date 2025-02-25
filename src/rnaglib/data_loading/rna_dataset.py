"""Main class for collections of RNAs."""

import json
import os
import copy

from bidict import bidict
from pathlib import Path
from typing import Dict, List, Union, Literal
from collections.abc import Iterable

from bidict import bidict
import networkx as nx
import numpy as np

from rnaglib.transforms.transform import Transform, AnnotationTransform
from rnaglib.transforms.represent import Representation
from rnaglib.transforms.featurize import FeaturesComputer
from rnaglib.data_loading.create_dataset import database_to_dataset
from rnaglib.utils import download_graphs, load_graph, dump_json
from rnaglib.utils.graph_io import get_all_existing, get_name_extension


class RNADataset:
    """
    This class is the main object to hold the core RNA data annotations.

    The ``RNAglibDataset.all_rnas`` object is a list of networkx objects that holds all the annotations for each RNA in the dataset.

    :param rnas: One can instantiate directly from a list of RNA files
    :param dataset_path: The path to the folder containing the graphs.
    :param rna_id_subset: In the given directory, ``'dataset_path'``, one can choose to provide a list of graphs filenames to keep instead of using all available.
    :param multigraph: Whether to load RNAs as multi-graphs or simple graphs. Multigraphs can have backbone and base pairs between the same two residues.
    :param in_memory: Whether to load all RNA graphs in memory or to load them on the fly
    :param features_computer: A FeaturesComputer object, useful to transform raw RNA data into tensors.
    :param representations: List of :class:`~rnaglib.representations.Representation` objects to apply to each item.

    The dataset holds an attribute self.all_rnas = bidict({rna_name: i for i, rna_name in enumerate(all_rna_names)})
    Where rna_name is expected to match the file name the rna should be saved in.

    Examples
    ---------
    Create a default dataset::

        >>> from rnaglib.data_loading import RNADataset
        >>> dataset = RNADataset()


    Access the first item in the dataset::

        >>> dataset[0]


    Each item is a dictionary with the key 'rna' holding annotations as a networkx Graph.

        >>> dataset['rna'].nodes()
        >>> dataset['rna'].edges()


    Access an RNA by its PDBID::

        >>> dataset.get_pdbid('4nlf')




    .. Hint::
        Pass ``debug=True`` to ``RNADataset`` to quickly load a small dataset for testing.



    """

    def __init__(
        self,
        rnas: List[nx.Graph] = None,
        dataset_path: Union[str, os.PathLike] = None,
        version="2.0.2",
        redundancy="nr",
        rna_id_subset: List[str] = None,
        recompute_mapping: bool = True,
        in_memory: bool = True,
        features_computer: FeaturesComputer = None,
        representations: Union[List[Representation], Representation] = None,
        debug: bool = False,
        get_pdbs: bool = True,
        overwrite: bool = False,
        multigraph: bool = False,
        pre_transforms: Union[List[Transform], Transform] = None,
        transforms: Union[List[Transform], Transform] = None,
        **kwargs,
    ):
        self.in_memory = in_memory
        self.transforms = transforms
        self.pre_transforms = pre_transforms
        self.multigraph = multigraph

        if dataset_path is not None:
            self.dataset_path = dataset_path

        # Distance is computed as a cached property
        # We potentially want to save distances and the bidict mapping
        self.distances_ = None
        self.distances_path = os.path.join(dataset_path, "distances.npz") if dataset_path is not None else None
        self.bidict_path = os.path.join(dataset_path, "bidict.json") if dataset_path is not None else None

        if rnas is None:
            if dataset_path is None:
                # By default, use non redundant (nr), v1.0.0 dataset of rglib
                dataset_path, structures_path = download_graphs(
                    redundancy=redundancy,
                    version=version,
                    debug=debug,
                    get_pdbs=get_pdbs,
                    overwrite=overwrite,
                )
                self.dataset_path = dataset_path
                self.structures_path = structures_path

            # One can restrict the number of graphs to use
            existing_all_rnas, extension = get_all_existing(dataset_path=self.dataset_path, all_rnas=rna_id_subset)
            self.extension = extension
            if recompute_mapping or not os.path.exists(self.bidict_path):

                # If debugging, only keep the first few
                if debug:
                    existing_all_rnas = existing_all_rnas[:50]

                # Keep track of a list_id <=> system mapping. First remove extensions
                existing_all_rna_names = [get_name_extension(rna, permissive=True)[0] for rna in existing_all_rnas]
                self.all_rnas = bidict({rna: i for i, rna in enumerate(existing_all_rna_names)})

            else:
                simple_dict = json.load(open(self.bidict_path, "r"))
                self.all_rnas = bidict(simple_dict)

            if in_memory:
                self.to_memory()
            else:
                self.rnas = None
        else:
            self.structures_path = None
            assert in_memory, (
                "Conflicting arguments: if an RNADataset is instantiated with a list of graphs, "
                "it must use 'in_memory=True'"
            )
            self.rnas = rnas

            # Here we assume that rna lists contain a relevant rna.name field, which is the case
            # if it was constructed using build_dataset above
            rna_names = set([rna.name for rna in rnas])
            assert "" not in rna_names, "Empty RNA name found"
            assert len(rna_names) == len(rnas), (
                "When creating a RNAdataset from rnas, please " "use uniquely named networkx graphs"
            )
            self.all_rnas = bidict({rna.name: i for i, rna in enumerate(rnas)})

        # Now that we have the raw data setup, let us set up the features we want to be using:
        self.features_computer = FeaturesComputer() if features_computer is None else features_computer

        # pass transforms to the features computer to make the features available to the feat_dict
        if not pre_transforms is None:
            # tranforms work on the dict so have to get back the graph with the 'rna' key
            # this is annoying
            self.rnas = [pre_transforms({"rna": rna})["rna"] for rna in self.rnas]

        # Finally, let us set up the list of representations that we will be using
        if representations is None:
            self.representations = []
        elif not isinstance(representations, list):
            self.representations = [representations]
        else:
            self.representations = representations

    @property
    def distances(self):
        """
        Using a cached property is useful for loading precomputed data.
        If this is the first call, try loading. Otherwise, return the loaded value
        """
        if self.distances_ is not None:
            return self.distances_
        if self.distances_path is not None:
            if os.path.exists(self.distances_path):
                # Actually materialize memory (lightweight anyway) since npz loading is lazy
                self.distances_ = {k: v for k, v in np.load(self.distances_path).items()}
                self.distances_ = {k: v for k, v in self.distances_.items() if v.shape[0] == v.shape[1] == len(self)}
                return self.distances_
        return None

    def remove_distance(self, name):
        if self.distances is not None and name in self.distances:
            del self.distances_[name]

    def add_distance(self, name, distance_mat):
        assert distance_mat.shape[0] == distance_mat.shape[1] == len(self)
        if self.distances is None:
            self.distances_ = {name: distance_mat}
        else:
            self.distances_[name] = distance_mat

    def save_distances(self):
        if self.distances is not None:
            np.savez(self.distances_path, **self.distances)

    @classmethod
    def from_database(
        cls,
        representations=None,
        features_computer=None,
        in_memory=True,
        **dataset_build_params,
    ):
        """Run the steps to build a dataset from scratch.

        :param cls: the ``RNADataset`` class instance.
        :param representations: which representations to apply.
        :param in_memory: whetherb to store dataset in memory or on disk.
        :param dataset_build_params: parameters for the ``database_to_dataset`` function.
        """
        # if user added annotation, try to update the encoders so it can be used
        # as a feature
        dataset_path, all_rnas_name, rnas = database_to_dataset(
            features_computer=features_computer,
            return_rnas=in_memory,
            **dataset_build_params,
        )
        return cls(
            rnas=rnas,
            dataset_path=dataset_path,
            all_rnas=all_rnas_name,
            representations=representations,
            features_computer=features_computer,
            in_memory=in_memory,
        )

    def __len__(self):
        """Return the length of the current dataset."""
        return len(self.all_rnas)

    def __getitem__(self, idx):
        """Fetch one RNA and converts item from raw data to a dictionary.

        Applies representations and annotations to be used by loaders.

        :param idx: Index of dataset item to fetch.
        """
        if idx >= len(self):
            raise IndexError

        rna_name = self.all_rnas.inv[idx]
        nx_path, cif_path = None, None

        if hasattr(self, "dataset_path"):
            if self.dataset_path is not None:
                nx_path = Path(self.dataset_path) / f"{rna_name}{self.extension}"
        if hasattr(self, "structures_path"):
            if self.structures_path is not None:
                cif_path = Path(self.structures_path) / f"{rna_name}.cif"

        if self.in_memory:
            rna_graph = self.rnas[idx]
        else:
            rna_graph = load_graph(str(nx_path), multigraph=self.multigraph)
            rna_graph.name = rna_name

        # Compute features
        rna_dict = {"rna": rna_graph, "graph_path": nx_path, "cif_path": cif_path}

        if not self.transforms is None:
            self.transforms(rna_dict)

        features_dict = self.features_computer(rna_dict)
        # apply representations to the res_dict
        # each is a callable that updates the res_dict
        for rep in self.representations:
            rna_dict[rep.name] = rep(rna_graph, features_dict)
        return rna_dict

    def get_by_name(self, rna_name):
        """Grab an RNA by its pdbid"""
        rna_idx = self.all_rnas[rna_name]
        return self.__getitem__(rna_idx)

    def add_representation(self, representations: Union[List[Representation], Representation]):
        """Add a representation object to dataset.

        Provided representations are added on the fly to the dataset.

        :param representations: List of ``Representation`` objects to add.

        """
        representations = [representations] if not isinstance(representations, list) else representations
        to_print = [repr.name for repr in representations] if len(representations) > 1 else representations[0].name
        print(f">>> Adding {to_print} to dataset representations.")
        self.representations.extend(representations)

    def add_feature(
        self,
        feature: Union[str, AnnotationTransform],
        feature_level: Literal["residue", "rna"] = "residue",
        is_input: bool = True,
    ):
        """Add a feature to the dataset for model training. If you pass a string,
        we use it to pull the feature from the RNA dictionary. If you pass an AnnotationTransform,
        we check if it has been applied already , if not apply it to store the annotation in the
        dataset and then use it as a feature.

        :param feature: Can be a string representing a key in the RNA dict or an AnnotationTransform.
        :param feature_level: Residue-level (`residue`), or RNA-level (`rna`) feature.
        :param is_input: Are you using the feature on the input side (`True`) or as a prediction target (`False`)?
        """

        feature_name = feature
        custom_encoders = None
        # using an existing key in the RNA dictionary as feature
        if isinstance(feature, Transform):
            # check if transform has already been applied
            g = self[0]["rna"]
            node = next(iter(g.nodes))
            feature_exists = False
            if feature_level == "residue" and g.nodes[node].get(feature.name) is not None:
                feature_exists = True
            if feature_level == "rna" and g.graph.get(feature.name) is not None:
                feature_exists = True

            # transform not applied, so apply it
            feature(self)

            feature_name = feature.name
            custom_encoders = {feature_name: feature.encoder}

        self.features_computer.add_feature(
            feature_names=feature_name,
            feature_level=feature_level,
            input_feature=is_input,
            custom_encoders=custom_encoders,
        )
        pass

    def remove_representation(self, names):
        names = [names] if not isinstance(names, Iterable) else names
        for name in names:
            self.representations = [
                representation for representation in self.representations if representation.name != name
            ]

    def subset(self, list_of_ids=None, list_of_names=None):
        """
        Create another dataset with only the specified graphs.

        :param list_of_names: a list of rna names (no extension is expected)
        :param list_of_ids: a list of rna ids
        :return: An RNADataset with only the specified graphs/ids
        """
        # You can't subset on both simultaneously
        assert list_of_ids is None or list_of_names is None
        if list_of_names is not None:
            existing_names = set(self.all_rnas.keys())
            list_of_ids = [self.all_rnas[name] for name in list_of_names if name in existing_names]
        else:
            existing_ids = set(self.all_rnas.values())
            list_of_ids = [id_rna for id_rna in list_of_ids if id_rna in existing_ids]

        # Copy existing dataset, avoid expansive deep copy of rnas if in memory
        temp = self.rnas
        self.rnas = None
        subset = copy.deepcopy(self)
        self.rnas = temp

        # Subset the bidict of names and the rna if in_memory
        if self.in_memory:
            subset.rnas = [self.rnas[i] for i in list_of_ids]
        subset_names = [self.all_rnas.inv[i] for i in list_of_ids]
        subset.all_rnas = bidict({rna: i for i, rna in enumerate(subset_names)})

        # Update the distance matrices
        if self.distances is not None:
            for distance_name in self.distances:
                subset.add_distance(distance_name, self.distances[distance_name][np.ix_(list_of_ids, list_of_ids)])
        return subset

    def save(self, dump_path, recompute=False, verbose=True):
        """Save a local copy of the dataset"""

        print(f"dumping {len(self.all_rnas)} rnas")
        os.makedirs(dump_path, exist_ok=True)

        self.save_distances()

        with open(self.bidict_path, "w") as js:
            json.dump(dict(self.all_rnas), js)

        if os.path.exists(dump_path) and os.listdir(dump_path) and not recompute:
            if verbose:
                print('files already exist, set "recompute=True" to overwrite')
            return
        for rna_name, i in self.all_rnas.items():
            if not self.in_memory:
                # It's already saved !
                if self.dataset_path == dump_path:
                    break
                rna_graph = load_graph(os.path.join(self.dataset_path, f"{rna_name}.json"))
            else:
                rna_graph = self.rnas[i]
            dump_json(os.path.join(dump_path, f"{rna_name}.json"), rna_graph)

    def to_memory(self):
        """
        Make in_memory=True from a dataset not in memory
        :return:
        """
        self.rnas = [
            load_graph(os.path.join(self.dataset_path, f"{g_name}{self.extension}")) for g_name in self.all_rnas
        ]
        for rna, name in zip(self.rnas, self.all_rnas):
            rna.name = name

    def get_pdbid(self, pdbid):
        """Grab an RNA by its pdbid"""
        rna_idx = self.all_rnas[pdbid.lower()]
        return self.__getitem__(rna_idx)

    def check_consistency(self):
        if self.in_memory:
            assert list(self.all_rnas) == [rna.name for rna in self.rnas]


if __name__ == "__main__":
    from rnaglib.transforms import GraphRepresentation

    features_computer = FeaturesComputer(nt_features="nt_code", nt_targets="binding_protein")
    graph_rep = GraphRepresentation(framework="dgl")
    all_rnas = [
        "1a9n.json",
        "1b23.json",
        "1b7f.json",
        "1csl.json",
        "1d4r.json",
        "1dfu.json",
        "1duq.json",
        "1e8o.json",
        "1ec6.json",
        "1et4.json",
    ]
    all_rna_names = [name[:-5] for name in all_rnas]
    script_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(script_dir, "../data/test")

    # # First case
    # supervised_dataset = RNADataset(all_rnas=all_rnas,
    #                                 features_computer=features_computer,
    #                                 representations=[graph_rep])
    # g1 = supervised_dataset[0]
    # a = list(g1['rna'].nodes(data=True))[0][1]

    # This instead uses from_database, hence features_computer is called during dataset preparation, which saves spaces
    # supervised_dataset = RNADataset.from_database(all_rnas_db=all_rnas,
    #                                           features_computer=features_computer,
    #                                           representations=[graph_rep])
    # g2 = supervised_dataset[0]
    # b = list(g2['rna'].nodes(data=True))[0][1]

    # Test dumping/loading
    # rnas = build_dataset(all_rnas_db=all_rnas,
    #                      dataset_path=dataset_path,
    #                      features_computer=features_computer,
    #                      recompute=True)
    # supervised_dataset = RNADataset(dataset_path=dataset_path, representations=graph_rep)
    # g2 = supervised_dataset[0]
    # b = list(g2['rna'].nodes(data=True))[0][1]

    # supervised_dataset = RNADataset(rnas=rnas)
    # g2 = supervised_dataset[0]

    # Test in_memory field
    # supervised_dataset = RNADataset(dataset_path=dataset_path, representations=graph_rep, in_memory=False)
    # g2 = supervised_dataset[0]

    # Test subsetting
    # supervised_dataset = RNADataset(dataset_path=dataset_path, representations=graph_rep, in_memory=False)
    # subset = supervised_dataset.subset(list_of_names=all_rna_names[:5])
    # subset2 = subset.subset(list_of_ids=[1, 3, 4])

    # Test saving
    supervised_dataset = RNADataset(dataset_path=dataset_path, representations=graph_rep, in_memory=True)
    # supervised_dataset.save(os.path.join(script_dir, "../data/test_dump"))
    supervised_dataset.check_consistency()
    a = 1
