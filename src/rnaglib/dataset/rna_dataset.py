"""Main class for collections of RNAs."""

from collections.abc import Iterable
import copy
import json
import os
from pathlib import Path
from typing import Literal, Union
from torch.utils.data import Dataset

from bidict import bidict
import networkx as nx
import numpy as np

from rnaglib.transforms.featurize import FeaturesComputer
from rnaglib.transforms.represent import Representation
from rnaglib.transforms.transform import AnnotationTransform, Transform
from rnaglib.utils import download_graphs, dump_json, load_graph
from rnaglib.utils.graph_io import get_all_existing, get_name_extension


class RNADataset(Dataset):
    """This class is the main object to hold RNA data, and is compatible with Pytorch Dataset.

    A key feature is a bidict that holds a mapping between RNA names and their index in the dataset.
    This allows for constant time access to an RNA with a given name.

    The RNAs contained in an RNADataset can either live in the memory, or be specified as files.
    In the latter case, an RNADataset can be seen as an ordered list of file in a given directory.
    One can put a dataset in memory by calling to_memory()

    Once a dataset is built, you can save it, subset it and access its elements by name or by index.

    :param rnas: For use in memory, list of RNA objects represented as networkx graphs.
    :param dataset_path: If using filenames, this is the path to the folder containing the graphs to load.
    :param version: If using filenames, and no dataset_path is provided, this is the version of the RNA dataset
     download that will be used, and set as dataset_path
    :param redundancy: same as version, sets the redundancy mode to use if neither rnas nor dataset_path is provided.
    :param rna_id_subset: List of graphs filenames to grab in the dataset_path to keep instead of using all available.
    :param recompute_mapping: When loading a dataset, you can choose to use an existing bidict_mapping
    (for instance if some graphs are irrelevant)
    :param in_memory: When loading a dataset from files, you can choose to load the data in memory by setting
    in memory to true
    :param debug: if True, will only report 50 items
    :param get_pdbs: if True, will also fetch the corresponding structures.
    :param multigraph: Whether to load RNAs as multi-graphs or simple graphs. Multigraphs can have
                      backbone and base pairs between the same two residues.
    :param transforms: An optional list of transforms to apply to rnas before calling the features computer and
    the representations in get_item
    :param features_computer: A FeaturesComputer object, useful to transform raw RNA data into tensors.
    :param representations: List of :class:`~rnaglib.representations.Representation` objects to
                          apply to each item.

    Examples:
    ---------
    Create a default dataset::
    >>> from rnaglib.dataset import RNADataset
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
            rnas: list[nx.Graph] = None,
            dataset_path: Union[str, os.PathLike] = None,
            version="2.0.2",
            redundancy="nr",
            rna_id_subset: list[str] = None,
            recompute_mapping: bool = True,
            in_memory: bool = None,
            features_computer: FeaturesComputer = None,
            representations: Union[list[Representation] , Representation] = None,
            debug: bool = False,
            get_pdbs: bool = True,
            multigraph: bool = False,
            transforms: Union[list[Transform], Transform] = None,
    ):
        self.transforms = [transforms] if transforms is not None and not isinstance(transforms, Iterable) else []
        self.multigraph = multigraph
        self.version = version

        if dataset_path is not None:
            self.dataset_path = dataset_path

        # Distance is computed as a cached property
        # We potentially want to save distances and the bidict mapping
        self.distances_ = None
        self.distances_path = Path(dataset_path) / "distances.npz" if dataset_path is not None else None
        self.bidict_path = Path(dataset_path) / "bidict.json" if dataset_path is not None else None
        if rnas is None:
            if dataset_path is None:
                # By default, use non redundant (nr), v1.0.0 dataset of rglib
                dataset_path, structures_path = download_graphs(redundancy=redundancy,
                                                                version=version,
                                                                debug=debug,
                                                                get_pdbs=get_pdbs)
                self.dataset_path = dataset_path
                self.structures_path = structures_path

            # One can restrict the number of graphs to use
            existing_all_rnas, extension = get_all_existing(dataset_path=self.dataset_path, all_rnas=rna_id_subset)
            self.extension = extension
            if recompute_mapping or not self.bidict_path.exists():
                # Keep track of a list_id <=> system mapping. First remove extensions
                existing_all_rna_names = [get_name_extension(rna, permissive=True)[0] for rna in existing_all_rnas]
                self.all_rnas = bidict({rna: i for i, rna in enumerate(existing_all_rna_names)})

            else:
                with self.bidict_path.open() as f:
                    simple_dict = json.load(f)
                self.all_rnas = bidict(simple_dict)

            # If debugging, only keep the first few
            if debug:
                nb_items_to_keep = min(50, len(self.all_rnas))
                self.all_rnas = bidict(
                    {rna: i for idx, (rna, i) in enumerate(self.all_rnas.items()) if idx < nb_items_to_keep})

            if in_memory is not None and in_memory:
                self.to_memory()
            else:
                self.rnas = None
                self.in_memory = False
        else:
            # handle default choice
            self.in_memory = in_memory if in_memory is not None else True
            assert self.in_memory, (
                "Conflicting arguments: if an RNADataset is instantiated with a list of graphs, "
                "it must use 'in_memory=True'"
            )
            self.rnas = rnas
            self.structures_path = None

            # Here we assume that rna lists contain a relevant rna.name field, which is the case for defaults
            rna_names = {rna.name for rna in rnas}
            assert "" not in rna_names, "Empty RNA name found"
            assert len(rna_names) == len(
                rnas,
            ), "When creating a RNAdataset from rnas, please use uniquely named networkx graphs"
            self.all_rnas = bidict({rna.name: i for i, rna in enumerate(rnas)})

        # Now that we have the raw data setup, let us set up the features we want to be using:
        self.features_computer = FeaturesComputer() if features_computer is None else features_computer

        # Finally, let us set up the list of representations that we will be using
        if representations is None:
            self.representations = []
        elif not isinstance(representations, list):
            self.representations = [representations]
        else:
            self.representations = representations

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

        # Recover rna name from passed index.
        rna_name = self.all_rnas.inv[idx]
        # Initialise paths
        nx_path, cif_path = None, None

        # Setting path to default path if no other is specified.
        if getattr(self, "dataset_path", None) is not None:
            nx_path = Path(self.dataset_path) / f"{rna_name}{self.extension}"

        if getattr(self, "structures_path", None) is not None:
            cif_path = Path(self.structures_path) / f"{rna_name}.cif"

        if self.in_memory:
            rna_graph = self.rnas[idx]
        else:
            rna_graph = load_graph(str(nx_path), multigraph=self.multigraph)
            rna_graph.name = rna_name

        # Compute features
        rna_dict = {"rna": rna_graph, "graph_path": nx_path, "cif_path": cif_path}

        if len(self.transforms) > 0:
            for transform in self.transforms:
                transform(rna_dict)

        features_dict = self.features_computer(rna_dict)
        # apply representations to the res_dict
        # each is a callable that updates the res_dict
        for rep in self.representations:
            rna_dict[rep.name] = rep(rna_graph, features_dict)
        return rna_dict

    def get_by_name(self, rna_name):
        """Grab an RNA by its name."""
        rna_idx = self.all_rnas[rna_name]
        return self.__getitem__(rna_idx)

    def get_pdbid(self, pdbid):
        """Grab an RNA by its pdbid. Just a shortcut that includes a lower() call"""
        return self.get_by_name(pdbid.lower())

    def to_memory(self):
        """Make in_memory=True from a dataset not in memory."""
        if not hasattr(self, "rnas"):
            self.rnas = [load_graph(Path(self.dataset_path) / f"{g_name}{self.extension}") for g_name in self.all_rnas]
            for rna, name in zip(self.rnas, self.all_rnas, strict=False):
                rna.name = name
            self.in_memory = True

    def check_consistency(self):
        """Make sure all RNAs actually present when in_memory is true."""
        if self.in_memory:
            assert list(self.all_rnas) == [rna.name for rna in self.rnas]
        else:
            print("Check consistency only works if in_memory is true.")

    @property
    def distances(self):
        """Using a cached property is useful for loading precomputed data.

        If this is the first call, try loading. Otherwise, return the loaded value
        """
        if self.distances_ is not None:
            return self.distances_
        if self.distances_path is not None and Path(self.distances_path).exists():
            # Actually materialize memory (lightweight anyway) since npz loading is lazy
            all_distances = dict(np.load(self.distances_path).items())
            # Filter to keep only square matrices with dimensions matching our dataset length
            self.distances_ = {k: v for k, v in all_distances.items() if v.shape[0] == v.shape[1] == len(self)}
            return self.distances_
        return None

    def remove_distance(self, name):
        """Removes a distance from the dataset."""
        if self.distances is not None and name in self.distances:
            del self.distances_[name]

    def add_distance(self, name, distance_mat):
        """Adds a distance matrix to the dataset."""
        assert distance_mat.shape[0] == distance_mat.shape[1] == len(self)
        if self.distances is None:
            self.distances_ = {name: distance_mat}
        else:
            self.distances_[name] = distance_mat

    def save_distances(self, dump_path=None):
        """Saves distances to distance path."""
        if self.distances is not None:
            dump_path = dump_path if dump_path is not None else self.distances_path
            np.savez(dump_path, **self.distances)

    def add_representation(self, representations: Union[list[Representation],
                                                         Representation]):
        """Add a representation object to dataset.

        Provided representations are added on the fly to the dataset.

        :param representations: List of ``Representation`` objects to add.

        """
        representations = [representations] if not isinstance(representations, list) else representations
        to_print = [repre.name for repre in representations] if len(representations) > 1 else representations[0].name
        print(f">>> Adding {to_print} to dataset representations.")

        # Remove old representations with the same name
        new_representations = set([repre.name for repre in representations])
        to_remove = {repr.name for repr in self.representations if repr.name in new_representations}
        if len(to_remove) > 0:
            print(f"Removing old representations of {to_remove} from existing representation to avoid clash")
            self.representations = [repr for repr in self.representations if not repr.name in to_remove]
        self.representations.extend(representations)

    def remove_representation(self, names):
        """Removes specified representation."""
        names = [names] if not isinstance(names, Iterable) else names
        for name in names:
            self.representations = [
                representation for representation in self.representations if representation.name != name
            ]

    def add_feature(
            self,
            feature: Union[str, AnnotationTransform],
            feature_level: Literal["residue", "rna"] = "residue",
            *,  # enforce keyword only arguments
            is_input: bool = True,
    ):
        """Add a feature to the dataset for model training.

        If you pass a string,
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

            # Only apply transform if it hasn't been applied yet
            if not feature_exists:
                feature(self)

            feature_name = feature.name
            custom_encoders = {feature_name: feature.encoder}

        self.features_computer.add_feature(
            feature_names=feature_name,
            feature_level=feature_level,
            input_feature=is_input,
            custom_encoders=custom_encoders,
        )

    def subset(self, list_of_ids=None, list_of_names=None):
        """Create another dataset with only the specified graphs.

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

        # Copy existing dataset, avoid expensive deep copy of rnas if in memory
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

    def save(self, dump_path, *, recompute=False, verbose=True):
        """Save a local copy of the dataset."""
        print(f"dumping {len(self.all_rnas)} rnas")
        Path(dump_path).mkdir(parents=True, exist_ok=True)

        dump_dists = Path(dump_path) / "distances.npz"
        dump_bidict = Path(dump_path) / "bidict.json"
        self.save_distances(dump_path=dump_dists)

        with open(dump_bidict, "w") as js:
            json.dump(dict(self.all_rnas), js)

        # Check if all files are already there
        existing_files = set(os.listdir(dump_path))
        to_dump = set([x + '.json' for x in self.all_rnas.keys()])
        if to_dump.issubset(existing_files) and not recompute:
            if verbose:
                print('files already exist, set "recompute=True" to overwrite')
        for rna_name, i in self.all_rnas.items():
            if not self.in_memory:
                rna_graph = load_graph(Path(self.dataset_path) / f"{rna_name}.json")
            else:
                rna_graph = self.rnas[i]
            dump_json(Path(dump_path) / f"{rna_name}.json", rna_graph)


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
    script_dir = Path(__file__).resolve().parent
    dataset_path = script_dir / "../data/test"

    # # First case
    # supervised_dataset = RNADataset(all_rnas=all_rnas,
    #                                 features_computer=features_computer,
    #                                 representations=[graph_rep])
    # g1 = supervised_dataset[0]
    # a = list(g1['rna'].nodes(data=True))[0][1]

    # Test in_memory field
    # supervised_dataset = RNADataset(dataset_path=dataset_path, representations=graph_rep, in_memory=False)
    # g2 = supervised_dataset[0]

    # Test subsetting
    # supervised_dataset = RNADataset(dataset_path=dataset_path, representations=graph_rep, in_memory=False)
    # subset = supervised_dataset.subset(list_of_names=all_rna_names[:5])
    # subset2 = subset.subset(list_of_ids=[1, 3, 4])

    # Test saving
    # supervised_dataset = RNADataset(dataset_path=dataset_path, representations=graph_rep, in_memory=True)
    # supervised_dataset.save(os.path.join(script_dir, "../data/test_dump"))
    # supervised_dataset.check_consistency()
