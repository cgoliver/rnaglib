import os

from pathlib import Path
from bidict import bidict
from collections.abc import Iterable
import copy

from rnaglib.data_loading.features import FeaturesComputer
from rnaglib.utils import download_graphs, load_graph, dump_json
from rnaglib.utils.graph_io import get_all_existing, get_name_extension


def build_dataset_loop(all_rnas_db, db_path, rna_filter=None, nt_filter=None, annotator=None,
                       features_computer: FeaturesComputer = None):
    """ Iterates through database, applying filters and annotations"""
    from tqdm import tqdm as tqdm
    rna_list = []

    for rna_filename in tqdm(all_rnas_db):
        rna_path = os.path.join(db_path, rna_filename)
        rna = load_graph(rna_path)

        # Remove whole systems
        if rna_filter is not None and not rna_filter(rna):
            continue

        # Apply a chunking function to whole RNAs
        if nt_filter is not None:
            subgs = []

            for subg in nt_filter(rna):
                subgs.append(subg)
        else:
            subgs = [rna]

        # Apply a per graph/subgraph function
        if annotator is not None:
            for subg in subgs:
                annotator(subg)

        # Add a 'name' field to the graphs if annotator did not put one.
        rna_name, rna_extension = get_name_extension(rna_filename)
        for i, subg in enumerate(subgs):
            if subg.name == '':
                if len(subgs) == 1:
                    subg.name = rna_name
                else:
                    subg.name = f'{rna_name}_{i}'
        assert len(subgs) == len(set([g.name for g in subgs])), ("When adding several subgraphs in nt_filter,"
                                                                 " make sure to use unique names for each subgraphs")

        # Remove useless keys
        if features_computer is not None:
            subgs = [features_computer.remove_useless_keys(subg) for subg in subgs]

        rna_list.extend(subgs)
    return rna_list


def build_dataset(dataset_path=None,
                  recompute=False,
                  all_rnas=None,
                  return_rnas=True,
                  annotator=None,
                  nt_filter=None,
                  rna_filter=None,
                  features_computer=None,
                  db_path=None,
                  all_rnas_db=None,
                  version='1.0.0',
                  download_dir=None,
                  redundancy='nr',
                  annotated=False):
    """
    Function to
    :param dataset_path: Path to an already saved dataset, skips dataset creation if loaded.
    :param recompute: Boolean if we should recompute
    :param all_rnas: A list of file names if we're using precomputed data
    :param return_rnas: If we're using existing files, shall we load them or just the file names ?

    :param db_path: The original database directory to produce our data from. If unset, further params are used. (below)
    :param all_rnas_db: If we want to only precompute over only a subset of the db
    :param redundancy: To use all graphs or just the non-redundant set.
    :param download_dir: If one changed the default download directory of rglib
    :param version: Version of the dataset to use (default='1.0.0')
    :param annotated: To use for pretraining

    :param nt_filter: Callable which takes as input an RNA dictionary and filters out some nt
    :param annotator: Callable which takes as input an RNA dictionary and adds new key-value pairs.
    :param rna_filter: Callable which takes as input an RNA dictionary and returns whether we should keep it.
    """
    # If this corresponds to a dataset that was precomputed already, just return the graphs
    if dataset_path is not None and os.path.exists(dataset_path) and not recompute:
        existing_all_rnas = get_all_existing(dataset_path=dataset_path, all_rnas=all_rnas)
        if return_rnas:
            rnas = [load_graph(os.path.join(dataset_path, g_name)) for g_name in existing_all_rnas]
            for rna, name in zip(rnas, existing_all_rnas):
                rna.name = get_name_extension(name)[0]
        else:
            rnas = None
        return dataset_path, existing_all_rnas, rnas

    # Set up the original data to build the dataset from
    # If we don't input a data path, the right one according to redundancy, chop and annotated is fetched

    # By default, we set hashing to None and potential node sim should be specified when creating
    # the node_sim function. Then, if a download occurs and no hashing was provided to the loader, the hashing used
    # is the one fetched by the downloading process to ensure it matches the data we iterate over.
    # TODO, check annotations and pretraining still work
    if db_path is None:
        db_path = download_graphs(redundancy=redundancy,
                                  version=version,
                                  annotated=annotated,
                                  data_root=download_dir)
        db_path = os.path.join(db_path, 'graphs')

    all_rnas_db = os.listdir(db_path) if all_rnas_db is None else all_rnas_db

    # If no constructions args are given, just return the graphs
    if rna_filter is None and nt_filter is None and annotator is None and features_computer is None:
        rnas = [load_graph(os.path.join(db_path, g_name)) for g_name in all_rnas_db]
        return rnas

    # If some constructions args are given, launch processing.
    rnas = build_dataset_loop(all_rnas_db=all_rnas_db,
                              db_path=db_path,
                              rna_filter=rna_filter,
                              nt_filter=nt_filter,
                              annotator=annotator,
                              features_computer=features_computer)
    all_rnas_name = [rna.name for rna in rnas]
    if dataset_path is not None:
        os.makedirs(dataset_path, exist_ok=True)
        for i, rna in enumerate(rnas):
            dump_json(os.path.join(dataset_path, f"{rna.name}.json"), rna)
    return dataset_path, all_rnas_name, rnas


class RNADataset:
    """
    This class is the main object to hold the core RNA data annotations.
    The ``RNAglibDataset.all_rnas`` object is a list of networkx objects that holds all the annotations for each RNA
    in the dataset.
    You can also access individual RNAs on-disk with ``RNAGlibDataset()[idx]`` or ``RNAGlibDataset().get_pdbid('1b23')``

    :param rnas: One can instantiate directly from a list of RNA files
    :param dataset_path: The path to the folder containing the graphs.
    :param all_rnas: In the given directory, one can choose to provide a list of graphs to use
    :param in_memory: Whether to load all RNA graphs in memory or to load them on the fly
    :param features_computer: A FeaturesComputer object, useful to transform raw RNA data into tensors.
    :param representations: List of `rnaglib.Representation` objects to apply to each item.

    The dataset holds an attribute self.all_rnas = bidict({rna_name: i for i, rna_name in enumerate(all_rna_names)})
    Where rna_name is expected to match the file name the rna should be saved in.
    """

    def __init__(self,
                 rnas=None,
                 dataset_path=None,
                 all_rnas=None,
                 in_memory=True,
                 features_computer=None,
                 representations=None):
        self.in_memory = in_memory
        if rnas is None:
            if dataset_path is None:
                # By default, use non redundant (nr), v1.0.0 dataset of rglib
                dataset_path = download_graphs()
                dataset_path = os.path.join(dataset_path, 'graphs')

            # One can restrict the number of graphs to use
            existing_all_rnas = get_all_existing(dataset_path=dataset_path, all_rnas=all_rnas)
            if in_memory:
                self.rnas = [load_graph(os.path.join(dataset_path, g_name)) for g_name in existing_all_rnas]
            else:
                self.rnas = None
                self.dataset_path = dataset_path

            # Keep track of a list_id <=> system mapping. First remove extensions
            existing_all_rna_names = [get_name_extension(rna, permissive=True)[0] for rna in existing_all_rnas]
            self.all_rnas = bidict({rna: i for i, rna in enumerate(existing_all_rna_names)})
        else:
            assert in_memory, ("Conflicting arguments: if an RNADataset is instantiated with a list of graphs, "
                               "it must use 'in_memory=True'")
            self.rnas = rnas

            # Here we assume that rna lists contain a relevant rna.name field, which is the case
            # if it was constructed using build_dataset above
            rna_names = set([rna.name for rna in rnas])
            assert '' not in rna_names and len(rna_names) == len(rnas), ("When creating a RNAdataset from rnas, please "
                                                                         "use uniquely named networkx graphs")
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

    @classmethod
    def from_args(cls,
                  representations=None,
                  features_computer=None,
                  in_memory=True,
                  **dataset_build_params):
        dataset_path, all_rnas_name, rnas = build_dataset(features_computer=features_computer,
                                                          return_rnas=in_memory,
                                                          **dataset_build_params)
        return cls(rnas=rnas,
                   dataset_path=dataset_path,
                   all_rnas=all_rnas_name,
                   representations=representations,
                   features_computer=features_computer,
                   in_memory=in_memory)

    def __len__(self):
        return len(self.all_rnas)

    def __getitem__(self, idx):
        """ Fetches one RNA and converts it from raw data to a dictionary
        with representations and annotations to be used by loaders """
        if self.in_memory:
            rna_graph = self.rnas[idx]
        else:
            rna_name = self.all_rnas.inv[idx]
            rna_graph = load_graph(os.path.join(self.dataset_path, f"{rna_name}.json"))

        # Compute features
        features_dict = self.features_computer.compute_features(rna_graph)
        rna_dict = {'rna': rna_graph}

        # apply representations to the res_dict
        # each is a callable that updates the res_dict
        for rep in self.representations:
            rna_dict[rep.name] = rep(rna_graph, features_dict)
        return rna_dict

    def add_representation(self, representations):
        representations = [representations] if not isinstance(representations, list) else representations
        self.representations.extend(representations)

    def remove_representation(self, names):
        names = [names] if not isinstance(names, Iterable) else names
        for name in names:
            self.representations = [representation for representation in self.representations if
                                    representation.name != name]

    def subset(self, list_of_ids=None, list_of_names=None):
        """
        Create another dataset with only the specified graphs

        :param list_of_names: a list of rna names (no extension is expected)
        :param list_of_ids: a list of rna ids
        :return: An RNADataset with only the specified graphs/ids
        """
        # You can't subset on both simultaneously
        assert list_of_ids is None or list_of_names is None
        if list_of_names is not None:
            list_of_ids = [self.all_rnas[name] for name in list_of_names]

        # Copy existing dataset, subset the bidict of names and the rna if in_memory
        subset = copy.deepcopy(self)
        if self.in_memory:
            subset.rnas = [self.rnas[i] for i in list_of_ids]
        subset_names = [self.all_rnas.inv[i] for i in list_of_ids]
        subset.all_rnas = bidict({rna: i for i, rna in enumerate(subset_names)})
        return subset

    def save(self, dump_path, recompute=False):
        """ Save a local copy of the dataset"""
        if os.path.exists(dump_path) and not recompute:
            return
        os.makedirs(dump_path, exist_ok=True)
        for rna_name, i in self.all_rnas.items():
            if not self.in_memory:
                rna_graph = load_graph(os.path.join(self.dataset_path, f"{rna_name}.json"))
            else:
                rna_graph = self.rnas[i]
            dump_json(os.path.join(dump_path, f"{rna_name}.json"), rna_graph)

    def get_pdbid(self, pdbid):
        """ Grab an RNA by its pdbid """
        rna_idx = self.all_rnas[pdbid.lower()]
        return self.__getitem__(rna_idx)

    def check_consistency(self):
        if self.in_memory:
            assert list(self.all_rnas) == [rna.name for rna in self.rnas]


if __name__ == '__main__':
    from rnaglib.representations import GraphRepresentation

    features_computer = FeaturesComputer(nt_features='nt_code', nt_targets='binding_protein')
    graph_rep = GraphRepresentation(framework='dgl')
    all_rnas = ['1a9n.json', '1b23.json', '1b7f.json', '1csl.json', '1d4r.json', '1dfu.json', '1duq.json',
                '1e8o.json', '1ec6.json', '1et4.json']
    all_rna_names = [name[:-5] for name in all_rnas]
    script_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(script_dir, "../data/test")

    # # First case
    # supervised_dataset = RNADataset(all_rnas=all_rnas,
    #                                 features_computer=features_computer,
    #                                 representations=[graph_rep])
    # g1 = supervised_dataset[0]
    # a = list(g1['rna'].nodes(data=True))[0][1]

    # This instead uses from_args, hence features_computer is called during dataset preparation, which saves spaces
    # supervised_dataset = RNADataset.from_args(all_rnas_db=all_rnas,
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
