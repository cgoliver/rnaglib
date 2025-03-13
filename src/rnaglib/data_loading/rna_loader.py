import torch
from torch.utils.data import DataLoader

from rnaglib.dataset_transforms import split_dataset
from rnaglib.config.graph_keys import EDGE_MAP_RGLIB_REVERSE


class Collater:
    """
        Wrapper for collate function, so we can use different node similarities.
        We cannot use functools.partial as it is not picklable so incompatible with Pytorch loading
    """

    def __init__(self, dataset):
        """ Initialize a Collater object.
        :return: a picklable python function that can be called on a batch by Pytorch loaders
        """
        self.dataset = dataset

    def __call__(self, samples):
        """
        :param samples:
        :return: a dict

        """
        batch = dict()
        for representation in self.dataset.representations:
            representation_samples = [sample.pop(representation.name) for sample in samples]
            batched_representation = representation.batch(representation_samples)
            batch[representation.name] = batched_representation
        remaining_keys = set(samples[0].keys())
        for key in remaining_keys:
            batch[key] = [sample[key] for sample in samples]
        return batch


def get_loader(dataset,
               batch_size=5,
               num_workers=0,
               split=True,
               split_train=0.7,
               split_valid=0.85,
               verbose=False,
               persistent_workers=True,
               ):
    """ Fetch a loader object for a given dataset.

    :param rnaglib.data_loading.RNADataset dataset: Dataset for loading.
    :param int batch_size: number of items in batch
    :param bool split: whether to compute splits
    :param float split_train: proportion of dataset to keep for training
    :param float split_valid: proportion of dataset to keep for validation
    :param bool verbose: print updates
    :return: torch.utils.data.DataLoader

    """
    persistent_workers = False if num_workers == 0 else persistent_workers

    collater = Collater(dataset=dataset)
    if not split:
        loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size,
                            num_workers=num_workers, collate_fn=collater, persistent_workers=persistent_workers)
        return loader

    else:
        train_set, valid_set, test_set = split_dataset(dataset, split_train=split_train, split_valid=split_valid)

        if verbose:
            print(f"training items: ", len(train_set))

        def safe_loader_creation(**kwargs):
            """
            Just a small util wrapper to avoid raising errors on empty sets that can arise by splitting
            :param kwargs:
            :return:
            """
            num_items = len(kwargs['dataset'])
            if num_items > 0:
                return DataLoader(**kwargs)
            else:
                return None

        train_loader = safe_loader_creation(dataset=train_set, shuffle=True, batch_size=batch_size,
                                            num_workers=num_workers, collate_fn=collater,
                                            persistent_workers=persistent_workers)
        valid_loader = safe_loader_creation(dataset=valid_set, shuffle=True, batch_size=batch_size,
                                            num_workers=num_workers, collate_fn=collater,
                                            persistent_workers=persistent_workers)
        test_loader = safe_loader_creation(dataset=test_set, shuffle=True, batch_size=batch_size,
                                           num_workers=num_workers, collate_fn=collater,
                                           persistent_workers=persistent_workers)
        return train_loader, valid_loader, test_loader


def get_inference_loader(list_to_predict,
                         data_path=None,
                         dataset=None,
                         batch_size=5,
                         num_workers=20,
                         **kwargs):
    """
    This is to just make an inference over a list of graphs.
    """
    if (dataset is None and data_path is None) or (dataset is not None and data_path is not None):
        raise ValueError("To create an inference loader please provide either an existing dataset or a data path")

    subset = dataset.subset(list_to_predict)
    collater = Collater()
    train_loader = DataLoader(dataset=subset,
                              shuffle=False,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              collate_fn=collater)
    return train_loader


class EdgeLoaderGenerator:
    """
        This turns a graph dataloader or dataset into an edge data loader generator.
        It needs to be reinitialized every epochs because of the double iteration pattern

        Iterates over batches of base pairs and generates negative samples for each.
        Negative sampling is just uniform for the moment (eventually we should change it to only sample
        edges at a certain backbone distance.

        timing :
        - num workers should be used to load the graphs not in the inner loop
        - The inner batch size yields huge speedups (probably generating all MFGs is tedious)
    """

    def __init__(self,
                 graph_loader,
                 inner_batch_size=50,
                 sampler_layers=2,
                 neg_samples=1):
        """ Initialize the loader.
        
        :param graph_loader: A GraphLoader or GraphDataset. We will iterate over its graphs and then over its basepairs
        :param inner_batch_size: The amount of base-pairs to sample in each batch on each graph
        :param sampler_layers: The size of the neighborhood
        :param neg_samples: The number of negative sample to use per positive ones
        """
        self.graph_loader = graph_loader
        self.neg_samples = neg_samples
        self.sampler_layers = sampler_layers
        self.inner_batch_size = inner_batch_size
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.sampler_layers)
        self.negative_sampler = dgl.dataloading.negative_sampler.Uniform(self.neg_samples)
        self.eloader_args = {
            'shuffle': False,
            'batch_size': self.inner_batch_size,
            'negative_sampler': self.negative_sampler
        }

    @staticmethod
    def get_base_pairs(g):
        """
        Get edge IDS of edges in a base pair (non-backbone or unpaired).

        :param g: networkx graph
        :return: list of ids
        """
        eids = []
        for ind, e in enumerate(g.edata['edge_type']):
            if EDGE_MAP_RGLIB_REVERSE[e.item()][0] != 'B':
                eids.append(e)
        return eids

    def get_edge_loader(self):
        """
        Simply get the loader for one epoch. This needs to be called at each epoch

        :return: the edge loader
        """

        if DGL_VERSION < 1.8:
            from dgl.dataloading.pytorch import EdgeDataLoader
            edge_loader = (EdgeDataLoader(g_batched, self.get_base_pairs(g_batched), self.sampler, **self.eloader_args)
                           for g_batched, _ in self.graph_loader)
        else:
            sampler = dgl.dataloading.as_edge_prediction_sampler(
                self.sampler,
                negative_sampler=self.negative_sampler)
            edge_loader = (DGLDataLoader(g_batched,
                                         self.get_base_pairs(g_batched),
                                         sampler,
                                         shuffle=False,
                                         batch_size=self.inner_batch_size)
                           for g_batched, _ in self.graph_loader)
        return edge_loader


class DefaultBasePairLoader:
    """ Dataloader that yields base pairs """

    def __init__(self,
                 dataset=None,
                 data_path=None,
                 batch_size=5,
                 inner_batch_size=50,
                 sampler_layers=2,
                 neg_samples=1,
                 num_workers=4,
                 framework='dgl',
                 **kwargs):
        """
        Just a default edge base pair loader that deals with the splits

        :param dataset: A GraphDataset we want to loop over for base-pair prediction
        :param data_path: Optionnaly, we can use a data path to create a default GraphDataset
        :param batch_size: The desired batch size (number of whole graphs)
        :param inner_batch_size:The desired inner batch size (number of sampled edge in a batched graph)
        :param sampler_layers: The size of the neighborhood
        :param neg_samples: The number of negative sample to use per positive ones
        :param framework: which learning framework to use
        :param num_workers: The number of cores to use for loading
        """
        # Create default loaders
        if dataset is None:
            dataset = GraphDataset(data_path=data_path, **kwargs)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.g_train, self.g_val, self.g_test = get_loader(self.dataset,
                                                           batch_size=self.batch_size,
                                                           num_workers=self.num_workers)

        # Get the inner loader parameters
        self.inner_batch_size = inner_batch_size
        self.neg_samples = neg_samples
        self.sampler_layers = sampler_layers
        self.framework = framework

    def get_data(self):
        if self.framework == 'dgl':
            import dgl

            DGL_VERSION = dgl.__version__
            if DGL_VERSION < "0.8":
                from dgl.dataloading.pytorch import EdgeDataLoader
            else:
                from dgl.dataloading import DataLoader as DGLDataLoader

            train_loader = EdgeLoaderGenerator(graph_loader=self.g_train, inner_batch_size=self.inner_batch_size,
                                               sampler_layers=self.sampler_layers,
                                               neg_samples=self.neg_samples).get_edge_loader()
            val_loader = EdgeLoaderGenerator(graph_loader=self.g_val, inner_batch_size=self.inner_batch_size,
                                             sampler_layers=self.sampler_layers,
                                             neg_samples=self.neg_samples).get_edge_loader()
            test_loader = EdgeLoaderGenerator(graph_loader=self.g_test, inner_batch_size=self.inner_batch_size,
                                              sampler_layers=self.sampler_layers,
                                              neg_samples=self.neg_samples).get_edge_loader()

        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    pass
    node_features = ['nt_code', "alpha", "C5prime_xyz", "is_modified"]
    # node_features = None
    node_target = ['binding_ion', 'binding_protein']
    # node_target = None
    # node_simfunc = SimFunctionNode(method='R_1', depth=2)
    node_simfunc = None

    torch.random.manual_seed(42)

    from rnaglib.representations import GraphRepresentation, RingRepresentation
    from rnaglib.data_loading import RNADataset, FeaturesComputer
    from rnaglib.kernels import node_sim

    # GET THE DATA GOING
    features_computer = FeaturesComputer(nt_features=node_features, nt_targets=node_target)
    graph_rep = GraphRepresentation(framework='dgl')
    # ring_rep = RingRepresentation(node_simfunc=node_simfunc, max_size_kernel=None)

    toy_dataset = RNADataset(features_computer=features_computer, representations=graph_rep)
    train_loader, validation_loader, test_loader = get_loader(dataset=toy_dataset,
                                                              batch_size=3,
                                                              num_workers=0,
                                                              framework='dgl'
                                                              )

    for i, batch in enumerate(train_loader):
        for k, v in batch.items():
            if 'voxel' in k:
                print(k, [value.shape for value in v])
        if i > 10:
            break
        # if not i % 20: print(i)
        pass
