from typing import List, Any, Tuple
from collections import defaultdict
import random

from rnaglib.utils import load_index

graph_index = load_index()


def split_list_in_fractions(list_to_split: List[Any],
                            split_train: float = 0.7,
                            split_valid:  float = 0.15,
                            seed: int = 0) -> Tuple[List[Any], List[Any], List[Any]]:
    """ Split a list and return sub-lists by given fractions split and validation. The remainder of the
    dataset is used for the test set.

    :param list_to_split: list you want to split.
    :param split_train: fraction of dataet to use for train set
    :param split_valid: fraction of dataset to use for validation
    """
    copy_list = list_to_split.copy()
    random.Random(seed).shuffle(copy_list)

    train_index, valid_index = int(split_train * len(copy_list)), int(split_valid * len(copy_list))

    train_list = copy_list[:train_index]
    valid_list = copy_list[train_index:train_index + valid_index]
    test_list = copy_list[train_index+valid_index:]
    return train_list, valid_list, test_list


def random_split(dataset, split_train=0.7, split_valid=0.15, seed=0):
    """
    Just randomly split a dataset
    :param dataset:
    :param split_train:
    :param split_valid:
    :return:
    """
    indices = list(range(len(dataset)))
    train_indices, valid_indices, test_indices = split_list_in_fractions(indices,
                                                                         split_train=split_train,
                                                                         split_valid=split_valid,
                                                                         seed=seed)

    return train_indices, valid_indices, test_indices


def get_multitask_split(node_targets, graph_index=graph_index, split_train=0.7, split_valid=0.85):
    """
    :param node_targets: A subset of {'binding_protein', 'binding_small-molecule', 'is_modified', 'binding_ion'}
    :param graph_index: should be the opened output of the previous function a dict of dict of dict.
    :param target_fraction: The fraction of each task to have in the test set

    Correctly splitting the data for multitasking is hard,
    For instance in a triangle situation AB,BC,CD,DA : we can split in halves along each dimension but not along
    two at the same time

    This is a very basic, greedy version of data splitting for multi-task:
    1. Count the amount of nodes for each attrs
    2. Iterate through graphs
         -> Populate current split if the graph contains one of queried attributes. Keep track of how many attributes
         are in the current split compared to the overall number. If this number (attrs_fraction) exceeds the query
         fraction, this attribute is no longer sought after for this split
         -> If no attribute is left to be sought, save the current split and move on to the next one.

    :return: the splits in the form of a list of graphs.
    """
    # First count all occurences :
    total_counts = defaultdict(int)
    for graph, graph_attrs in graph_index.items():
        for graph_attrs_name, graph_attrs_counter in graph_attrs.items():
            if graph_attrs_name in node_targets:
                # Maybe there is something to be made here, but usually it's just absent from the encoding
                # So summing all values in counter makes sense
                total_counts[graph_attrs_name] += sum(graph_attrs_counter.values())

    # Filled splits store the results, query_attrs_insplit counts how many positive nodes are already in the split for
    # each property and query_attrs is just the list of targets that still need to be found in this split
    filled_splits = []
    query_attrs_insplit = defaultdict(int)
    query_attrs_copy = node_targets.copy()

    # Then iterate again and stop after having filled all queried sets following the queried fractions
    fractions = (1 - split_valid, split_valid - split_train)
    fractions = [frac for frac in fractions if frac > 0]
    current_fraction = fractions.pop()
    currrent_split = set()
    for graph, graph_attrs in graph_index.items():
        for graph_attrs_name, graph_attrs_counter in graph_attrs.items():
            if graph_attrs_name in query_attrs_copy:
                # Now add this graph and update the splits
                currrent_split.add(graph)
                # total_nodes_in_split += len(graph.nodes()) TODO get the number of nodes per graph
                query_attrs_insplit[graph_attrs_name] += sum(graph_attrs_counter.values())
                attrs_fraction = float(query_attrs_insplit[graph_attrs_name]) / total_counts[graph_attrs_name]
                if attrs_fraction > current_fraction:
                    query_attrs_copy.remove(graph_attrs_name)
        # If we found everything we needed for this split, save it and reset relevant variables
        if len(query_attrs_copy) == 0:
            filled_splits = [list(currrent_split)] + filled_splits
            if len(fractions) == 0:
                break
            # Get ready to search for another split
            query_attrs_insplit = defaultdict(int)
            query_attrs_copy = node_targets.copy()
            current_fraction = fractions.pop()
            currrent_split = set()

    all_but_train = set().union(*filled_splits)
    train_split = set(graph_index.keys()) - all_but_train
    assert len(train_split) > 0
    all_splits = [list(train_split)] + filled_splits
    return all_splits


def get_single_task_split(node_target, graph_index=graph_index, split_train=0.7, split_valid=0.85):
    all_list = []
    for graph, graph_attrs in graph_index.items():
        if node_target in graph_attrs:
            all_list.append(graph)

    train_graphs, valid_graphs, test_graphs = split_list_in_fractions(all_list,
                                                                      split_train=split_train,
                                                                      split_valid=split_valid)

    return train_graphs, valid_graphs, test_graphs


def split_dataset(dataset, split_train=0.7, split_valid=0.85):
    node_targets = [f"node_{target}" for target in dataset.nt_targets]
    # 1st strategy : if we are looking for a single property : subset the graphs that contain at least a node with this
    # property and make a random split among these.
    if len(node_targets) == 1:
        train_split, validation_split, test_split = get_single_task_split(node_targets[0],
                                                                          split_train=split_train,
                                                                          split_valid=split_valid)
    # 2nd strategy : for multitask objective, we could also subset, but a random split could end up with one of
    # the categories missing from a set
    else:
        train_split, validation_split, test_split = get_multitask_split(node_targets=node_targets,
                                                                        split_train=split_train,
                                                                        split_valid=split_valid)
    train_set = dataset.subset(train_split)
    validation_set = dataset.subset(validation_split)
    test_set = dataset.subset(test_split)
    return train_set, validation_set, test_set
