from collections import defaultdict
import random

from rnaglib.data_loading import DEFAULT_INDEX


def get_multitask_split(node_targets, graph_index=DEFAULT_INDEX, fractions=(0.15, 0.15)):
    """
    :param node_targets: A subset of {'binding_protein', 'binding_small-molecule', 'is_modified', 'binding_ion'}
    :param graph_index: should be the opened output of the previous function a dict of dict of dict.
    :param target_fraction: The fraction of each task to have in the test set

    Correctly splitting the data for multitasking is hard,
    For instance in a triangle situation AB,BC,CD,DA : we can split in halves along each dimension but not along
    two at the same time
    This is a very basic, greedy version of data splitting for multi task where we first count the amount of nodes
     for each attrs and we then fill a test split.

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

    # Then iterate again and stop after having filled all queried sets following the queried fractions
    splits_to_fill = [set() for _ in range(len(fractions))]
    filled_splits = []
    fractions = list(fractions)
    current_fraction = fractions.pop()
    currrent_split = splits_to_fill.pop()

    query_attrs_insplit = defaultdict(int)
    copy_query_attrs = node_targets.copy()

    for graph, graph_attrs in graph_index.items():
        for graph_attrs_name, graph_attrs_counter in graph_attrs.items():
            if graph_attrs_name in copy_query_attrs:
                # Now add this graph and update the splits
                currrent_split.add(graph)
                # total_nodes_in_split += len(graph.nodes()) TODO get the number of nodes per graph
                query_attrs_insplit[graph_attrs_name] += sum(graph_attrs_counter.values())
                attrs_fraction = float(query_attrs_insplit[graph_attrs_name]) / total_counts[graph_attrs_name]
                if attrs_fraction > current_fraction:
                    copy_query_attrs.remove(graph_attrs_name)
        # If we found everything we needed for this split, save it and reset relevant variables
        if len(copy_query_attrs) == 0:
            filled_splits = [list(currrent_split)] + filled_splits
            query_attrs_insplit = defaultdict(int)
            copy_query_attrs = node_targets.copy()
            if len(fractions) == 0:
                break
            current_fraction = fractions.pop()
            currrent_split = splits_to_fill.pop()

    all_but_train = set().union(*filled_splits)
    train_split = set(graph_index.keys()) - all_but_train
    all_splits = [list(train_split)] + filled_splits
    return all_splits


def get_single_task_split(node_target, graph_index=DEFAULT_INDEX, split_train=0.7, split_valid=0.85):
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
        train_set = dataset.subset(train_split)
        validation_set = dataset.subset(validation_split)
        test_set = dataset.subset(test_split)
        return train_set, validation_set, test_set

    # 2nd strategy : for multitask objective, we could also subset, but a random split could end up with one of
    # the categories missing from a set
    fractions = (1 - split_valid, split_valid - split_train)
    train_split, validation_split, test_split = get_multitask_split(node_targets=node_targets,
                                                                    fractions=fractions)

    train_set = dataset.subset(train_split)
    validation_set = dataset.subset(validation_split)
    test_set = dataset.subset(test_split)
    return train_set, validation_set, test_set


def split_dataset_in_fractions(dataset, split_train=0.7, split_valid=0.85):
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
                                                                         split_valid=split_valid)

    train_set = Subset(dataset, train_indices)
    valid_set = Subset(dataset, valid_indices)
    test_set = Subset(dataset, test_indices)
    return train_set, valid_set, test_set


def split_list_in_fractions(list_to_split, split_train=0.7, split_valid=0.85):
    copy_list = list_to_split.copy()
    random.shuffle(copy_list)

    train_index, valid_index = int(split_train * len(copy_list)), int(split_valid * len(copy_list))

    train_list = copy_list[:train_index]
    valid_list = copy_list[train_index:valid_index]
    test_list = copy_list[valid_index:]
    return train_list, valid_list, test_list
