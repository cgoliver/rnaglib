"""Small helper functions for the different splitting strategies"""

import random
from collections import Counter
from typing import Any


def label_counter(dataset):
    """Count the number of labels in a dataset and return the total counts.

    Handles both node-level (nt_targets) and graph-level (rna_targets) labels.

    Args:
        dataset: An iterable containing RNA data structures, where each RNA has:
                - A 'rna' attribute with nodes() method and name property
                - Access to a features_computer method that returns either
                  'nt_targets' (node-level) or 'rna_targets' (graph-level)

    Returns:
        tuple: (total_counts, per_rna_counts)
            - total_counts: Counter object with counts of all unique labels
            - per_rna_counts: Dictionary mapping RNA names to their label counters
    """
    all_labels = []  # Will contain all labels in the dataset
    per_rna_counts = {}

    # Check if we have node-level or graph-level labels
    first_features = dataset.features_computer(dataset[0])
    is_node_level = "nt_targets" in first_features

    for rna in dataset:
        features = dataset.features_computer(rna)

        if is_node_level:
            # Node-level labels (nt_targets)
            node_map = {n: i for i, n in enumerate(sorted(rna["rna"].nodes()))}
            labels = [features["nt_targets"][n] for n in node_map]
        else:
            # Graph-level labels (rna_targets)
            labels = [features["rna_targets"]]

        # Convert labels to tuples for counting
        tuple_labels = [tuple(t.flatten().tolist()) for t in labels]

        # Count labels for this specific RNA
        per_rna_counts[rna["rna"].name] = Counter(tuple_labels)

        # Add labels to overall collection
        all_labels.extend(labels)

    # Convert tensors to tuples and count unique combinations
    tuple_list = [tuple(t.flatten().tolist()) for t in all_labels]
    total_counts = Counter(tuple_list)

    return total_counts, per_rna_counts


def split_list_in_fractions(
    list_to_split: list[Any],
    split_train: float = 0.7,
    split_valid: float = 0.15,
    seed: int = 0,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Split a list and return sub-lists by given fractions split and validation.
    The remainder of the dataset is used for the test set.

    :param list_to_split: list you want to split.
    :param split_train: fraction of dataet to use for train set
    :param split_valid: fraction of dataset to use for validation
    """
    copy_list = list_to_split.copy()
    random.Random(seed).shuffle(copy_list)

    train_index, valid_index = (
        int(split_train * len(copy_list)),
        int(
            split_valid * len(copy_list),
        ),
    )

    train_list = copy_list[:train_index]
    valid_list = copy_list[train_index : train_index + valid_index]
    test_list = copy_list[train_index + valid_index :]
    return train_list, valid_list, test_list


def random_split(dataset, split_train=0.7, split_valid=0.15, seed=0):
    """Just randomly split a dataset
    :param dataset:
    :param split_train:
    :param split_valid:
    :return:
    """
    indices = list(range(len(dataset)))
    train_indices, valid_indices, test_indices = split_list_in_fractions(
        indices,
        split_train=split_train,
        split_valid=split_valid,
        seed=seed,
    )

    return sorted(train_indices), sorted(valid_indices), sorted(test_indices)


def split_dataset(dataset, split_train=0.7, split_valid=0.85):
    node_targets = [f"node_{target}" for target in dataset.nt_targets]
    # 1st strategy : if we are looking for a single property : subset the graphs that contain at least a node with this
    # property and make a random split among these.
    if len(node_targets) == 1:
        train_split, validation_split, test_split = get_single_task_split(
            node_targets[0],
            split_train=split_train,
            split_valid=split_valid,
        )
    # 2nd strategy : for multitask objective, we could also subset, but a random split could end up with one of
    # the categories missing from a set
    else:
        train_split, validation_split, test_split = get_multitask_split(
            node_targets=node_targets,
            split_train=split_train,
            split_valid=split_valid,
        )
    train_set = dataset.subset(train_split)
    validation_set = dataset.subset(validation_split)
    test_set = dataset.subset(test_split)
    return train_set, validation_set, test_set
