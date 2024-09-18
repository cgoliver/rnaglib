from typing import List, Any, Tuple
from collections import defaultdict
import random

"""
from rnaglib.utils import load_index

graph_index = load_index()
"""


def split_list_in_fractions(
    list_to_split: List[Any],
    split_train: float = 0.7,
    split_valid: float = 0.15,
    seed: int = 0,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Split a list and return sub-lists by given fractions split and validation. The remainder of the
    dataset is used for the test set.

    :param list_to_split: list you want to split.
    :param split_train: fraction of dataet to use for train set
    :param split_valid: fraction of dataset to use for validation
    """
    copy_list = list_to_split.copy()
    random.Random(seed).shuffle(copy_list)

    train_index, valid_index = int(split_train * len(copy_list)), int(
        split_valid * len(copy_list)
    )

    train_list = copy_list[:train_index]
    valid_list = copy_list[train_index : train_index + valid_index]
    test_list = copy_list[train_index + valid_index :]
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
    train_indices, valid_indices, test_indices = split_list_in_fractions(
        indices, split_train=split_train, split_valid=split_valid, seed=seed
    )

    return sorted(train_indices), sorted(valid_indices), sorted(test_indices)


def split_dataset(dataset, split_train=0.7, split_valid=0.85):
    node_targets = [f"node_{target}" for target in dataset.nt_targets]
    # 1st strategy : if we are looking for a single property : subset the graphs that contain at least a node with this
    # property and make a random split among these.
    if len(node_targets) == 1:
        train_split, validation_split, test_split = get_single_task_split(
            node_targets[0], split_train=split_train, split_valid=split_valid
        )
    # 2nd strategy : for multitask objective, we could also subset, but a random split could end up with one of
    # the categories missing from a set
    else:
        train_split, validation_split, test_split = get_multitask_split(
            node_targets=node_targets, split_train=split_train, split_valid=split_valid
        )
    train_set = dataset.subset(train_split)
    validation_set = dataset.subset(validation_split)
    test_set = dataset.subset(test_split)
    return train_set, validation_set, test_set
