import os
import sys

import random

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..', '..'))

from rnaglib.data_loading import loader, get_all_labels
from rnaglib.learning import learn

"""
This script is to be used for reproducible benchmarking : we propose an official train/test split 
for four possible tasks : 
- Protein Binding
- Small-molecule Binding
- Link Prediction 
- Chemical Modification

These splits rely on fixed seed random splitting. One can ask to get a multi-task setting to get another relevant split
"""


def get_task_split(node_target, seed=0):
    """

    :param node_target: A subset of {'binding_protein', 'binding_small-molecule', 'is_modified', 'binding_ion'}
    :param seed:
    :return:
    """
    random.seed(seed)
    if isinstance(node_target, str):
        node_target = set(node_target)
    query_attrs = {f'node_{target_feature}' for target_feature in node_target}
    train_split, test_split = get_all_labels.get_splits(query_attrs=query_attrs, return_train=True)
    train_split, test_split = sorted(train_split), sorted(test_split)
    return train_split, test_split


def get_performance(node_target, model, node_features=None):
    train_split, test_split = get_task_split(node_target=node_target)
    test_dataset = loader.SupervisedDataset(node_features=node_features,
                                            node_target=node_target,
                                            all_graphs=test_split)
    test_loader = loader.Loader(dataset=test_dataset, split=False).get_data()
    loss = learn.evaluate_model_supervised(model=model, validation_loader=test_loader)
    return loss


if __name__ == '__main__':
    pass
    task = {'node_binding_protein_id', "node_binding_ion",
            "is_modified", 'node_binding_small-molecule'}
    get_task_split(node_target=task)
    # get_task_split('node_binding_ion')
