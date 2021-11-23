import os
import sys

import random
from sklearn.metrics import roc_auc_score

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..', '..'))

from rnaglib.data_loading import graphloader, get_all_labels
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
    This function takes a supervised task (possibly multi-task) and generates balanced lists of graphs to load from 
    the NR set in a deterministic way.
    
    Using these splits is a compulsory step of entering the leaderboard

    :param node_target: A subset of {'binding_protein', 'binding_small-molecule', 'is_modified', 'binding_ion'}
    :param seed: Should be set to the default zero
    :return: Lists of graphs that consist in the train and test splits 
    """
    random.seed(seed)
    if isinstance(node_target, str):
        node_target = set(node_target)
    query_attrs = {f'node_{target_feature}' for target_feature in node_target}
    train_split, test_split = get_all_labels.get_splits(query_attrs=query_attrs, return_train=True)
    train_split, test_split = sorted(train_split), sorted(test_split)
    return train_split, test_split


def get_performance(node_target, model, node_features=None, evaluation_function=roc_auc_score):
    """
    Evaluates a model on a given task
    
    :param node_target: A subset of {'binding_protein', 'binding_small-molecule', 'is_modified', 'binding_ion'}
    :param model: A Pytorch model
    :param node_features: To build the evaluation loader, we need to know which features were used !
    :param evaluation_function: Function according to which we want to evaluate our model.
    Takes a predicted and true labels list and returns the value
    :return: The loss value
    """
    train_split, test_split = get_task_split(node_target=node_target)
    test_dataset = graphloader.SupervisedDataset(node_features=node_features,
                                                 node_target=node_target,
                                                 all_graphs=test_split)
    test_loader = graphloader.GraphLoader(dataset=test_dataset, split=False).get_data()

    loss = learn.evaluate_model_supervised(model=model, validation_loader=test_loader,
                                           evaluation_function=evaluation_function)
    return loss


if __name__ == '__main__':
    pass
    task = {'node_binding_protein_id', "node_binding_ion",
            "is_modified", 'node_binding_small-molecule'}
    get_task_split(node_target=task)
    # get_task_split('node_binding_ion')
