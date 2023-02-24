import os
import sys

import random
from sklearn.metrics import roc_auc_score

from rnaglib.data_loading import rna_loader, get_statistics
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
    test_dataset = graphloader.GraphDataset(node_features=node_features,
                                            node_target=node_target)
    _, _, test_loader = graphloader.get_loader(dataset=test_dataset, split=True)

    loss = learn.evaluate_model_supervised(model=model, validation_loader=test_loader,
                                           evaluation_function=evaluation_function)
    return loss


if __name__ == '__main__':
    pass
    task = {'node_binding_protein_id', "node_binding_ion",
            "is_modified", 'node_binding_small-molecule'}
    get_task_split(node_target=task)
    # get_task_split('node_binding_ion')
