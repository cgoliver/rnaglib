import os
import sys

import numpy as np
from sklearn import metrics


def run_virtual_screen(model, dataloader):
    """
    Run_virtual_screen.

    :param model: trained affinity prediction model
    :param dataloader: Loader of VirtualScreenDataset object
    :returns efs: list of efs, one for each pocket in the dataset
    """

    def mean_active_rank(scores, is_active):
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        fpr, tpr, thresholds = metrics.roc_curve(is_active, scores, drop_intermediate=True)
        return metrics.auc(fpr, tpr)

    efs = list()
    failed_set = set()
    print(f"Doing VS on {len(dataloader)} pockets.")
    for i, data in enumerate(dataloader):
        if not i % 20:
            print(f"Done {i}/{len(dataloader)}")

        pocket_name = data['group_rep']

        if len(data['active_ligands']) == 0:
            failed_set.add(pocket_name)
            print(f"VS fail")
            continue
        actives = data['active_ligands'][0]

        inactives = data['inactive_ligands'][0]
        if inactives.batch_size < 10:
            print(f"Skipping pocket{i}, not enough decoys")
            continue

        pocket = data['pocket']
        actives_scores = model.predict_ligands(pocket, actives)[:, 0].numpy()
        inactives_scores = model.predict_ligands(pocket, inactives)[:, 0].numpy()
        scores = np.concatenate((actives_scores, inactives_scores), axis=0)
        is_active = [1 for _ in range(len(actives_scores))] + [0 for _ in range(len(inactives_scores))]
        efs.append(mean_active_rank(scores, is_active))
    if len(failed_set) > 0:
        print(f"VS failed on {failed_set}")
    print('Mean EF :', np.mean(efs))
    return efs


if __name__ == "__main__":
    from rnaglib.tasks.rna_vs.task import VSTask
    from rnaglib.tasks.rna_vs.model import RNAEncoder, LigandGraphEncoder, Decoder, VSModel
    from rnaglib.representations.graph import GraphRepresentation

    # Get a test loader
    root = "../../data/tasks/rna_vs"
    framework = 'dgl'
    ef_task = VSTask(root, ligand_framework=framework)
    representations = [GraphRepresentation(framework=framework)]
    rna_dataset_args = {'representations': representations, 'nt_features': 'nt_code'}
    _, _, test_dataloader = ef_task.get_split_loaders(dataset_kwargs=rna_dataset_args)

    # Get a VS model
    rna_encoder = RNAEncoder()
    lig_encoder = LigandGraphEncoder()
    decoder = Decoder()
    model = VSModel(encoder=rna_encoder, lig_encoder=lig_encoder, decoder=decoder)
    model.eval()

    # Get an EF value
    run_virtual_screen(model, test_dataloader)
