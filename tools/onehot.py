"""
    One hot validation for motifs.

    Whole-RNA level prediction from motif one-hot.

    Possible datasets/prediction tasks:
        - RFAM family
        - GO annotation
        - Binds protein
        - Binds small molecule
"""

import os
import sys
import json
import pickle
from collections import Counter, defaultdict

from tqdm import tqdm
import numpy as np
import networkx as nx

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from motif_build.meta_graph import MGraph, MGraphAll, MGraphNC


"""
Building data
"""
def build_onehot_external(whole_graphs,
                          pdb_annotations,
                          method='bgsu',
                          maximal_only=True,
                          task='rfam'
                          ):
    """
    Extract onehots for each PDB in the annotated folder.

    :param annotated graphs: path to whole graphs.

    :return X: one hot array of number of PDBs by number of motifs.
    """

    from sklearn.preprocessing import OneHotEncoder

    with open(pdb_annotations, 'r') as labels:
        pdb_annot_dict = json.load(labels)

    motif_set = set()
    motif_occurrences = defaultdict(set)

    for g in tqdm(os.listdir(whole_graphs)):
        pdbid = g.split('.')[0]
        if pdbid not in pdb_annot_dict[task]:
            continue
        G = nx.read_gpickle(os.path.join(whole_graphs, g))
        for _,d in G.nodes(data=True):
            motif_occurrences[pdbid] = motif_occurrences[pdbid] | set(d[method])
            motif_set = motif_set | set(d[method])
            pass
        pass
    # get one hot
    hot_map = {motif: i for i, motif in enumerate(sorted(motif_set))}
    X = np.zeros((len(motif_occurrences), len(hot_map)))
    pdbs = []
    for i, (pdb, motif_set) in enumerate(motif_occurrences.items()):
        pdbs.append(pdb)
        for motif in motif_set:
            X[i][hot_map[motif]] = 1.

    # encode prediction targets
    target_labels = []
    for pdb in pdbs:
        if task == 'go':
            label = pdb_annot_dict[task][pdb]['go_id']
        else:
            label = pdb_annot_dict[task][pdb]
        target_labels.append(label)

    target_encode = {label: i for i, label in
                     enumerate(sorted(list(set(target_labels))))}

    y = [target_encode[label] for label in target_labels]

    return X, y

def build_onehot(meta_graph_path,
                 pdb_annotations,
                 maximal_only=True,
                 task='all'
                 ):
    """
    Extract onehots for each PDB in the meta_graph

    :param meta_graph_path: path to meta graph
    :param maximal_only: if True, only keeps largest motif if superset.

    :return X: one hot array of number of PDBs by number of motifs.
    """

    from sklearn.preprocessing import OneHotEncoder

    # map pdbs to dict of motif occurrences
    pdb_to_motifs = defaultdict(Counter)

    with open(pdb_annotations, 'r') as labels:
        pdb_annot_dict = json.load(labels)

    maga_graph = pickle.load(open(meta_graph_path, 'rb'))

    meta_nodes = sorted(maga_graph.maga_graph.nodes(data=True),
                        key=lambda x: len(x[0]),
                        reverse=True)
    motif_set = set()

    for motif, d in meta_nodes:
        # motif_id = "-".join(map(str, list(motif)))
        for i, instance in enumerate(d['node_set']):
            node = list(instance).pop()
            pdbid = node[0].split("_")[0]

            # skip if we don't have annotation for this pdb
            if pdbid not in pdb_annot_dict[task]:
                print("Missing")
                continue

            if maximal_only:
                # make sure larger motif not already counted
                for larger in pdb_to_motifs[pdbid].keys():
                    if motif.issubset(larger):
                        break
                else:
                    pdb_to_motifs[pdbid].update([motif])
                    motif_set.add(motif)
            else:
                pdb_to_motifs[pdbid].update([motif])
                motif_set.add(motif)

    # get one hot
    hot_map = {motif: i for i, motif in enumerate(sorted(motif_set))}
    X = np.zeros((len(pdb_to_motifs), len(hot_map)))
    pdbs = []
    for i, (pdb, motif_counts) in enumerate(pdb_to_motifs.items()):
        pdbs.append(pdb)
        for motif, count in motif_counts.items():
            X[i][hot_map[motif]] = count

    # encode prediction targets
    target_labels = []
    for pdb in pdbs:
        label = pdb_annot_dict[task][pdb]
        target_labels.append(label)

    target_encode = {label: i for i, label in
                     enumerate(sorted(list(set(target_labels))))}

    y = [target_encode[label] for label in target_labels]

    return X, y


def build_prediction_set(pdbids, onehots, task='rfam'):
    """
    Build standard ML X,y matrices from onehots and pdbs.
    """
    pass


def _parse_rfam_pdbs(rfam_pdb_txt='../data/Rfam/rfam_pdb.txt'):
    """
        Extract rfam to pdb data.

        {<pdbid>:<rfam id>}
    """
    data = {}
    with open(rfam_pdb_txt, 'r') as rfam:
        next(rfam)
        for line in rfam:
            rfam_id, pdb_id = line.split()[:2]
            data[pdb_id] = rfam_id

    return data


def _parse_rfam_go_annotations(rfam_pdbs,
                               go_annots='../data/Rfam/rfam2go.txt'
                               ):
    """
        Extract GO annots for each id in list of rfam_ids.
    """
    data = {}
    with open(go_annots, 'r') as go:
        rfam_to_pdb = {v: k for k, v in rfam_pdbs.items()}
        for line in go:
            info = line.split()
            rfam_id = info[0].replace("Rfam:", "")
            if rfam_id in rfam_pdbs.values():
                data[rfam_to_pdb[rfam_id]] = {'rna_info': info[1],
                                              'go_id': info[-1]}
    return data


def build_rna_annot_dataset():
    """
    Build big JSON with annotations for all RNAs.

    { <annot_type>: [{<pdbid>:<annot>}, ] }
    """
    import json

    data = {}
    data['rfam'] = _parse_rfam_pdbs()
    data['go'] = _parse_rfam_go_annotations(data['rfam'])

    json.dump(data, open('../data/onehot_data.json', 'w'))
    pass


def accuracy(X, y):
    """
    :param task, which task to predict on.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import SGDClassifier
    from sklearn.dummy import DummyClassifier

    print(f"Data shape {X.shape}.")

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.2,
                                                        random_state=0
                                                        )

    dummy = DummyClassifier().fit(X_train, y_train)
    dummy_acc = dummy.score(X_test, y_test)

    model = SGDClassifier().fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    print("accuracy ", acc)
    print("dummy ", dummy_acc)


if __name__ == "__main__":
    # build_rna_annot_dataset()
    X, y = build_onehot('../data/magas/mgraph_1hop_epsilon.p',
                        '../data/onehot_data.json'
                        )
    print("maga rows ", X.shape)
    accuracy(X, y)
    X, y = build_onehot_external(
                        '../data/whole_v3',
                        '../data/onehot_data.json',
                        method='bgsu'
                        )
    print("external rows ", X.shape)
    accuracy(X, y)
