import os
import sys
import pickle

"""
Checks whether annotation folder and hashtable are complete.
"""

if __name__ == '__main__':
    from tqdm import tqdm

    data_folder = sys.argv[1]

    datasets = ['all_graphs_annot']

    ok = True
    for d in datasets:
        data_path = os.path.join("..", "data", data_folder)
        _, hashtable = pickle.load(open(os.path.join(data_path, f'{d}_hash.p'), 'rb'))
        for g in tqdm(os.listdir(os.path.join(data_path, d))):
            G = pickle.load(open(os.path.join(data_path, d, g), 'rb'))
            for node, rings in G['rings']['graphlet'].items():
                for ring in rings:
                    for r in ring:
                        try:
                            hashtable[r]
                        except KeyError:
                            print(f"Missing {r}")
                            ok = False

    if ok:
        print("DATASET OK")
