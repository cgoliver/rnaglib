from collections import defaultdict, Counter
import os
import numpy as np
import pandas as pd
import pickle


def pdb_sel_to_rfam():
    """
    Using the PDB-RFAM mapping hosted there: https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.pdb.gz
    return a df holding RFAM annotations along their PDB occurences
    :return:
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(script_dir, "Rfam.pdb")
    df = pd.read_csv(file_name, sep='\t')
    df['pdbsel'] = df.apply(lambda row: f"{row['pdb_id']}_{row['chain']}_{row['pdb_start']}_{row['pdb_end']}",
        axis=1)
    df = df[['pdbsel', 'pdb_id', 'rfam_acc']]
    return df


def get_rfam_to_go():
    """
    Using the RFAM-GO mapping hosted there : https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/rfam2go/rfam2go
    :return: a dict mapping RFAM ids to GO terms
    """
    rfam_to_go = defaultdict(list)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(script_dir, "rfam2go")
    with open(file_name) as f:
        l = f.readlines()

    l = [x.split('GO:00') for x in l]
    for line in l:
        if len(line) == 2:
            rfam_id = line[0][5:12]
            go_term = '00' + line[1][:5]
            rfam_to_go[rfam_id].append(go_term)
    return rfam_to_go


def get_frequent_go_pdbsel(min_count=50, cache=True):
    """
    Get all GO annotations for the PDB, remove overrepresented ones (ribosomes and tRNAs) and
      underrepresented ones (less than min_count), resulting in 15 classes
    Then, remove redundant GO terms (very correlated columns), which results in 10 classes
    :param min_count: the frequency cutoff, set at 50 following DeepFRI
    :return: the final df with PDB selections and their associated labels
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_file_name = os.path.join(script_dir, "precomputed_rfam.p")
    if os.path.exists(out_file_name) and out_file_name:
        return pickle.load(open(out_file_name, "rb"))
    # Get all pdbs with an existing RFAM annotation and group the corresponding selections by pdb
    df = pdb_sel_to_rfam()
    # pdb_dict = df.groupby('pdb_id').apply(lambda group: group.to_dict(orient='records')).to_dict()
    # print(len(pdb_dict))
    # print(sum([len(x) for x in pdb_dict.values()]))
    # df holds 14709 lines, present in 3121 PDB files

    # from rnaglib.data_loading import RNADataset
    # data = RNADataset(redundancy='all', in_memory=False)
    # list_systems = data.all_rnas.keys()
    # existing_keys = set(pdb_dict.keys()).intersection(set(list_systems))
    # Most systems are present in our database (3110/3121)

    rfam_to_go = get_rfam_to_go()
    pdbsel_go_terms = [rfam_to_go[rfam] for rfam in df['rfam_acc']]

    # Count which go term happen at which frequency
    flattened = [i for j in pdbsel_go_terms for i in j]
    counter = Counter(flattened)
    # a = sorted(counter.items(), key=lambda x: x[1])
    # Counter({
    # '0003735': 11260, '0005840': 11260, '0030533': 2242, '0000244': 202, '0000353': 201, '0046540': 201,
    # '0010468': 124, ... '0005691': 1, '0030622': 1, '0019079': 1, '0039705': 1, '0008380': 1, '0003729': 1,
    # '0017148': 1, '0050897': 1, '0055065': 1})
    # As can be seen, there are rare values, and a few very frequent ones.
    # The most frequent ones (0003735, 0005840) are 'structural constituent of ribosome' and 'ribosome',
    # while 0030533 is 'tRNA'.

    # Filter go terms based on size
    filtered_go_terms = {go for go, count in counter.items() if 1000 > count > min_count}
    # Keep only the lines in the df that amount to an RFAM id corresponding to a relevant GO term
    row_filter = [len(set(pdbsel_go_list).intersection(filtered_go_terms)) > 0 for pdbsel_go_list in pdbsel_go_terms]
    filtered_df = df[row_filter]

    # Now we want to remove duplicated go-terms
    # First, get the data in the form of a matrix
    filtered_pdbsel_go_terms = [rfam_to_go[rfam] for rfam in filtered_df['rfam_acc']]
    stacked = np.zeros((len(filtered_pdbsel_go_terms), len(filtered_go_terms)))
    one_hot = {filtered_go_term: i for i, filtered_go_term in enumerate(filtered_go_terms)}
    for i, gos in enumerate(filtered_pdbsel_go_terms):
        for go in gos:
            try:
                idx = one_hot[go]
            except KeyError:
                continue
            stacked[i, idx] = 1

    # Find highly correlated column pairs by computing the correlation matrix
    correlation_matrix = np.corrcoef(stacked, rowvar=False)
    threshold = 0.9  # Define your correlation threshold
    correlated_pairs = []
    num_columns = correlation_matrix.shape[0]
    for i in range(num_columns):
        for j in range(i + 1, num_columns):
            if abs(correlation_matrix[i, j]) >= threshold:
                correlated_pairs.append((i, j, correlation_matrix[i, j]))
    # print("\nHighly correlated column pairs:", correlated_pairs)

    # Pick one representative per correlated label
    to_keep = set(range(num_columns))
    for pair in correlated_pairs:
        if pair[1] in to_keep:
            to_keep.remove(pair[1])

    # Now this is our final list of go-terms to predict. Subset again the df and return the result
    final_go_terms = {filtered_go_term for filtered_go_term, i in one_hot.items() if i in to_keep}
    final_filter = np.sum(stacked[:, list(to_keep)], axis=1) > 0
    final_df = filtered_df[final_filter]
    pruned_rfam2go = {rfam: [go for go in rfam_to_go[rfam] if go in final_go_terms] for rfam in final_df['rfam_acc']}
    # final_pdbsel_go_terms = [[x for x in rfam_to_go[rfam] if x in final_go_terms] for rfam in filtered_df['rfam_acc']]
    if cache:
        pickle.dump((final_df, pruned_rfam2go), open(out_file_name, 'wb'))
    return final_df, pruned_rfam2go


if __name__ == "__main__":
    df = pdb_sel_to_rfam()
    pdb_dict = {pdb: (rfam, pdb_sel) for pdb_sel, pdb, rfam in df.values}
    # print(pdb_dict)

    rfam_to_go = get_rfam_to_go()
    # print(rfam_to_go)

    get_frequent_go_pdbsel()
