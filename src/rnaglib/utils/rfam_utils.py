from collections import defaultdict
import os
import pandas as pd


def pdb_sel_to_rfam():
    """
    Using the PDB-RFAM mapping hosted there: https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.pdb.gz
    return a df holding RFAM annotations along their PDB occurences
    :return:
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(script_dir, "Rfam.pdb")
    df = pd.read_csv(file_name, sep='\t')
    df['unique_id'] = df.apply(lambda row: f"{row['pdb_id']}_{row['chain']}_{row['pdb_start']}{row['pdb_end']}",
        axis=1)
    df = df[['unique_id', 'pdb_id', 'rfam_acc']]
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
            go_term = line[1][:5]
            rfam_to_go[rfam_id].append(go_term)
    return rfam_to_go


if __name__ == "__main__":
    df = pdb_sel_to_rfam()
    pdb_dict = {pdb: (rfam, pdb_sel) for pdb_sel, pdb, rfam in df.values}
    # print(pdb_dict)

    rfam_to_go = get_rfam_to_go()
    pdb_gos = []
    all_rfams = list(df['rfam_acc'].unique())
    for rfam in all_rfams:
        go_term = rfam_to_go[rfam]
        pdb_gos.extend(go_term)
    pdb_gos = list(set(pdb_gos))
    print(len(all_rfams))
    print(len(pdb_gos))
