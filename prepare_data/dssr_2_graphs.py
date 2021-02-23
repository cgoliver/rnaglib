"""

Build annotated graphs using [x3dna DSSR](http://docs.x3dna.org/dssr-manual.pdf) annotations.
Requires a x3dna-dssr executable to be in $PATH.

"""

import json
import multiprocessing as mp
from subprocess import check_output

import networkx as nx

def dssr_exec(cif):
    """ Run DSSR on a given MMCIF

    Args:
    ---
    cif (str): Path to MMCIF for annotation.

    Returns:
    ---
    annot (dict): raw DSSR output dictionary.
    """
    try:
        annot = check_output(["x3dna-dssr", "--json", f"-i={cif}"] )
    except Exception as e:
        print(e)
        return (1, None)
    return (0, json.loads(annot))

def annot_2_graph(annot):
    """
    DSSR Annotation JSON Keys:

        dict_keys(['num_pairs', 'pairs', 'num_helices', 'helices',
        'num_stems', 'stems', 'num_coaxStacks', 'coaxStacks', 'num_stacks',
        'stacks', 'nonStack', 'num_atom2bases', 'atom2bases', 'num_hairpins',
        'hairpins', 'num_bulges', 'bulges', 'num_splayUnits', 'splayUnits',
        'dbn', 'chains', 'num_nts', 'nts', 'num_hbonds', 'hbonds',
        'refCoords', 'metadata']
    """

    def get_backbones(nts):
        """ Get backbone pairs.
        Args:
        ___
        nts (dict): DSSR nucleotide info.

        Returns:
        ---
        bb (list): list of tuples (5' base, 3' base)
        """
        bb = []
        for i, three_p in enumerate(nts):
            if i == 0:
                continue
            five_p = nts[i-1]
            if five_p['chain_name'] != three_p['chain_name']:
                continue
            if 'break' not in three_p['summary']:
                bb.append((five_p, three_p))
        return bb

    G = nx.DiGraph()

    # add nucleotides
    G.add_nodes_from((d['nt_id'] for d in annot['nts']))

    # add backbones
    bbs = get_backbons(annot['nts'])
    G.add_edges_from(((five_p['nt_id'], three_p['nt_id'], {'label': 'B53'}) \
                      for five_p, three_p in bbs))

    # add base pairs
    G.add_edges_from(((pair['nt1'], pair['nt2']) for pair in annot['pairs']))

    # import matplotlib.pyplot as plt
    # nx.draw(G)
    # plt.show()

    pass

def build_one(cif):
    exit_code, annot = dssr_exec(cif)
    print(annot['pairs'][0])
    G = annot_2_graph(annot)
    pass

def build_all():
    pass

if __name__ == "__main__":
    build_one("../data/1aju.cif")
