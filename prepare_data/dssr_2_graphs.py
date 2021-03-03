"""

Build annotated graphs using [x3dna DSSR](http://docs.x3dna.org/dssr-manual.pdf) annotations.
Requires a x3dna-dssr executable to be in $PATH.

"""
import os
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

def snap_exec(cif):
    try:
        annot = check_output(["x3dna-dssr", "snap", f"-i={cif}"] )
    except Exception as e:
        print(e)
        return (1, None)
    return (0, annot.decode("utf-8"))

def snap_parse(snap_out):
    """ It seems SNAP output is raw text so we have to parse it.
    For now we just retrieve nucleotide-aa contacts and related info.

    Output has sections that look like this

    ****************************************************************************
    List of 7 nucleotide/amino-acid interactions
        id   nt-aa   nt           aa              Tdst    Rdst     Tx      Ty      Tz      Rx      Ry      Rz
    1  1aju  A-arg  A.A22        A.ARG47          7.27 -144.55   -2.96    6.04    2.75   19.89   63.03 -137.41
    2  1aju  U-arg  A.U23        A.ARG47         -9.86  159.76    7.92   -5.66    1.55  -12.69   76.33  153.98
    3  1aju  G-arg  A.G26        A.ARG47          7.04  153.29   -1.62   -6.69   -1.49  -49.89  -49.73  147.15
    4  1aju  A-arg  A.A27        A.ARG47         -6.60  142.39   -0.97   -4.94   -4.26  -54.29  -28.82  135.95
    5  1aju  C-arg  A.C37        A.ARG47          6.62 -114.00   -1.15    1.89    6.23  -53.57   19.41 -103.42
    6  1aju  U-arg  A.U38        A.ARG47          6.41 -139.96   -1.23    5.41    3.20  -61.93   31.00 -130.83
    7  1aju  C-arg  A.C39        A.ARG47         -7.42 -171.37   -0.96    7.35   -0.33  -67.02   33.23 -169.13

    ****************************************************************************


    """
    import re

    lines = iter(snap_out.split("\n"))

    # skip to nucleotide amino acid section (i know this is ugly)
    while True:
        l = next(lines)
        print(l)
        if re.match("List of [0-9]+ nucleotide/amino-acid", l):
            break
        else:
            next(lines)

    interface_nts = dict()
    for i,l in enumerate(lines):
        if i == 0:
            header = l.split()[1:]
            continue
        if l.startswith("*"):
            break
        if not l:
            break
        # get rid of first two columns
        l = l.split()[2:]
        nt_id = l[1]
        interface_nts[nt_id] = dict(zip(header, l))

    return interface_nts

def find_nt(nt_annot, nt_id):
    for nt in nt_annot:
        if nt['nt_id'] == nt_id:
            return nt

def rna_only_nts(annot):
    """ Filter nucleotide annotations to only keep RNA.
    """
    return filter(lambda x: x['nt_type'] == 'RNA', annot['nts'])

def rna_only_pairs(annot):
    """ Only keep pairs between RNAs."""
    return filter(lambda x: find_nt(annot['nts'], x['nt1'])['nt_type'] == 'RNA' and \
                            find_nt(annot['nts'], x['nt2'])['nt_type'] == 'RNA', \
                  annot['pairs'])

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

def add_sses(g, annot):
    """ Return dict of nodes that are in an sse as a list of
    annotations. TODO: add helices and stems """
    sse_annots = dict()
    sse_types = ['hairpins', 'junctions', 'bulges', 'internal']
    for sse in sse_types:
        try:
            elements = annot[sse]
        except KeyError:
            continue
        for elem in elements:
            for nt in elem['nts_long'].split(','):
                if nt in g.nodes():
                    sse_annots[nt] = {'sse': f'{sse[:-1]}_{elem["index"]}'}
    return sse_annots

def annot_2_graph(annot, rbp_annot, pdbid):
    """
    DSSR Annotation JSON Keys:

        dict_keys(['num_pairs', 'pairs', 'num_helices', 'helices',
        'num_stems', 'stems', 'num_coaxStacks', 'coaxStacks', 'num_stacks',
        'stacks', 'nonStack', 'num_atom2bases', 'atom2bases', 'num_hairpins',
        'hairpins', 'num_bulges', 'bulges', 'num_splayUnits', 'splayUnits',
        'dbn', 'chains', 'num_nts', 'nts', 'num_hbonds', 'hbonds',
        'refCoords', 'metadata']
    """

    G = nx.DiGraph()

    nt_annot = rna_only_nts(annot)

    # add nucleotides
    G.add_nodes_from(((d['nt_id'], d) for d in nt_annot))

    # print(annot['nts'])
    # add backbones
    bbs = get_backbones(annot['nts'])
    G.add_edges_from(((five_p['nt_id'], three_p['nt_id'], {'LW': 'B53', 'backbone': True}) \
                      for five_p, three_p in bbs))

    # add base pairs
    rna_pairs = rna_only_pairs(annot)
    G.add_edges_from(((pair['nt1'], pair['nt2'], pair)\
                      for pair in rna_pairs))
    G.add_edges_from(((pair['nt2'], pair['nt1'], pair)\
                      for pair in rna_pairs))

    # add SSE data
    sse_nodes = add_sses(G, annot)
    for node in G.nodes():
        try:
            G.nodes[node]['sse'] = sse_nodes[node]
        except KeyError:
            G.nodes[node]['sse'] = {'sse': None}

    # add RNA-Protein interface data
    for node in G.nodes():
        try:
            G.nodes[node]['binding_protein'] = rbp_annot[node]
            print(node)
        except KeyError:
            G.nodes[node]['binding_protein'] = None

    new_labels = {n: ".".join([pdbid, str(d['chain_name']), str(d['nt_resnum'])])\
                    for n,d in G.nodes(data=True)}

    G = nx.relabel_nodes(G, new_labels)

    # import matplotlib.pyplot as plt
    # nx.draw(G)
    # plt.show()

    return G

def build_one(cif):
    exit_code, annot = dssr_exec(cif)
    rbp_exit_code, rbp_out = snap_exec(cif)
    try:
        rbp_annot = snap_parse(rbp_out)
    except:
        rbp_annot = {}
    # print(annot['pairs'][0])
    pdbid = os.path.basename(cif).split(".")[0]
    G = annot_2_graph(annot, rbp_annot, pdbid)

    return G
    # G_data = nx.readwrite.json_graph.node_link_data(G)
    # with open("../examples/1aju.json", "w") as out:
        # json.dump(G_data, out)
    # pass

def build_all():
    pass

if __name__ == "__main__":
    # doc example with multiloop
    build_one("../data/1aju.cif")
    # multi chain
    # build_one("../data/4q0b.cif")
