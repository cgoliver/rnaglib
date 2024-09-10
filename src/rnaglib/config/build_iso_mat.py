import os
import sys
import numpy as np

from rnaglib.config.graph_keys import EDGE_MAP_RGLIB

""" Isostericity matrix parsing and loading.

"""

s = """
,CHH,TWH,CWW,THS,CWS,CSS,CWH,CHS,TWS,TSS,TWW,THH,B53
CHH,8.9,12,14.7,14,13.7,12.7,15.1,14.7,16.2,16.6,16.2,14,19
TWH,12,2.6,10.6,9.7,14.3,15.6,11.2,15.2,13.8,15.4,11.9,11.4,19
CWW,14.7,10.6,4.1,8.2,9.2,13.1,14.5,16,12.4,11.3,11.1,15.5,19
THS,14,9.7,8.2,2.1,7,12.7,12,12.1,10,11.9,13.1,15.8,19
CWS,13.7,14.3,9.2,7,3.5,7.4,14.9,12.3,10.9,10.8,14.6,17.7,19
CSS,12.7,15.6,13.1,12.7,7.4,1.3,15.8,12.9,13.8,12,17.1,19,19
CWH,15.1,11.2,14.5,12,14.9,15.8,3.2,8.8,8.4,11.5,10.6,10.8,19
CHS,14.7,15.2,16,12.1,12.3,12.9,8.8,2.4,7.9,11.2,14.7,14.9,19
TWS,16.2,13.8,12.4,10,10.9,13.8,8.4,7.9,3.4,6.4,9.6,13.4,19
TSS,16.6,15.4,11.3,11.9,10.8,12,11.5,11.2,6.4,2.2,9,14.4,19
TWW,16.2,11.9,11.1,13.1,14.6,17.1,10.6,14.7,9.6,9,3.8,9,19
THH,14,11.4,15.5,15.8,17.7,19,10.8,14.9,13.4,14.4,9,4,19
B53,19,19,19,19,19,19,19,19,19,19,19,19,0
"""

s2 = """8.9,12,14.7,14,13.7,12.7,15.1,14.7,16.2,16.6,16.2,14,19
12,2.6,10.6,9.7,14.3,15.6,11.2,15.2,13.8,15.4,11.9,11.4,19
14.7,10.6,4.1,8.2,9.2,13.1,14.5,16,12.4,11.3,11.1,15.5,19
14,9.7,8.2,2.1,7,12.7,12,12.1,10,11.9,13.1,15.8,19
13.7,14.3,9.2,7,3.5,7.4,14.9,12.3,10.9,10.8,14.6,17.7,19
12.7,15.6,13.1,12.7,7.4,1.3,15.8,12.9,13.8,12,17.1,19,19
15.1,11.2,14.5,12,14.9,15.8,3.2,8.8,8.4,11.5,10.6,10.8,19
14.7,15.2,16,12.1,12.3,12.9,8.8,2.4,7.9,11.2,14.7,14.9,19
16.2,13.8,12.4,10,10.9,13.8,8.4,7.9,3.4,6.4,9.6,13.4,19
16.6,15.4,11.3,11.9,10.8,12,11.5,11.2,6.4,2.2,9,14.4,19
16.2,11.9,11.1,13.1,14.6,17.1,10.6,14.7,9.6,9,3.8,9,19
14,11.4,15.5,15.8,17.7,19,10.8,14.9,13.4,14.4,9,4,19
19,19,19,19,19,19,19,19,19,19,19,19,0
"""

lines = s2.splitlines()
matrix = list()
for line in lines:
    matrix.append(line.split(','))
matrix = np.asarray(matrix)
matrix = np.asarray(matrix, dtype=float)
matrix = np.exp(-matrix / 8)

keys = list("CHH,TWH,CWW,THS,CWS,CSS,CWH,CHS,TWS,TSS,TWW,THH,B53".split(','))
key_map = {bp: i for i, bp in enumerate(keys)}


def get_undirected_iso(bpa, bpb):
    """
    Given two directed edges, get the values from the undirected isostericity matrix.

    :param bpa: LW edge code
    :type bpa: str
    :param bpb: LW edge code
    :type bpb: str 

    :return: isostericty value
    :rtype float

    """
    bpa = bpa.upper()
    bpb = bpb.upper()
    bpa = bpa if bpa in keys else bpa[0] + bpa[2] + bpa[1]
    bpb = bpb if bpb in keys else bpb[0] + bpb[2] + bpb[1]
    return matrix[key_map[bpa], key_map[bpb]]


def build_iso():
    """
    Build a directed isostericity matrix.

    The heuristic is as follows :
    - It has a diagonal of ones : max similarity is self
    - Backbone is set aside, and has a little cost for reversing the direction
    - Different edges types are computed to have the associated undirected isostericity value

    :return: A np matrix that yields the isostericity values, ordered as EDGE_MAP

    """
    iso_mat = np.zeros(shape=(len(EDGE_MAP_RGLIB), len(EDGE_MAP_RGLIB)), dtype=np.float32)
    for i, bpa in enumerate(EDGE_MAP_RGLIB.keys()):
        for j, bpb in enumerate(EDGE_MAP_RGLIB.keys()):
            # BB to anything else
            if (bpa in ['B53', 'B35'] and bpb not in ['B53', 'B35']) \
                    or (bpb in ['B53', 'B35'] and bpa not in ['B53', 'B35']):
                value = 0.
            # B53 to B35
            elif (bpa == 'B53' and bpb == 'B35') or (bpb == 'B53' and bpa == 'B35'):
                value = 0.2
            # Same bp
            elif bpa == bpb:
                value = 1
            # iso value based on undirected
            else:
                value = get_undirected_iso(bpa, bpb)
            iso_mat[i, j] = value
    return iso_mat


iso_mat = build_iso()
pass
