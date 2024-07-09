import os

script_dir = os.path.dirname(os.path.realpath(__file__))

EDGE_MAP_FR3D = {'B53': 0, 'CHH': 1, 'CHS': 2, 'CHW': 3, 'CSH': 4, 'CSS': 5, 'CSW': 6, 'CWH': 7, 'CWS': 8, 'CWW': 9,
                 'THH': 10, 'THS': 11, 'THW': 12, 'TSH': 13, 'TSS': 14, 'TSW': 15, 'TWH': 16, 'TWS': 17, 'TWW': 18}
EDGE_MAP_RGLIB = {'B53': 0, 'cHH': 1, 'cHS': 2, 'cHW': 3, 'cSH': 4, 'cSS': 5, 'cSW': 6, 'cWH': 7, 'cWS': 8, 'cWW': 9,
                  'tHH': 10, 'tHS': 11, 'tHW': 12, 'tSH': 13, 'tSS': 14, 'tSW': 15, 'tWH': 16, 'tWS': 17, 'tWW': 18,
                  'B35': 19, }

EDGE_MAP_FR3D_REVERSE = {v: k for k, v in EDGE_MAP_FR3D.items()}
EDGE_MAP_RGLIB_REVERSE = {v: k for k, v in EDGE_MAP_RGLIB.items()}

CANONICALS_FR3D = {'B53', 'B35', 'CWW'}
CANONICALS_RGLIB = {'B53', 'B35', 'cWW'}

IDF = {'TSS': 1.3508944643423815, 'TWW': 2.2521850545837103, 'CWW': 0.7302387734487946, 'B53': 0.6931471805599453,
       'CSS': 1.3562625353981017, 'TSH': 1.0617196804844624, 'THS': 1.0617196804844624, 'CSH': 1.6543492684466312,
       'CHS': 1.6543492684466312, 'THW': 1.3619066730630602, 'TWH': 1.3619066730630602, 'THH': 2.3624726636947186,
       'CWH': 2.220046456989285, 'CHW': 2.220046456989285, 'TSW': 2.3588208814802263, 'TWS': 2.3588208814802263,
       'CWS': 2.0236918714028707, 'CHH': 4.627784875752877, 'CSW': 2.0236918714028707}
IDF_RGLIB = {key[0].lower() + key[1:]: value for key, value in IDF.items()}

INDEL_VECTOR_FR3D = [1 if e[0] == 'B' else 2 if e == 'CWW' else 3 for e in sorted(EDGE_MAP_FR3D.keys())]
INDEL_VECTOR_RGLIB = [1 if e[0] == 'B' else 2 if e == 'cWW' else 3 for e in sorted(EDGE_MAP_RGLIB.keys())]

DEFAULT_GRAPH_DIR = os.path.join(script_dir, "..", "data", "graphs", "rna_graphs_nr")
DEFAULT_ANNOT_DIR = os.path.join(script_dir, "..", "data", "annotated", "all_rna_nr")

GRAPH_KEYS = {'nt_position': {'RGLIB': 'nt_resnum', 'FR3D': 'pdb_pos'},
              'chain': {'RGLIB': 'chain_name', 'FR3D': 'chain'},
              'bp_type': {'RGLIB': 'LW', 'FR3D': 'label'},
              'edge_map': {'RGLIB': EDGE_MAP_RGLIB, 'FR3D': EDGE_MAP_FR3D},
              'canonical': {'RGLIB': CANONICALS_RGLIB, 'FR3D': CANONICALS_FR3D},
              'indel_vector': {'RGLIB': INDEL_VECTOR_RGLIB, 'FR3D': INDEL_VECTOR_FR3D},
              'valid_edges': {'RGLIB': EDGE_MAP_RGLIB.keys(), 'FR3D': EDGE_MAP_FR3D.keys()},
              'idf': {'RGLIB': IDF_RGLIB, 'FR3D': IDF},
              }
TOOL = 'RGLIB'
