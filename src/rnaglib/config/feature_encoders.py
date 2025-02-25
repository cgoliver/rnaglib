from rnaglib.encoders import FloatEncoder, BoolEncoder, OneHotEncoder, ListEncoder

"""
Assign encoders to features available by default.
"""

NODE_FEATURE_MAP = {
    'index': None,  # Not a feature
    'index_chain': None,  # Not a feature
    'chain_name': None,  # Not a feature
    'nt_resnum': None,  # Not a feature
    "nt_name": None,  # This looks crappy to me but maybe there are some that are canonical and a lot of modified ones ?
    'nt_code': OneHotEncoder(mapping={'A': 0, 'U': 1, 'C': 2, 'G': 3, 'a': 0, 'u': 1, 'c': 2, 'g': 3}, num_values=4),
    "nt_id": None,  # This looks crappy, it looks like all possible node ids (number of possibilities 600k)...
    "nt_type": None,  # Constant = 'RNA'
    "dbn": OneHotEncoder(mapping={'(': 0, '.': 1, ')': 2, '[': 3, ']': 4, '{': 5, '}': 6,
                                  '>': 7, '<': 8, 'A': 9, 'a': 9, 'B': 10, 'b': 10}),
    "summary": None,  # This looks a bit fishy, with 74k entries and a lot are quite populated. TODO :understand better
    "alpha": FloatEncoder(),
    "beta": FloatEncoder(),
    "gamma": FloatEncoder(),
    "delta": FloatEncoder(),
    "epsilon": FloatEncoder(),
    "zeta": FloatEncoder(),
    "epsilon_zeta": FloatEncoder(),
    "bb_type": OneHotEncoder(mapping={'BI': 0, '--': 1, 'BII': 2}),
    "chi": FloatEncoder(),
    "glyco_bond": OneHotEncoder(mapping={'anti': 0, '--': 1, 'syn': 2}),
    "C5prime_xyz": ListEncoder(list_length=3),
    "P_xyz": ListEncoder(list_length=3),
    # This looks like a redundant feature with glyco_bond...
    "form": OneHotEncoder(mapping={'anti': 0, '--': 1, 'syn': 2}),
    "ssZp": FloatEncoder(),
    "Dp": FloatEncoder(),
    "splay_angle": FloatEncoder(),
    "splay_distance": FloatEncoder(),
    "splay_ratio": FloatEncoder(),
    "eta": FloatEncoder(),
    "theta": FloatEncoder(),
    "eta_prime": FloatEncoder(),
    "theta_prime": FloatEncoder(),
    "eta_base": FloatEncoder(),
    "theta_base": FloatEncoder(),
    "v0": FloatEncoder(),
    "v1": FloatEncoder(),
    "v2": FloatEncoder(),
    "v3": FloatEncoder(),
    "v4": FloatEncoder(),
    "amplitude": FloatEncoder(),
    "phase_angle": FloatEncoder(),
    # TODO : understand better these ones
    "puckering": OneHotEncoder(
        mapping={"C3'-endo": 0, "C4'-exo": 1, "C2'-exo": 2, "C3'-exo": 3, "O4'-endo": 4, "O4'-exo": 5,
                 "C4'-endo": 6, "C2'-endo": 7, "C1'-exo": 8, "C1'-endo": 9, '': 10}),
    "sugar_class": OneHotEncoder(mapping={"~C3'-endo": 0, "~C2'-endo": 1, '--': 2}),
    "bin": OneHotEncoder(mapping={'inc': 0, '33p': 1, 'trig': 2, '32p': 3, '22t': 4, '23p': 5, '33t': 6, '32t': 7,
                                  '23m': 8, '23t': 9, '22p': 10, '22m': 11, '33m': 12, '32m': 13}),
    "cluster": OneHotEncoder(mapping={'__': 0, '1a': 1, '1L': 2, '!!': 3, '1[': 4, '0a': 5, '1c': 6, '&a': 7, '1e': 8,
                                      '1g': 9, '9a': 10, '7a': 11, '1b': 12, '2a': 13, '0b': 14, '4d': 15, '6g': 16,
                                      '4b': 17, '6n': 18, '5n': 19, '1m': 20, '1z': 21, '2[': 22, '3d': 23, '5j': 24,
                                      '6j': 25, '1t': 26, '2g': 27, '7d': 28, '2h': 29, '6d': 30, '7p': 31, '2o': 32,
                                      '2u': 33, '1o': 34, '2z': 35, '5z': 36, '6p': 37, '8d': 38, '3a': 39, '1f': 40,
                                      '#a': 41, '3b': 42, '4n': 43, '5d': 44, '0i': 45, '4a': 46, '7r': 47, '5p': 48,
                                      '4p': 49, '4g': 50, '5q': 51, '5r': 52, '0k': 53, '4s': 54}),
    "suiteness": FloatEncoder(),
    "filter_rmsd": FloatEncoder(),
    "frame_rmsd": FloatEncoder(),
    "frame_origin": ListEncoder(list_length=3),
    "frame_x_axis": ListEncoder(list_length=3),
    "frame_y_axis": ListEncoder(list_length=3),
    "frame_z_axis": ListEncoder(list_length=3),
    "frame_quaternion": ListEncoder(list_length=3),
    "sse_sse": None,  # TODO : ?onehot?
    "binding_protein": BoolEncoder(),
    "binding_ion": BoolEncoder(),
    "binding_small-molecule": BoolEncoder(),
    "binding_small-molecule-4.0A": BoolEncoder(),
    "binding_small-molecule-6.0A": BoolEncoder(),
    "binding_small-molecule-8.0A": BoolEncoder(),
    # These stuff can be dicts, but I guess most of the time it will be a binary rather than a categorical prediction.
    # This is more advanced.
    # "binding_ion": None,  # TODO : ? onehot ?
    # "binding_small-molecule": None,  # TODO : ? onehot ?
    "binding_protein_id": None,  # trash
    "binding_protein_nt-aa": None,  # trash
    "binding_protein_nt": None,  # trash
    "binding_protein_aa": None,  # trash
    "binding_protein_Tdst": FloatEncoder(),
    "binding_protein_Rdst": FloatEncoder(),
    "binding_protein_Tx": FloatEncoder(),
    "binding_protein_Ty": FloatEncoder(),
    "binding_protein_Tz": FloatEncoder(),
    "binding_protein_Rx": FloatEncoder(),
    "binding_protein_Ry": FloatEncoder(),
    "binding_protein_Rz": FloatEncoder(),
    "is_modified": BoolEncoder(),
    "is_broken": BoolEncoder(),
    "protein_binding": BoolEncoder(),
    "protein_content": ListEncoder(list_length=3)
}

# TODO : include edge information, but it's not trivial to deal with edges beyond RGCN...
EDGE_FEATURE_MAP = {
    "LW": None,  #
    "backbone": BoolEncoder(),
    "nt1": None,  # trash
    "nt2": None,  # trash
    "bp": None,  # trash
    "name": None,  # trash
    "Saenger": None,  # trash
    "DSSR": None,  # trash
}
