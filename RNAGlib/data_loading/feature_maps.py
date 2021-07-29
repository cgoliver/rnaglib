"""
Functions to build feature map for each feature type.
"""

import torch


class FeatureMap():
    def __init__(self, values_list=None, min_val=None, max_val=None):
        """
        :param value: feature value we want to encode
        :param possible_values (list, optional): list of possible values feature
                                                can take.
        """
        self.values_list_raw, self.values_list = (values_list, values_list)
        self.clean_values()

        self.min_val = min_val
        self.max_val = max_val

        if not values_list is None:
            self.dims = len(self.values_list)
            self.values_enum = {v: k for k, v in enumerate(self.values_list)}
            self.values_enum_r = dict(enumerate(self.values_list))
        else:
            self.dims = 1
        pass

    def encode(self, value):
        raise NotImplementedError

    def clean_values(self):
        pass


class NTCode(FeatureMap):
    """ This feature represents the standard nucleotide type (A,U,C,G,..).
    We one-hot encode this feature.
    """

    def clean_values(self):
        self.values_list = sorted(list(set(map(str.upper, self.values_list))))

    def encode(self, value):
        """ Assign encoding of `value` according to known possible
        values.
        """
        assert value in self.values_list_raw, f"Not a valid value: {value}, must be {self.values_list}"
        x = torch.zeros(self.dims)
        ind = self.values_enum[value.upper()]
        x[ind] = 1.
        return x

    def decode(self, one_hot):
        return self.values_enum_r[torch.where(one_hot)[0].item()]


# """
# TESTING
# """
#
# nt = NTCode(values_list=['A', 'U', 'C', 'G', 'P', 'c', 'a', 'u', 't', 'g'])
# print(nt.encode('g'))
# print(nt.encode('G'))
# assert nt.decode(nt.encode('G')) == 'G'

# import json
# dict_feat = json.load(open('all_annots.json','r'))
# ': None, '.join([string[5:] for string in list(dict_feat.keys())])

class OneHotEncoder:
    """
    To one-hot encode this feature.
    """

    def __init__(self, mapping, num_values=None):
        self.mapping = mapping
        self.reverse_mapping = {value: key for key, value in mapping.items()}
        if num_values is None:
            num_values = max(mapping.values())
        self.num_values = num_values

    def encode(self, value):
        """ Assign encoding of `value` according to known possible
        values.
        """
        x = torch.zeros(self.num_values)
        try:
            ind = self.mapping[value]
            x[ind] = 1.
            return x
        except KeyError:
            return x

    def decode(self, one_hot):
        try:
            decoded = self.reverse_mapping[torch.where(one_hot)[0].item()]
            return decoded
        except KeyError:
            return None


class FloatEncoder:
    """
    To encode floats
    """

    def __init__(self, default_value=0):
        self.default_value = default_value

    def encode(self, value):
        """ Assign encoding of `value` according to known possible
        values.
        """
        if value is None:
            return self.encode_default()
        x = torch.tensor([value], dtype=torch.float)
        return x

    def encode_default(self):
        x = torch.tensor([self.default_value], dtype=torch.float)
        return x

    def decode(self, value):
        return value.item()


class BoolEncoder:
    """
    To encode bools
    """

    def __init__(self, default_value=False):
        self.default_value = default_value

    def encode(self, value):
        """ Assign encoding of `value` according to known possible
        values.
        """
        # Sometimes we encode other stuff as booleans. Then if it's here return True, else False
        if not isinstance(value, bool):
            x = torch.tensor([True], dtype=torch.float)
            return x
        x = torch.tensor([value], dtype=torch.float)
        return x

    def encode_default(self):
        x = torch.tensor([self.default_value], dtype=torch.float)
        return x

    def decode(self, value):
        return value.item()


class ListEncoder:
    """
    To encode bools
    """

    def __init__(self, list_length):
        size = [list_length]
        self.default_value = torch.zeros(size=size, dtype=torch.float)

    def encode(self, value):
        """ Assign encoding of `value` according to known possible
        values.
        """
        if value is None or any([val is None for val in value]):
            return self.encode_default()
        x = torch.tensor(value, dtype=torch.float)
        return x

    def encode_default(self):
        return self.default_value

    def decode(self, value):
        return value.item()


# Interesting Counters :
# To get those, run 'get all labels with the counter option. This is useful to produce the
# one hot encoding (by discarding the really scarce ones
# node dbn : 
# {'(': 1595273, '.': 2694367, ')': 1596160, '[': 52080, ']': 51598, '{': 9862, '}': 9916, '>': 2076, '<': 2078,
# 'A': 529, 'a': 524, 'B': 30, 'b': 29, 'O': 1, 'P': 1, 'C': 1, 'Q': 1, 'D': 1, 'E': 1, 'R': 1, 'F': 1, 'G': 1, 
# 'S': 1, 'H': 1, 'I': 1, 'T': 1, 'J': 1, 'K': 1, 'U': 1, 'L': 1, 'M': 1, 'V': 1, 'N': 1}
# Node puckering :
# {"C3'-endo": 4899464, "C4'-exo": 109199, "C2'-exo": 182706, "C3'-exo": 38963, "O4'-endo": 18993, "O4'-exo": 1143,
#  "C4'-endo": 2825, "C2'-endo": 720607, "C1'-exo": 36287, "C1'-endo": 4277, '': 78}

# Then to produce a default one hot, use : {k: v for v, k in enumerate(dict_to_one_hot)}

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


def build_node_feature_parser(asked_features=None):
    if asked_features is None:
        return {}
    global NODE_FEATURE_MAP
    if any([feature not in NODE_FEATURE_MAP for feature in asked_features]):
        problematic_keys = tuple([feature for feature in asked_features if feature not in NODE_FEATURE_MAP])
        raise ValueError(f'{problematic_keys} were asked as a feature or target but do not exist')

    # filter out the None, we don't know how to encode those...
    encoding_features = [feature for feature in asked_features if NODE_FEATURE_MAP[feature] is not None]
    if len(encoding_features) < len(asked_features):
        unencodable_keys = [feature for feature in asked_features if NODE_FEATURE_MAP[feature] is None]
        print(f'{unencodable_keys} were asked as a feature or target but do not exist')
    subset_dict = {k: NODE_FEATURE_MAP[k] for k in encoding_features}
    return subset_dict


def build_edge_feature_parser(asked_features=None):
    raise NotImplementedError
