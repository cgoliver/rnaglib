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


class OneHotEncoder:
    """ This feature represents the standard nucleotide type (A,U,C,G,..).
    We one-hot encode this feature.
    """

    def __init__(self, mapping, num_values=None, default_value='zero'):
        self.mapping = mapping
        self.reverse_mapping = {value: key for key, value in mapping.items()}
        if num_values is None:
            num_values = max(mapping.values())
        self.num_values = num_values
        self.default_value = default_value

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


"""
TESTING
"""

nt = NTCode(values_list=['A', 'U', 'C', 'G', 'P', 'c', 'a', 'u', 't', 'g'])
print(nt.encode('g'))
print(nt.encode('G'))
assert nt.decode(nt.encode('G')) == 'G'

# import json
# dict_feat = json.load(open('all_annots.json','r'))
# ': None, '.join([string[5:] for string in list(dict_feat.keys())])

FEATURE_MAP = {
    'index': None,       #    Not a feature
    'index_chain': None,       #     Not a feature
    'chain_name': None,       #     Not a feature
    'nt_resnum': None,       #     Not a feature
    "nt_name": None,  # This looks crappy to me but maybe there are some that are canonical and a lot of modified ones ?
    'nt_code': OneHotEncoder(mapping={'A': 0, 'U': 1, 'C': 2, 'G': 3, 'a': 0, 'u': 1, 'c': 2, 'g': 3}, num_values=4),
    "nt_id": None,       #     TODO onehot
    "nt_type": None,       #   Constant = RNA
    "dbn": None,       #     TODO onehot
    "summary": None,       # TODO onehot
    "alpha": None,       # TODO :  float
    "beta": None,       # TODO : float
    "gamma": None,       # TODO : float
    "delta": None,       # TODO : float
    "epsilon": None,       # TODO : float
    "zeta": None,       # TODO : float
    "epsilon_zeta": None,       # TODO : float
    "bb_type":  None,       # TODO : onehot
    "chi":  None,       # TODO :  TODO : float
    "glyco_bond":  None,       # TODO : onehot
    "C5prime_xyz":  None,       # TODO : list
    "P_xyz":  None,       # TODO : list
    "form":  None,       # TODO : onehot
    "ssZp":  None,       # TODO : float
    "Dp":  None,       # TODO : float
    "splay_angle":  None,       # TODO : float
    "splay_distance":  None,       # TODO : float
    "splay_ratio":  None,       # TODO : float
    "eta":  None,       # TODO : float
    "theta":  None,       # TODO : float
    "eta_prime":  None,       # TODO : float
    "theta_prime":  None,       # TODO : float
    "eta_base":  None,       # TODO : float
    "theta_base":  None,       # TODO : float
    "v0":  None,       # TODO : float
    "v1":  None,       # TODO : float
    "v2":  None,       # TODO : float
    "v3":  None,       # TODO : float
    "v4":  None,       # TODO : float
    "amplitude":  None,       # TODO : float
    "phase_angle":  None,       # TODO : float
    "puckering":  None,       # TODO : onehot
    "sugar_class":  None,       # TODO : onehot
    "bin":  None,       # TODO : onehot
    "cluster":  None,       # TODO : onehot
    "suiteness":  None,       # TODO : float
    "filter_rmsd":  None,       # TODO : float
    "frame_rmsd":  None,       # TODO : float
    "frame_origin":  None,       # TODO : list
    "frame_x_axis":  None,       # TODO : list
    "frame_y_axis":  None,       # TODO : list
    "frame_z_axis":  None,       # TODO : list
    "frame_quaternion":  None,       # TODO : list
    "sse_sse":  None,       # TODO : ?onehot?
    "binding_protein":  None,       # TODO : constant None ?
    "binding_ion":  None,       # TODO : ? onehot ?
    "binding_small-molecule":  None,       # TODO : ? onehot ?
    "LW":  None,       # TODO : onehot
    "backbone":  None,       # TODO : bool Constant True
    "nt1":  None,       # TODO : trash
    "nt2":  None,       # TODO : trash
    "bp":  None,       # TODO : trash
    "name":  None,       # TODO : trash
    "Saenger":  None,       # TODO : trash
    "DSSR":  None,       # TODO : trash
    "binding_protein_id":  None,       # TODO : trash
    "binding_protein_nt-aa":  None,       # TODO : trash
    "binding_protein_nt":  None,       # TODO : trash
    "binding_protein_aa":  None,       # TODO : trash
    "binding_protein_Tdst":  None,       # TODO : float
    "binding_protein_Rdst":  None,       # TODO : float
    "binding_protein_Tx":  None,       # TODO : float
    "binding_protein_Ty":  None,       # TODO : float
    "binding_protein_Tz":  None,       # TODO : float
    "binding_protein_Rx":  None,       # TODO : float
    "binding_protein_Ry":  None,       # TODO : float
    "binding_protein_Rz":  None,       # TODO : float
    "is_modified":  None,       # TODO : bool Constant True
    "is_broken": None,       # TODO : bool Constant True
}
