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
# ': None, '.join([string.lstrip('node_').lstrip('_edge') for string in list(dict_feat.keys())])

FEATURE_MAP = {
    'index': None,
    'index_chain': None,
    'chain_name': None,
    'nt_resnum': None,
    "nt_name": None,
    'nt_code': OneHotEncoder(mapping={'A': 0, 'U': 1, 'C': 2, 'G': 3, 'a': 0, 'u': 1, 'c': 2, 'g': 3}, num_values=4),
    "nt_id": None,
    "nt_type": None,
    "dbn": None,
    "summary": None,
    "alpha": None,
    "beta": None,
    "gamma": None,
    "delta": None,
    "epsilon": None,
    "zeta": None,
    "epsilon_zeta": None,
    "bb_type": None,
    "chi": None,
    "glyco_bond": None,
    "C5prime_xyz": None,
    "P_xyz": None,
    "form": None,
    "ssZp": None,
    "Dp": None,
    "splay_angle": None,
    "splay_distance": None,
    "splay_ratio": None,
    "eta": None,
    "theta": None,
    "eta_prime": None,
    "theta_prime": None,
    "eta_base": None,
    "theta_base": None,
    "v0": None,
    "v1": None,
    "v2": None,
    "v3": None,
    "v4": None,
    "amplitude": None,
    "phase_angle": None,
    "puckering": None,
    "sugar_class": None,
    "bin": None,
    "cluster": None,
    "suiteness": None,
    "filter_rmsd": None,
    "frame_rmsd": None,
    "frame_origin": None,
    "frame_x_axis": None,
    "frame_y_axis": None,
    "frame_z_axis": None,
    "frame_quaternion": None,
    "sse_sse": None,
    "binding_protein": None,
    "binding_ion": None,
    "binding_small-molecule": None,
    "LW": None,
    "backbone": None,
    "nt1": None,
    "nt2": None,
    "bp": None,
    "name": None,
    "Saenger": None,
    "DSSR": None,
    "binding_protein_id": None,
    "binding_protein_nt-aa": None,
    "binding_protein_nt": None,
    "binding_protein_aa": None,
    "binding_protein_Tdst": None,
    "binding_protein_Rdst": None,
    "binding_protein_Tx": None,
    "binding_protein_Ty": None,
    "binding_protein_Tz": None,
    "binding_protein_Rx": None,
    "binding_protein_Ry": None,
    "binding_protein_Rz": None,
    "is_modified": None,
    "is_broken": None

}
