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
        self.values_list = values_list
        self.clean_values()

        self.min_val = min_val
        self.max_val = max_val

        if not values_list is None:
            self.dims = len(values_list)
            self.values_enum = {v:k for k,v in enumerate(self.values_list)}
        else:
            self.dims = 1
        pass

    def encode(self, value):
        raise NotImplementedError

    def clean_values(self):
        pass

class NTCode(FeatureMap):

    def clean_values(self):
        self.values_list = sorted(list(set(map(str.upper, self.values_list))))

    def encode(self, value):
        """ Assign encoding of `value` according to known possible
        values.
        """
        assert type(value) is str, "Wrong input type"
        x = torch.zeros(self.dims)
        ind = self.values_enum[value.upper()]
        x[ind] = 1.
        return x


"""
TESTING
"""

nt = NTCode(values_list=['A', 'U', 'C', 'G', 'P', 'c', 'a', 'u', 't', 'g'])
print(nt.encode('g'))
print(nt.encode('G'))
