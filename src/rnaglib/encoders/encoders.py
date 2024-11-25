"""
Functions to build feature map for each feature type.
"""

import torch


class OneHotEncoder:
    def __init__(self, mapping, num_values=None):
        """
        To one-hot encode this feature.

        :param mapping: This is a dictionnary that gives an index for each possible value.
        :param num_values: If the mapping can be many to one, you should specifiy it here.
        """
        self.mapping = mapping
        self.reverse_mapping = {value: key for key, value in mapping.items()}
        if num_values is None:
            num_values = max(mapping.values())
            if 0 in mapping.values():
                num_values += 1
        self.num_values = num_values

    def encode(self, value):
        """
        Assign encoding of `value` according to known possible values.

        :param value: The value to encode. If missing a default vector of full zeroes is produced.
        """
        x = self.encode_default()
        try:
            ind = self.mapping[value]
            x[ind] = 1.0
            return x
        except KeyError:
            return x

    def encode_default(self):
        x = torch.zeros(self.num_values)
        return x

    def decode(self, one_hot):
        try:
            decoded = self.reverse_mapping[torch.where(one_hot)[0].item()]
            return decoded
        except KeyError:
            return None


class FloatEncoder:

    def __init__(self, default_value=0):
        """
        Utility class to encode floats

        :param default_value: The value to return in case of failure
        """
        self.default_value = default_value

    def encode(self, value):
        """
        Assign encoding of `value` according to known possible values.

        :param value: The value to encode. If missing a default value (by default zero) is produced.
        """
        try:
            return torch.tensor([value], dtype=torch.float)
        except:
            return self.encode_default()

    def encode_default(self):
        return torch.tensor([self.default_value], dtype=torch.float)

    def decode(self, value):
        return value.item()


class BoolEncoder:

    def __init__(self, default_value=False):
        """
        To encode bools. A possible encoding is to have no value in which case it defaults to False.

        :param default_value: To switch the default behavior. This is not recommended because not aligned with the data
        """
        self.default_value = default_value

    def encode(self, value):
        """
        Assign encoding of `value` according to known possible values.

        :param value: The value to encode. If missing the default value (False by default) is produced.
        """
        if value is None:
            return self.encode_default()
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
    def __init__(self, list_length):
        """
        To encode lists, cast them as tensor if possible, otherwise just return zeroes.

        :param list_length: We need the lists to be fixed length
        """
        size = [list_length]
        self.default_value = torch.zeros(size=size, dtype=torch.float)

    def encode(self, value):
        """
        Assign encoding of `value` according to known possible values.

        :param value: The value to encode. If missing the default value (A list of zeros) is produced.
        """
        if value is None or any([val is None for val in value]):
            return self.encode_default()
        else:
            try:
                x = torch.tensor(value, dtype=torch.float)
            except:
                return self.encode_default()
        return x

    def encode_default(self):
        return self.default_value

    def decode(self, value):
        return value.item()


class NucleotideEncoder:
    def __init__(self):
        """
        Fixed encoder for ACGU nucleotides.
        Maps: A->1, C->2, G->3, U->4, everything else->0
        """
        self.mapping = {"A": 1, "C": 2, "G": 3, "U": 4}
        self.reverse_mapping = {1: "A", 2: "C", 3: "G", 4: "U", 0: "X"}
        self.num_classes = 5  # Including the default class

    def encode(self, value):
        """
        Convert nucleotide to its integer label.
        Returns 0 for any non-ACGU input.

        :param value: The nucleotide to encode
        :return: Tensor containing the integer label
        """
        try:
            value = value.upper()
            label = self.mapping.get(value, 0)
            return torch.tensor([label], dtype=torch.long)
        except (AttributeError, TypeError):
            return torch.tensor([0], dtype=torch.long)

    def decode(self, value):
        """
        Convert the integer label back to nucleotide
        Returns 'X' for 0 (unknown/non-standard nucleotide)

        :param value: Integer tensor to decode
        :return: Corresponding nucleotide or 'X' for non-standard
        """
        try:
            int_value = value.item()
            return self.reverse_mapping.get(int_value, "X")
        except (AttributeError, TypeError):
            return "X"


# Interesting Counters :
# To get those, run 'get_all_labels with the counter option. This is useful to produce the
# one hot encoding (by discarding the really scarce ones)
# node dbn :
# {'(': 1595273, '.': 2694367, ')': 1596160, '[': 52080, ']': 51598, '{': 9862, '}': 9916, '>': 2076, '<': 2078,
# 'A': 529, 'a': 524, 'B': 30, 'b': 29, 'O': 1, 'P': 1, 'C': 1, 'Q': 1, 'D': 1, 'E': 1, 'R': 1, 'F': 1, 'G': 1,
# 'S': 1, 'H': 1, 'I': 1, 'T': 1, 'J': 1, 'K': 1, 'U': 1, 'L': 1, 'M': 1, 'V': 1, 'N': 1}
# Node puckering :
# {"C3'-endo": 4899464, "C4'-exo": 109199, "C2'-exo": 182706, "C3'-exo": 38963, "O4'-endo": 18993, "O4'-exo": 1143,
#  "C4'-endo": 2825, "C2'-endo": 720607, "C1'-exo": 36287, "C1'-endo": 4277, '': 78}

# Then to produce a default one hot, use : {k: v for v, k in enumerate(dict_to_one_hot)}
