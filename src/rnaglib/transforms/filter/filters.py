from typing import Iterator, Any, Callable
import requests

import networkx as nx
from rnaglib.transforms import FilterTransform

""" Filters return a boolean after receiving an RNA.
This can be used to exclude RNAs from a datataset based on some
desired conditione.
"""


class DummyFilter(FilterTransform):
    """Always return True"""

    def forward(self, rna_dict: dict) -> bool:
        return True


class SizeFilter(FilterTransform):
    """Reject RNAs that are not in the given size bounds.

    :param min_size: smallest allowed number of residues
    :param max_size: largest allowed number of residues. Default -1 which means no upper bound.
    """

    def __init__(self, min_size: int = 0, max_size: int = -1, **kwargs):
        self.min_size = min_size
        self.max_size = max_size
        super().__init__(**kwargs)

    def forward(self, rna_dict: dict) -> bool:
        n = len(rna_dict["rna"].nodes())
        if self.max_size == -1:
            return n > self.min_size
        else:
            return n > self.min_size and n < self.max_size


class RNAAttributeFilter(FilterTransform):
    """Reject RNAs that lack a certain annotation at the whole RNA level.

    :param attribute: which RNA-level attribute to look for.
    """

    def __init__(self, attribute: str, value_checker: Callable = None, **kwargs):
        self.attribute = attribute
        self.value_checker = value_checker
        super().__init__(**kwargs)
        pass

    def forward(self, data: dict):
        try:
            val = data["rna"].graph[self.attribute]
        except KeyError:
            return False
        else:
            return self.value_checker(val)


class ResidueAttributeFilter(FilterTransform):
    """Reject RNAs that lack a certain annotation at the whole residue-level.

    :param attribute: which node-level attribute to look for.
    :param aggregation_mode: str (either "aggfunc" or "min_valid"); if set to "aggfunc", keeps an RNA if the output of
    the aggregation function of the residue attribute at the RNA level passes the value_checker; if set to "min_valid", 
    keeps an RNA if more than min_valid nodes pass the value_checker
    :param value_checker: function with accepts the value of the desired attribute and returns True/False
    :param aggfunc: function to aggregate the residue labels at the RNA level (only if aggregarion_mode is "aggfunc")
    :param min_valid: minium number of valid nodes that pass the filter for keeping the RNA. (only if aggregation_mode
    is "min_valid")


    Example
    ---------

    Keep RNAs with at least 1 chemically modified residue::

        >>> from rnaglib.data_loading import RNADataset
        >>> from rnaglib.transforms import ResidueAttributeFilter

        >>> dset = RNADataset(debug=True)
        >>> t = ResidueAttributeFilter(attribute='is_modified',
                                   value_checker: lambda val: val == True,
                                   min_valid=1)
        >>> len(dset)
        >>> rnas = list(t(dset))
        >>> len(rnas)


    """

    def __init__(
            self,
            attribute: str,
            aggregation_mode: str = "min_valid",
            value_checker: Callable = None,
            min_valid: int = 1,
            aggfunc: Callable = None,
            **kwargs,
    ):
        self.attribute = attribute
        self.aggregation_mode = aggregation_mode
        self.min_valid = min_valid
        self.aggfunc = aggfunc
        self.value_checker = value_checker
        super().__init__(**kwargs)
        pass

    def forward(self, data: dict):
        n_valid = 0
        g = data["rna"]
        if self.aggregation_mode == "aggfunc":
            vals_list = []
        for node, ndata in g.nodes(data=True):
            try:
                val = ndata[self.attribute]
            except KeyError:
                continue
            else:
                if self.aggregation_mode == "min_valid" and self.value_checker(val):
                    n_valid += 1
                elif self.aggregation_mode == "aggfunc":
                    vals_list.append(val)
            if self.aggregation_mode == "min_valid" and n_valid >= self.min_valid:
                return True
        if self.aggregation_mode == "min_valid":
            return False
        else:
            return self.aggfunc(vals_list)


class ResidueNameFilter(FilterTransform):
    def __init__(
            self,
            value_checker: Callable = None,
            min_valid: int = 1,
            **kwargs,
    ):
        self.min_valid = min_valid
        self.value_checker = value_checker
        super().__init__(**kwargs)
        pass

    def forward(self, data: dict):
        n_valid = 0
        g = data["rna"]
        for node, ndata in g.nodes(data=True):
            if self.value_checker(node):
                n_valid += 1
            if n_valid >= self.min_valid:
                return True
        return False


class RibosomalFilter(FilterTransform):
    """Remove RNA if ribosomal"""

    ribosomal_keywords = ["ribosomal", "rRNA", "50S", "30S", "60S", "40S"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def forward(self, data: dict):
        pdbid = data["rna"].graph["pdbid"][0]
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdbid}"
        response = requests.get(url)

        data = response.json()

        # Check title and description
        title = data.get("struct", {}).get("title", "").lower()
        if any(keyword in title for keyword in self.ribosomal_keywords):
            return False
        # Check keywords
        keywords = data.get("struct_keywords", {}).get("pdbx_keywords", "").lower()
        if any(keyword in keywords for keyword in self.ribosomal_keywords):
            return False

        # Check polymer descriptions (for RNA and ribosomal proteins)
        for polymer in data.get("polymer_entities", []):
            description = polymer.get("rcsb_polymer_entity", {}).get("pdbx_description", "").lower()
            if any(keyword in description for keyword in self.ribosomal_keywords):
                return False

        return True


class NameFilter(FilterTransform):
    """
    Filter RNAs based on their names.

    This filter keeps only the RNAs whose names are present in the provided list.

    :param names: A list of RNA names to keep.
    """

    def __init__(self, names: list, **kwargs):
        self.names = {name.lower() for name in names}
        super().__init__(**kwargs)

    def forward(self, data: dict) -> bool:
        """
        Check if the RNA's name is in the list of allowed names.

        :param data: Dictionary containing RNA data.
        :return: True if the RNA's name is in the allowed list, False otherwise.
        """
        rna_name = data["rna"].name
        return rna_name in self.names


class ChainFilter(FilterTransform):
    """
    Filter RNAs based on valid chain names for each structure.
    Keeps any RNA with at least one residue having a valid chain name,
    and removes residues with invalid chain names from kept RNAs.

    :param valid_chains_dict: Dictionary mapping structure names to lists of valid chain names.
    """

    def __init__(self, valid_chains_dict: dict, **kwargs):
        self.valid_chains_dict = {
            pdb.lower(): [chain for chain in chains] for pdb, chains in valid_chains_dict.items()  # .upper()
        }
        super().__init__(**kwargs)

    def forward(self, data: dict) -> bool:
        g = data["rna"]
        structure_name = g.name
        valid_chains = set(self.valid_chains_dict.get(structure_name, []))
        nodes_to_remove = []
        has_valid_node = False

        for node, ndata in g.nodes(data=True):
            chain_name = node.split(".")[1]  # .upper()
            if chain_name in valid_chains:
                has_valid_node = True
            else:
                nodes_to_remove.append(node)

        if has_valid_node:
            # Remove nodes with invalid chain names
            g.remove_nodes_from(nodes_to_remove)
            return True
        else:
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(valid_chains_dict={self.valid_chains_dict})"


class ResolutionFilter(RNAAttributeFilter):
    def __init__(self, resolution_threshold: float, **kwargs):
        def value_checker(val):
            try:
                return float(val[0]) < resolution_threshold
            except:
                return False

        super().__init__(attribute="resolution_high", value_checker=value_checker, **kwargs)
