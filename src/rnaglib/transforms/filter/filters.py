from typing import Iterator, Any, Callable
import requests

import networkx as nx
from rnaglib.transforms import FilterTransform

""" Filters return a boolean after receiving an RNA.
This can be used to exclude RNAs from a datataset based on some
desired conditione.
"""


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

    def __init__(self, attribute: str, **kwargs):
        self.attribute = attribute
        super().__init__(**kwargs)
        pass

    def forward(self, data: dict):
        try:
            annot = data["rna"].graph[self.attribute]
        except KeyError:
            return False
        else:
            if annot is None:
                return False
        return True

    pass


class ResidueAttributeFilter(FilterTransform):
    """Reject RNAs that lack a certain annotation at the whole residue-level.

    :param attribute: which node-level attribute to look for.
    :param value_checker: function with accepts the value of the desired attribute and returns True/False
    :param min_valid: minium number of valid nodes that pass the filter for keeping the RNA.


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
        value_checker: Callable = None,
        min_valid: int = 1,
        **kwargs,
    ):
        self.attribute = attribute
        self.min_valid = min_valid
        self.value_checker = value_checker
        super().__init__(**kwargs)
        pass

    def forward(self, data: dict):
        n_valid = 0
        g = data["rna"]
        for node, ndata in g.nodes(data=True):
            try:
                val = ndata[self.attribute]
            except KeyError:
                continue
            else:
                if self.value_checker(val):
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
            description = (
                polymer.get("rcsb_polymer_entity", {})
                .get("pdbx_description", "")
                .lower()
            )
            if any(keyword in description for keyword in self.ribosomal_keywords):
                return False

        return True
