from typing import List, Union, Any

import networkx as nx

class Transform:
    """ Abstract class for modifying and updating an RNA graph.
    Receives an RNA graph and returns an RNA graph.
    This can be applied at dataset construction time, or a retrieval.
    """
    def __call__(self, data: Any) -> Any:
        return self.forward(data['rna'])

    def forward(self):
        raise NotImplementedError

class Compose(Transform):
    """ String together and apply a list of transforms """
    def __init__(self, transforms):
        self.transforms = transforms
        pass

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            if isinstance(list, data):
                data = [transform(d) for d in data]
            else:
                data = transform(data)
        return data
