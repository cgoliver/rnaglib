from .splitting_utils import random_split

from .splitters import Splitter
from .splitters import RandomSplitter, BenchmarkBindingSiteSplitter, DasSplitter
from .similarity_splitter import ClusterSplitter, RNAalignSplitter, CDHitSplitter

__all__ = ['Splitter',
           'RandomSplitter',
           'random_split',
           'BenchmarkBindingSiteSplitter',
           'DasSplitter',
           'ClusterSplitter',
           'RNAalignSplitter',
           'CDHitSplitter'
           ]


classes = __all__

