r"""Utility package for computing evaluation metrics for network aligment."""

import copy

from .hits import *
from .mrr import *

__all__ = [
    'hits_ks_scores',
    'mrr_score'
]

classes = copy.copy(__all__)
