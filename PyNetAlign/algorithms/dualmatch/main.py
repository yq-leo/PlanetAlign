import time
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from PyNetAlign.data import Dataset
from PyNetAlign.utils import get_anchor_pairs
from PyNetAlign.metrics import hits_ks_scores, mrr_score
from PyNetAlign.algorithms.base_model import BaseModel


class DualMatch(BaseModel):
    """
    Embedding-based method DualMatch for pairwise unsupervised entity alignment. DualMatch is proposed by the
    paper "`Unsupervised Entity Alignment for Temporal Knowledge Graphs <https://arxiv.org/pdf/2302.00796>`_" in WWW
    2023.
    """

    def __init__(self,
                 dtype: torch.dtype = torch.float32):
        super(DualMatch, self).__init__(dtype=dtype)

    def train(self,
              dataset: Dataset,
              gid1: int,
              gid2: int,
              use_attr: bool = True,
              save_log: bool = True,
              verbose: bool = True):
        pass
