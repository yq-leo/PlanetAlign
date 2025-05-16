import time
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from PyNetAlign.data import Dataset
from PyNetAlign.utils import get_anchor_pairs
from PyNetAlign.metrics import hits_ks_scores, mrr_score
from PyNetAlign.algorithms.base_model import BaseModel


class MEAformer(BaseModel):
    """
    Embedding-based method MEAformer for pairwise entity alignment. DualMatch is proposed by the paper
    "`MEAformer: Multi-modal Entity Alignment Transformer for Meta Modality Hybrid <https://arxiv.org/pdf/2212.14454>`_"
    in ACM MM 2023.
    """

    def __init__(self,
                 dtype: torch.dtype = torch.float32):
        super(MEAformer, self).__init__(dtype=dtype)

    def train(self,
              dataset: Dataset,
              gid1: int,
              gid2: int,
              use_attr: bool = True,
              save_log: bool = True,
              verbose: bool = True):
        pass
