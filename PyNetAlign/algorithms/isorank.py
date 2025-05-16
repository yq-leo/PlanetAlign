import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import time
import psutil
import os

from PyNetAlign.data import Dataset
from PyNetAlign.utils import get_anchor_pairs
from PyNetAlign.metrics import hits_ks_scores, mrr_score

from .base_model import BaseModel


class IsoRank(BaseModel):
    """Consistency-based method IsoRank for pairwise plain network alignment.
    IsoRank is proposed by the paper "`Global alignment of multiple protein interaction networks with application to functional orthology detection <https://www.pnas.org/doi/full/10.1073/pnas.0806627105>`_"
    in PNAS 2008.

    Parameters
    ----------
    alpha: float, optional
        The decay factor for the optimization. Default is 0.4.
    dtype: torch.dtype, optional
        Data type of the tensors, choose from torch.float32 or torch.float64. Default is torch.float32.
    """
    def __init__(self,
                 alpha: float = 0.4,
                 dtype: torch.dtype = torch.float32):
        super(IsoRank, self).__init__(dtype=dtype)
        assert 0 <= alpha <= 1, 'Alpha must be in [0, 1]'
        self.alpha = alpha

    def train(self,
              dataset: Dataset,
              gid1: int,
              gid2: int,
              use_attr: bool = False,
              total_epochs: int = 100,
              tol: float = 1e-10,
              save_log: bool = True,
              verbose: bool = True):
        """
        Parameters
        ----------
        dataset : Dataset
            The dataset containing graphs to be aligned and the training/test data.
        gid1 : int
            The graph id of the first graph to be aligned.
        gid2 : int
            The graph id of the second graph to be aligned.
        use_attr : bool, optional
            Flag for using attributes. **Must be False for IsoRank**. Default is False.
        total_epochs : int, optional
            Maximum number of training epochs. Default is 100.
        tol : float, optional
            Tolerance for convergence. Default is 1e-10.
        save_log : bool, optional
            Flag for saving the logs. Default is True.
        verbose : bool, optional
            Flag for printing the logs. Default is True.
        Returns
        -------
        torch.Tensor
            The final similarity matrix **S**.
        """

        assert tol > 0, 'Tolerance must be positive'
        self.check_inputs(dataset, (gid1, gid2), plain_method=True, use_attr=use_attr, pairwise=True, supervised=True)

        logger = self.init_training_logger(dataset, use_attr, additional_headers=['memory', 'infer_time'], save_log=save_log)
        process = psutil.Process(os.getpid())

        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        n1, n2 = graph1.num_nodes, graph2.num_nodes
        anchor_links = get_anchor_pairs(dataset.train_data, gid1, gid2)
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)

        H = torch.zeros(n1, n2, dtype=self.dtype)
        H[anchor_links[:, 0], anchor_links[:, 1]] = 1
        H = H.to(self.device)

        S = torch.rand(n1, n2, dtype=self.dtype).to(self.device)

        adj1 = to_dense_adj(graph1.edge_index, max_num_nodes=graph1.num_nodes).squeeze().to(self.dtype).to(self.device)
        adj2 = to_dense_adj(graph2.edge_index, max_num_nodes=graph2.num_nodes).squeeze().to(self.dtype).to(self.device)
        col_norm_adj1 = F.normalize(adj1, p=1, dim=0)
        col_norm_adj2 = F.normalize(adj2, p=1, dim=0)

        infer_time = 0
        for i in range(total_epochs):
            t0 = time.time()
            last_S = S
            S = self.alpha * col_norm_adj1 @ S @ col_norm_adj2.T + (1 - self.alpha) * H
            t1 = time.time()
            infer_time += t1 - t0
            diff = torch.norm(S - last_S)
            if diff < tol:
                break
            hits = hits_ks_scores(S, test_pairs, mode='mean')
            mrr = mrr_score(S, test_pairs, mode='mean')
            mem_gb = process.memory_info().rss / 1024 ** 3
            logger.log(epoch=i+1,
                       loss=diff.item(),
                       epoch_time=t1-t0,
                       mrr=mrr,
                       hits=hits,
                       memory=round(mem_gb, 4),
                       infer_time=round(infer_time, 4),
                       verbose=verbose)

        return S, logger
