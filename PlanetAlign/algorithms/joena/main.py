import torch
import time
import psutil
import os

from PlanetAlign.data import Dataset
from PlanetAlign.utils import get_anchor_pairs, get_batch_rwr_scores
from PlanetAlign.metrics import hits_ks_scores, mrr_score
from PlanetAlign.algorithms.base_model import BaseModel

from .model import MLP, FusedGWLoss


class JOENA(BaseModel):
    """OT-based method JOENA for pairwise network alignment.
    JOENA is proposed by the paper "`Joint Optimal Transport and Embedding for Network Alignment <https://arxiv.org/pdf/2502.19334>`_"
    in WWW 2025.

    Parameters
    ----------
    alpha : float, optional
        The hyparameter balancing the Wasserstein and Gromov-Wasserstein distances. Default is 0.7.
    gamma_p : float, optional
        The weight of proximal operator. Default is 1e-2.
    init_lambda : float, optional
        The initial value of the threshold lambda. Default is 1.0.
    hid_dim : int, optional
        The hidden dimension of the MLP. Default is 128.
    out_dim : int, optional
        The output dimension of the MLP. Default is 128.
    lr : float, optional
        The learning rate of the optimizer. Default is 1e-4.
    dtype : torch.dtype, optional
        Data type of the tensors, choose from torch.float32 or torch.float64. Default is torch.float32.
    """
    def __init__(self,
                 alpha: float = 0.7,
                 gamma_p: float = 1e-2,
                 init_lambda: float = 1.0,
                 hid_dim: int = 128,
                 out_dim: int = 128,
                 lr: float = 1e-4,
                 dtype: torch.dtype = torch.float32):
        super(JOENA, self).__init__(dtype=dtype)

        self.alpha = alpha
        self.gamma_p = gamma_p
        self.init_lambda = init_lambda
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.lr = lr

    def train(self,
              dataset: Dataset,
              gid1: int,
              gid2: int,
              use_attr: bool = True,
              total_epochs: int = 100,
              save_log: bool = True,
              verbose: bool = True):
        """
        Parameters
        ----------
        dataset : Dataset
            The dataset containing graphs to be aligned and the training/test data.
        gid1 : int
            The index of the first graph to be aligned.
        gid2 : int
            The index of the second graph to be aligned.
        use_attr : bool, optional
            Whether to use node attributes for alignment. Default is True.
        total_epochs : int, optional
            The total number of training epochs. Default is 100.
        save_log : bool, optional
            Whether to save the training log. Default is True.
        verbose : bool, optional
            Whether to print the training progress. Default is True.
        """

        self.check_inputs(dataset, (gid1, gid2), plain_method=False, use_attr=use_attr, pairwise=True, supervised=True)

        logger = self.init_training_logger(dataset, use_attr, additional_headers=['memory', 'infer_time'], save_log=save_log)
        process = psutil.Process(os.getpid())

        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        n1, n2 = graph1.num_nodes, graph2.num_nodes
        anchor_links = get_anchor_pairs(dataset.train_data, gid1, gid2)
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)

        rwr_t0 = time.time()
        rwr_emb1 = get_batch_rwr_scores(graph1, anchor_links[:, 0], device=self.device).cpu().to(self.dtype)
        rwr_emb2 = get_batch_rwr_scores(graph2, anchor_links[:, 1], device=self.device).cpu().to(self.dtype)
        rwr_time = time.time() - rwr_t0
        if use_attr:
            node_attr1, node_attr2 = graph1.x.to(self.dtype), graph2.x.to(self.dtype)
            input_emb1 = torch.concatenate((node_attr1, rwr_emb1), dim=1)
            input_emb2 = torch.concatenate((node_attr2, rwr_emb2), dim=1)
        else:
            input_emb1 = rwr_emb1
            input_emb2 = rwr_emb2
        input_emb1, input_emb2 = input_emb1.to(self.device), input_emb2.to(self.device)

        gw_weight = self.alpha / (1 - self.alpha) * min(n1, n2) ** 0.5

        # Initialize model
        model = MLP(input_dim=input_emb1.shape[1],
                    hidden_dim=self.hid_dim,
                    output_dim=self.out_dim).to(self.dtype).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = FusedGWLoss(graph1, graph2, gw_weight=gw_weight, gamma_p=self.gamma_p, init_lambda=self.init_lambda,
                                in_iter=5, out_iter=10, dtype=self.dtype).to(self.device)

        # Training
        S = torch.ones(n1, n2, dtype=self.dtype).to(self.device) / (n1 * n2)
        for epoch in range(total_epochs):
            refer_time = rwr_time
            t0 = time.time()
            model.train()
            optimizer.zero_grad()
            ref_t0 = time.time()
            out1, out2 = model(input_emb1, input_emb2)
            loss, S, _ = criterion(out1=out1, out2=out2)
            refer_time += time.time() - ref_t0
            loss.backward()
            optimizer.step()
            t1 = time.time()

            # testing
            with torch.no_grad():
                model.eval()
                hits, mrr = hits_ks_scores(S, test_pairs, mode='mean'), mrr_score(S, test_pairs, mode='mean')
                mem_gb = process.memory_info().rss / 1024 ** 3
                logger.log(epoch=epoch+1,
                           loss=loss.item(),
                           epoch_time=t1-t0,
                           hits=hits,
                           mrr=mrr,
                           memory=round(mem_gb, 4),
                           infer_time=round(refer_time, 4),
                           verbose=verbose)

        return S.detach(), logger
