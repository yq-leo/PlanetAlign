import time
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import psutil
import os

from PyNetAlign.data import Dataset
from PyNetAlign.utils import get_anchor_pairs, get_batch_rwr_scores, pairwise_cosine_similarity
from PyNetAlign.metrics import hits_ks_scores, mrr_score
from PyNetAlign.algorithms.base_model import BaseModel

from .model import BrightUNet, BrightANet, MarginalRankingLoss


class BRIGHT(BaseModel):
    """Embedding-based method BRIGHT for pairwise network alignment.
    BRIGHT is proposed by the paper "`BRIGHT: A Bridging Algorithm for Network Alignment. <https://doi.org/10.1145/3442381.3450053>`_"
    in WWW 2021.

    Parameters
    ----------
    restart_prob : float, optional
        The restart probability for random walk with restart. Default is 0.15.
    out_dim : int, optional
        The dimension of the output embeddings. Default is 128.
    neg_sample_size : int, optional
        The number of negative samples per anchor link. Default is 500.
    margin : float, optional
        The margin for the ranking loss. Default is 10.
    lr : float, optional
        The learning rate for the optimizer. Default is 1e-3.
    dtype : torch.dtype, optional
        Data type of the tensors, choose from torch.float32 or torch.float64. Default is torch.float32.
    """
    def __init__(self,
                 restart_prob: float = 0.15,
                 out_dim: int = 128,
                 neg_sample_size: int = 500,
                 margin: float = 10,
                 lr: float = 1e-3,
                 dtype: torch.dtype = torch.float32):
        super(BRIGHT, self).__init__(dtype=dtype)

        self.restart_prob = restart_prob
        self.out_dim = out_dim
        self.neg_sample_size = neg_sample_size
        self.margin = margin
        self.lr = lr

    def train(self,
              dataset: Dataset,
              gid1: int,
              gid2: int,
              use_attr: bool = True,
              total_epochs: int = 250,
              save_log: bool = True,
              verbose: bool = True):
        """
        Parameters
        dataset : Dataset
            The dataset containing the graphs to be aligned and the training/test data.
        gid1 : int
            The index of the first graph in the dataset to be aligned.
        gid2 : int
            The index of the second graph in the dataset to be aligned.
        use_attr : bool, optional
            Whether to use node and edge attributes for alignment. Default is True.
        save_log: bool, optional
            Whether to save the training log. Default is True.
        verbose : bool, optional
            Whether to print the progress during training. Default is True.
        """

        self.check_inputs(dataset, (gid1, gid2), plain_method=False, use_attr=use_attr, pairwise=True, supervised=True)

        logger = self.init_training_logger(dataset, use_attr, additional_headers=['memory', 'infer_time'], save_log=save_log)
        process = psutil.Process(os.getpid())

        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        anchor_links = get_anchor_pairs(dataset.train_data, gid1, gid2)
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)

        n1, n2 = graph1.num_nodes, graph2.num_nodes
        neg_sample_size = self.neg_sample_size if self.neg_sample_size < min(n1, n2) else min(n1, n2)

        # Initialization
        if verbose:
            print('Constructing RWR embeddings...', end=' ')
        start = time.time()
        rwr_emb1 = get_batch_rwr_scores(graph1, anchor_links[:, 0], self.restart_prob).to(self.dtype)
        rwr_emb2 = get_batch_rwr_scores(graph2, anchor_links[:, 1], self.restart_prob).to(self.dtype)
        if verbose:
            print(f'Done, Time Spent: {time.time() - start:.2f}s')
        rwr_time = time.time() - start

        # Model
        num_anchor_links = anchor_links.shape[0]
        if not use_attr:
            model = BrightUNet(in_dim=num_anchor_links, out_dim=self.out_dim).to(self.dtype).to(self.device)
        else:
            num_attr = graph1.x.shape[1]
            model = BrightANet(rwr_dim=num_anchor_links, in_dim=num_attr, out_dim=self.out_dim).to(self.dtype).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = MarginalRankingLoss(k=neg_sample_size, margin=self.margin)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

        # Training
        rwr_emb1, rwr_emb2 = rwr_emb1.to(self.device), rwr_emb2.to(self.device)
        x1 = graph1.x.to(self.device) if use_attr else None
        x2 = graph2.x.to(self.device) if use_attr else None
        edge_index1, edge_index2 = graph1.edge_index.to(self.device), graph2.edge_index.to(self.device)
        anchor_links = anchor_links.to(self.device)
        test_pairs = test_pairs.to(self.device)

        out1, out2 = None, None
        for epoch in range(total_epochs):
            t0 = time.time()
            model.train()
            optimizer.zero_grad()
            out1, out2 = model(rwr_emb1, rwr_emb2, x1, x2, edge_index1, edge_index2)
            infer_time = time.time() - t0 + rwr_time
            loss = criterion(out1=out1, out2=out2, anchor_links=anchor_links)
            loss.backward()
            optimizer.step()
            scheduler.step()
            t1 = time.time()

            model.eval()
            out1, out2 = out1.detach(), out2.detach()
            with torch.no_grad():
                S = pairwise_cosine_similarity(out1, out2)
                hits = hits_ks_scores(S, test_pairs, mode='mean')
                mrr = mrr_score(S, test_pairs, mode='mean')
                mem_gb = process.memory_info().rss / 1024 ** 3
                logger.log(epoch=epoch+1,
                           loss=loss.item(),
                           epoch_time=t1-t0,
                           hits=hits,
                           mrr=mrr,
                           memory=round(mem_gb, 4),
                           infer_time=round(infer_time, 4),
                           verbose=verbose)

        return out1, out2, logger
        