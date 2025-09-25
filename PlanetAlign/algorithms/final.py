import time
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import psutil
import os

from PlanetAlign.data import Dataset
from PlanetAlign.utils import get_anchor_pairs
from PlanetAlign.metrics import hits_ks_scores, mrr_score
from .base_model import BaseModel


class FINAL(BaseModel):
    """Consistency-based method FINAL for pairwise attributed network alignment.
    FINAL is proposed by the paper "`FINAL: Fast Attributed Network Alignment <https://dl.acm.org/doi/abs/10.1145/2939672.2939766>`_"
    in KDD 2016.

    Parameters
    ----------
    alpha : float, optional
        The regularization parameter in the optimization objective. Default is 0.5.
    dtype : torch.dtype, optional
        Data type of the tensors, choose from torch.float32 or torch.float64. Default is torch.float32.
    """

    def __init__(self,
                 alpha: float = 0.5,
                 dtype: torch.dtype = torch.float32):
        super(FINAL, self).__init__(dtype=dtype)
        assert 0 <= alpha <= 1, 'Alpha must be in the range [0, 1]'
        self.alpha = alpha

    def train(self, 
              dataset: Dataset,
              gid1: int,
              gid2: int,
              use_attr: bool = True,
              total_epochs: int = 50,
              tol: float = 1e-10,
              save_log: bool = True,
              verbose: bool = True):
        """
        Parameters
        ----------
        dataset : Dataset
            The dataset containing the graphs to be aligned and the training/test data.
        gid1 : int
            The index of the first graph in the dataset to be aligned.
        gid2 : int
            The index of the second graph in the dataset to be aligned.
        use_attr : bool, optional
            Whether to use node and edge attributes for alignment. Default is True.
        total_epochs : int, optional
            The maximum number of epochs for the optimization. Default is 50.
        tol : float, optional
            The tolerance for convergence. Default is 1e-10.
        save_log : bool, optional
            Whether to save the log of the training process. Default is True.
        verbose : bool, optional
            Whether to print the progress during training. Default is True.
        """

        assert tol > 0, 'Tolerance must be positive'
        self.check_inputs(dataset, (gid1, gid2), plain_method=False, use_attr=use_attr, pairwise=True, supervised=True)

        logger = self.init_training_logger(dataset, use_attr, additional_headers=['memory', 'infer_time'], save_log=save_log)
        process = psutil.Process(os.getpid())

        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        n1, n2 = graph1.num_nodes, graph2.num_nodes
        anchor_links = get_anchor_pairs(dataset.train_data, gid1, gid2)
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)

        # Preprocess
        inf_t0 = time.time()

        adj1 = to_dense_adj(graph1.edge_index, max_num_nodes=graph1.num_nodes).squeeze().to(self.dtype).to(self.device)
        adj2 = to_dense_adj(graph2.edge_index, max_num_nodes=graph2.num_nodes).squeeze().to(self.dtype).to(self.device)
        node_attr1, node_attr2 = self.init_node_feat(graph1, use_attr), self.init_node_feat(graph2, use_attr)
        edge_attr1_adj, edge_attr2_adj = self.init_edge_feat_adj(graph1, use_attr), self.init_edge_feat_adj(graph2, use_attr)

        N = torch.zeros(n2, n1, dtype=self.dtype).to(self.device)
        d = torch.zeros(n2, n1, dtype=self.dtype).to(self.device)
        h = torch.zeros(n2, n1, dtype=self.dtype).to(self.device)
        h[anchor_links[:, 1], anchor_links[:, 0]] = 1
        S = torch.clone(h).T

        num_node_attr = node_attr1.shape[1]
        num_edge_attr = edge_attr1_adj.shape[0]

        # Compute node feature cosine cross-similarity
        for k in range(num_node_attr):
            N += torch.outer(node_attr2[:, k], node_attr1[:, k])

        # Compute the Kronecker degree vector
        start_time = time.time()
        for i in range(num_edge_attr):
            for j in range(num_node_attr):
                d += torch.outer((edge_attr2_adj[i] * adj2 @ node_attr2[:, j]),
                                 (edge_attr1_adj[i] * adj1 @ node_attr1[:, j]))
        if verbose:
            print(f'Time for computing Kronecker degree vector: {time.time() - start_time:.4f} s')

        D = N * d
        maskD = D > 0
        D[maskD] = torch.reciprocal(torch.sqrt(D[maskD]))
        N = N * D

        infer_time = time.time() - inf_t0

        # Optimization
        for epoch in range(total_epochs):
            t0 = time.time()
            prev_S = torch.clone(S)
            M = N * S.T
            sim = torch.zeros_like(N)

            for i in range(num_edge_attr):
                sim += (edge_attr2_adj[i] * adj2) @ M @ (edge_attr1_adj[i] * adj1)

            S = ((1 - self.alpha) * h + self.alpha * N * sim).T
            t1 = time.time()
            infer_time += t1 - t0
            diff = torch.norm(S - prev_S)

            hits = hits_ks_scores(S, test_pairs, mode='mean')
            mrr = mrr_score(S, test_pairs, mode='mean')
            mem_gb = process.memory_info().rss / 1024 ** 3
            logger.log(epoch=epoch+1,
                       loss=diff.item(),
                       epoch_time=t1-t0,
                       mrr=mrr,
                       hits=hits,
                       memory=round(mem_gb, 4),
                       infer_time=round(infer_time, 4),
                       verbose=verbose)

            if diff < tol:
                break

        self.S = S
        return S, logger

    def init_node_feat(self, graph, use_attr=True):
        if graph.x is None or not use_attr:
            node_attr = torch.ones(graph.num_nodes, 1, dtype=self.dtype).to(self.device)
        else:
            node_attr = graph.x.to(self.dtype).to(self.device)
        return F.normalize(node_attr, p=2, dim=1)

    def init_edge_feat_adj(self, graph, use_attr=True):
        if graph.edge_attr is None or not use_attr:
            edge_attr = torch.ones(graph.edge_index.shape[1], 1, dtype=self.dtype).to(self.device)
        else:
            edge_attr = graph.edge_attr.to(self.dtype).to(self.device)
        edge_attr = F.normalize(edge_attr, p=2, dim=1)

        edge_attr_adj = torch.zeros(edge_attr.shape[1], graph.num_nodes, graph.num_nodes, dtype=self.dtype).to(self.device)
        edge_attr_adj[:, graph.edge_index[0], graph.edge_index[1]] = edge_attr.T
        return edge_attr_adj
