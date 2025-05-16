import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, degree
import time
import psutil
import os

from PlanetAlign.data import Dataset
from PlanetAlign.metrics import hits_ks_scores, mrr_score
from PlanetAlign.utils import get_anchor_pairs

from .base_model import BaseModel


class REGAL(BaseModel):
    r"""Embedding-based method REGAL for unsupervised pairwise attributed network alignment.
    REGAL is proposed by the paper: "`REGAL: Representation Learning-based Graph Alignment. <https://dl.acm.org/doi/10.1145/3269206.3271788>`_"
    in CIKM 2018.

    Parameters
    ----------
    k : int, optional
        Hyperparameter for tuning the number of landmarks. Default is 10.
    num_layers : int, optional
        Number of layers for the neighborhood when generating structural embeddings. Default is 2.
    alpha : float, optional
        Hyperparameter for the decay factor of the structural embedding. Default is 0.01.
    gammastruc : float, optional
        Weight for the structural similarity. Default is 1.
    gammaattr : float, optional
        Weight for the attribute similarity. Default is 1.
    buckets : int, optional
        Number of buckets for the structural embedding learning. Default is 2.
    dtype : torch.dtype, optional
        Data type of the tensors, choose from torch.float32 or torch.float64. Default is torch.float32.
    """

    def __init__(self,
                 k: int = 10,
                 num_layers: int = 2,
                 alpha: float = 0.01,
                 gammastruc: float = 1,
                 gammaattr: float = 1,
                 buckets: int = 2,
                 dtype: torch.dtype = torch.float32):
        super(REGAL, self).__init__(dtype=dtype)

        assert isinstance(k, int) and k > 0, 'k must be a positive integer'        
        assert isinstance(num_layers, int) and num_layers > 0, 'Number of layers must be a positive integer'
        assert 0 < alpha < 1, 'Alpha must be in the range (0, 1)'
        assert gammastruc >= 0, 'Gamma for structural similarity must be non-negative'
        assert gammaattr >= 0, 'Gamma for attribute similarity must be non-negative'
        assert isinstance(buckets, int) and buckets > 1, 'Buckets must be an integer greater than 1'

        self.k = k
        self.num_layers = num_layers
        self.alpha = alpha
        self.gammastruc = gammastruc
        self.gammaattr = gammaattr
        self.buckets = buckets
    
    def train(self,
              dataset: Dataset,
              gid1: int,
              gid2: int,
              use_attr: bool = True,
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
        save_log : bool, optional
            Whether to save the log of the training process. Default is True.
        verbose : bool, optional
            Whether to print the progress during training. Default is True.
        """
        self.check_inputs(dataset, (gid1, gid2), plain_method=False, use_attr=use_attr, pairwise=True, supervised=False)

        logger = self.init_training_logger(dataset, use_attr, additional_headers=['memory', 'infer_time'], save_log=save_log)
        process = psutil.Process(os.getpid())

        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        anchor_links = get_anchor_pairs(dataset.train_data, gid1, gid2)
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)

        t0 = time.time()
        emb1, emb2, loss = self.get_xnetmf_embedding(graph1, graph2, anchor_links, use_attr, verbose)
        t1 = time.time()

        # Evaluation
        S = emb1 @ emb2.T
        hits = hits_ks_scores(S, test_pairs, mode='mean')
        mrr = mrr_score(S, test_pairs, mode='mean')
        mem_gb = process.memory_info().rss / 1024 ** 3
        logger.log(epoch=1,
                   loss=loss,
                   epoch_time=t1-t0,
                   hits=hits,
                   mrr=mrr,
                   memory=round(mem_gb, 4),
                   infer_time=round(t1-t0, 4),
                   verbose=verbose)

        return emb1, emb2, logger

    def get_xnetmf_embedding(self, graph1, graph2, anchor_links, use_attr, verbose=True):
        degree1 = degree(graph1.edge_index[0], num_nodes=graph1.num_nodes)
        degree2 = degree(graph2.edge_index[0], num_nodes=graph2.num_nodes)
        max_degree = max(degree1.max().item(), degree2.max().item())

        if verbose:
            print('Generating structural embeddings...', end=' ')
        struct_emb1 = self.get_structural_embedding(graph1, max_degree)
        struct_emb2 = self.get_structural_embedding(graph2, max_degree)
        if verbose:
            print('Done')

        if verbose:
            print('Generating xNetMF embeddings...', end=' ')
        n1, n2 = graph1.num_nodes, graph2.num_nodes
        num_landmarks = min(int(self.k * np.log(n1 + n2) / np.log(2)), n1 + n2)
        sampled_landmarks = torch.randperm(n1 + n2)[:num_landmarks]

        # Merge embeddings of the two graphs for effective similarity computation
        struct_emb = torch.vstack([struct_emb1, struct_emb2])
        landmarks_struct_emb = struct_emb[sampled_landmarks]
        struct_dist = torch.norm(struct_emb[:, None, :] - landmarks_struct_emb[None, :, :], dim=2)

        num_anchors = anchor_links.shape[0]
        anchor_emb1 = torch.empty((n1, 0), dtype=self.dtype)
        anchor_emb2 = torch.empty((n2, 0), dtype=self.dtype)
        if num_anchors > 0:
            anchor_emb1 = torch.zeros((n1, num_anchors), dtype=self.dtype)
            anchor_emb2 = torch.zeros((n2, num_anchors), dtype=self.dtype)
            anchor_emb1[anchor_links[:, 0], torch.arange(num_anchors)] = 1
            anchor_emb2[anchor_links[:, 1], torch.arange(num_anchors)] = 1

        attr_emb1 = torch.empty((n1, 0), dtype=self.dtype)
        attr_emb2 = torch.empty((n2, 0), dtype=self.dtype)
        if use_attr and graph1.x is not None and graph2.x is not None:
            attr_emb1 = graph1.x
            attr_emb2 = graph2.x

        input_emb1 = torch.hstack([anchor_emb1, attr_emb1])
        input_emb2 = torch.hstack([anchor_emb2, attr_emb2])

        if input_emb1.shape[1] == 0 or input_emb2.shape[1] == 0:
            attribute_dist = 0
        else:
            input_emb = torch.vstack([input_emb1, input_emb2])
            landmarks_attr_emb = input_emb[sampled_landmarks]
            attribute_dist = (input_emb[:, None, :] != landmarks_attr_emb[None, :, :]).to(self.dtype).sum(dim=2)

        C = torch.exp(-(self.gammastruc * struct_dist + self.gammaattr * attribute_dist))

        W_pinv = torch.pinverse(C[sampled_landmarks])
        U, X, V = torch.svd(W_pinv)
        Wfac = U @ torch.diag(torch.sqrt(X))
        xnetmf_emb = C @ Wfac
        xnetmf_emb = F.normalize(xnetmf_emb, p=2, dim=1)
        if verbose:
            print('Done')

        W_reconstructed = U @ torch.diag(X) @ V.T
        svd_recon_loss = F.mse_loss(W_reconstructed, W_pinv)

        return xnetmf_emb[:n1], xnetmf_emb[n1:], svd_recon_loss.item()

    def get_structural_embedding(self, graph, max_degree):
        degrees = degree(graph.edge_index[0], num_nodes=graph.num_nodes)
        assert degrees.max().item() <= max_degree, 'Max degree is less than the maximum degree in the graph'

        neighborhood = self.get_k_hop_neighbors(graph, self.num_layers)
        num_features = int(np.log(max_degree) / np.log(self.buckets)) + 1

        # TODO: Optimize this part (1. vectorized manner may be possible, 2. get_k_hop_neighbors may not be necessary)
        structural_embedding = torch.zeros(graph.num_nodes, num_features)
        for n in range(graph.num_nodes):
            for layer in neighborhood[n].keys():
                neighbors = neighborhood[n][layer]
                if len(neighbors) > 0:
                    # get degree sequence
                    neighbors_degrees = degrees[neighbors]
                    filtered_neighbors_degrees = neighbors_degrees[neighbors_degrees > 0]
                    index_vec = (torch.log(filtered_neighbors_degrees) / np.log(self.buckets)).to(torch.int)
                    structural_embedding[n, :] += torch.bincount(index_vec, minlength=num_features) * (self.alpha ** layer)

        return structural_embedding

    @staticmethod
    def get_k_hop_neighbors(graph, k):
        adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).squeeze().to(torch.bool)

        neighborhoods = {node: {0: torch.tensor([node])} for node in range(graph.num_nodes)}
        for node in range(graph.num_nodes):
            last_neighbor_vec = torch.zeros(graph.num_nodes, dtype=torch.bool)
            last_neighbor_vec[node] = True
            for layer in range(1, k + 1):
                neighbor_vec = (adj[neighborhoods[node][layer - 1]].sum(dim=0) > 0) & ~last_neighbor_vec
                if neighbor_vec.sum() == 0:
                    break
                neighborhoods[node][layer] = torch.where(neighbor_vec)[0]
                last_neighbor_vec |= neighbor_vec

        return neighborhoods
    