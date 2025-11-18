import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import degree
import time
import psutil
import os

from PlanetAlign.data import Dataset
from PlanetAlign.utils import get_anchor_pairs, get_anchor_embeddings, balance_samples
from PlanetAlign.algorithms.base_model import BaseModel
from PlanetAlign.metrics import hits_ks_scores, mrr_score

from .model import NetTransModel
from .data import ContextDataset
from .sampling import *


class NetTrans(BaseModel):
    r"""Embedding-based method NetTrans for pairwise network alignment via network transformation.
    NetTrans is proposed by the paper "`NetTrans: Neural Cross-Network Transformation. <https://dl.acm.org/doi/10.1145/3394486.3403141>`_"
    in KDD 2020.

    Parameters
    ----------
    hid_dim : int, optional
        Hidden dimension of the model. Default is 128.
    depth : int, optional
        Network depth of the model. Default is 2.
    pooling_ratio : float, optional
        Pooling ratio of the model. Default is 0.2.
    attr_coeff : float, optional
        Coefficient for the attribute loss. Default is 1.0.
    adj_coeff : float, optional
        Coefficient for the structural loss. Default is 1.0.
    rank_coeff : float, optional
        Coefficient for the ranking loss. Default is 1.0.
    margin : float, optional
        Margin for the ranking loss. Default is 1.0.
    neg_size : int, optional
        Number of negative samples per anchor link. Default is 20.
    batch_size : int, optional
        Batch size for training. Default is 300.
    lr : float, optional
        Learning rate for the optimizer. Default is 0.001.
    temperature : float, optional
        Initial temperature for the model. Default is 1.0.
    min_temperature : float, optional
        Minimum temperature for the model. Default is 0.1.
    anneal_rate : float, optional
        Anneal rate for the temperature. Default is 2e-5.
    dtype : torch.dtype, optional
        Data type of the tensors, choose from torch.float32 or torch.float64. Default is torch.float32.
    """

    def __init__(self,
                 hid_dim: int = 128,
                 depth: int = 2,
                 pooling_ratio: float = 0.2,
                 attr_coeff: float = 1.,
                 adj_coeff: float = 1.,
                 rank_coeff: float = 1.,
                 margin: float = 1.,
                 neg_size: int = 20,
                 batch_size: int = 300,
                 lr: float = 0.001,
                 temperature: float = 1.,
                 min_temperature: float = 0.1,
                 anneal_rate: float = 2e-5,
                 dtype: torch.dtype = torch.float32):
        super(NetTrans, self).__init__(dtype=dtype)

        self.hid_dim = hid_dim
        self.depth = depth
        self.pooling_ratio = pooling_ratio
        self.attr_coeff = attr_coeff
        self.adj_coeff = adj_coeff
        self.rank_coeff = rank_coeff
        self.margin = margin
        self.neg_size = neg_size
        self.batch_size = batch_size
        self.lr = lr
        self.temperature = temperature
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate

    def train(self,
              dataset: Dataset,
              gid1: int,
              gid2: int,
              use_attr: bool = True,
              total_epochs: int = 50,
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
            Flag for using attributes. Default is True.
        total_epochs : int, optional
            Maximum number of training epochs. Default is 50.
        save_log : bool, optional
            Flag for saving the training log. Default is True.
        verbose : bool, optional
            Whether to print the progress during training. Default is True.
        """

        self.check_inputs(dataset, (gid1, gid2), plain_method=False, use_attr=use_attr, pairwise=True, supervised=True)

        logger = self.init_training_logger(dataset, use_attr, additional_headers=['memory', 'infer_time'], save_log=save_log)
        process = psutil.Process(os.getpid())

        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        n1, n2 = graph1.num_nodes, graph2.num_nodes
        anchor_links = get_anchor_pairs(dataset.train_data, gid1, gid2)
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)

        # Embedding initialization
        anchor_embeddings1 = get_anchor_embeddings(graph1, anchor_links[:, 0]).to(self.dtype)
        anchor_embeddings2 = get_anchor_embeddings(graph2, anchor_links[:, 1]).to(self.dtype)

        node_attr1 = torch.cat([graph1.x.to(self.dtype), anchor_embeddings1], dim=1) if use_attr else anchor_embeddings1
        node_attr2 = torch.cat([graph2.x.to(self.dtype), anchor_embeddings2], dim=1) if use_attr else anchor_embeddings2
        node_attr1, node_attr2 = node_attr1.to(self.device), node_attr2.to(self.device)

        edge_weight1 = torch.ones(graph1.edge_index.shape[1], dtype=self.dtype).to(self.device)
        edge_weight2 = torch.ones(graph2.edge_index.shape[1], dtype=self.dtype).to(self.device)

        # Data preparation
        anchor_context_pairs1 = self._sample_anchor_neighbor_pairs(graph1, anchor_links[:, 0])
        anchor_context_pairs2 = self._sample_anchor_neighbor_pairs(graph2, anchor_links[:, 1])
        anchor_context_pairs1, anchor_context_pairs2 = balance_samples(anchor_context_pairs1, anchor_context_pairs2)
        neg_context_prob1, anchor_map1 = self._get_neg_context_prob(graph1, anchor_links[:, 0])
        neg_context_prob2, anchor_map2 = self._get_neg_context_prob(graph2, anchor_links[:, 1])

        # Model initialization
        anchor_context_dataset = ContextDataset(anchor_context_pairs1, anchor_context_pairs2)
        data_loader = DataLoader(dataset=anchor_context_dataset, batch_size=self.batch_size, shuffle=True)

        model = NetTransModel(in_dim=node_attr1.shape[1],
                              hid_dim=self.hid_dim,
                              out_dim=node_attr2.shape[1],
                              pooling_ratio=self.pooling_ratio,
                              depth=self.depth,
                              margin=self.margin).to(self.dtype).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Training
        edge_index1, edge_index2 = graph1.edge_index.to(self.device), graph2.edge_index.to(self.device)
        anchor_links = anchor_links.to(self.device)
        emb1, emb2 = None, None
        for epoch in range(total_epochs):
            t0 = time.time()
            model.train()
            if epoch % 10 == 1:
                self.temperature = max(self.temperature * np.exp(-self.anneal_rate * epoch), self.min_temperature)
            total_loss = 0
            x, y = None, None
            infer_time = 0
            for i, (context1, context2) in enumerate(data_loader):
                context1, context2 = context1.to(self.device), context2.to(self.device)
                optimizer.zero_grad()
                infer_t0 = time.time()
                x, _, y, _, _, _ = model(x=node_attr1,
                                         edge_index=edge_index1,
                                         edge_weight=edge_weight1,
                                         y=node_attr2,
                                         edge_index_y=edge_index2,
                                         edge_weight_y=edge_weight2,
                                         anchor_links=anchor_links.T,
                                         temperature=self.temperature)
                infer_time += time.time() - infer_t0

                # Negative sampling
                with torch.no_grad():
                    anchor_nodes1 = context1[:, 0].cpu().reshape(-1)
                    pos_context_nodes1 = context1[:, 1].reshape(-1)
                    anchor_nodes2 = context2[:, 0].cpu().reshape(-1)
                    pos_context_nodes2 = context2[:, 1].reshape(-1)

                    negs1, negs2 = uniform_negative_sampling(anchor_nodes1, anchor_nodes2, n1, n2, self.neg_size)
                    neg_context1 = negative_edge_sampling(neg_context_prob1, anchor_map1[anchor_nodes1], self.neg_size)
                    neg_context2 = negative_edge_sampling(neg_context_prob2, anchor_map2[anchor_nodes2], self.neg_size)

                negs1, negs2 = negs1.flatten().to(self.device), negs2.flatten().to(self.device)
                neg_context1, neg_context2 = neg_context1.flatten().to(self.device), neg_context2.flatten().to(self.device)

                neg_emb1, neg_emb2 = y[negs1], x[negs2]
                neg_context_emb1, neg_context_emb2 = x[neg_context1], y[neg_context2]
                anchor_emb1, anchor_emb2 = x[anchor_nodes1], y[anchor_nodes2]
                pos_emb1, pos_emb2 = x[pos_context_nodes1], y[pos_context_nodes2]

                adj_loss = model.adj_loss(anchor_emb1, pos_emb1, neg_context_emb1) + model.adj_loss(anchor_emb2, pos_emb2, neg_context_emb2)
                align_loss = model.align_loss(anchor_emb1, anchor_emb2, neg_emb1, neg_emb2)
                batch_loss = self.adj_coeff * adj_loss + self.rank_coeff * align_loss
                total_loss += batch_loss.item()

                batch_loss.backward()
                optimizer.step()
            t1 = time.time()

            model.eval()
            emb1 = F.normalize(x.detach(), p=2, dim=1)
            emb2 = F.normalize(y.detach(), p=2, dim=1)

            S = emb1 @ emb2.T
            hits = hits_ks_scores(S, test_pairs, mode='mean')
            mrr = mrr_score(S, test_pairs, mode='mean')
            mem_gb = process.memory_info().rss / 1024 ** 3
            logger.log(epoch=epoch+1,
                       loss=total_loss,
                       epoch_time=t1-t0,
                       mrr=mrr,
                       hits=hits,
                       memory=round(mem_gb, 4),
                       infer_time=round(infer_time, 4),
                       verbose=verbose)

        return emb1, emb2, logger

    @staticmethod
    def _sample_anchor_neighbor_pairs(graph, anchors):
        degrees = degree(graph.edge_index[0], num_nodes=graph.num_nodes)

        sampled_context_pairs = torch.empty((0, 2), dtype=torch.int64)
        for node in anchors:
            neighbors = graph.edge_index[1, graph.edge_index[0] == node]
            if len(neighbors) > 100:
                p = degrees[neighbors] / degrees[neighbors].sum()
                neighbors = neighbors[torch.multinomial(p, 100, replacement=True)]
            context = torch.vstack([torch.tensor([node] * len(neighbors), dtype=torch.int64), neighbors]).T
            sampled_context_pairs = torch.vstack([sampled_context_pairs, context])

        # Shuffle the context pairs
        sampled_context_pairs = sampled_context_pairs[torch.randperm(sampled_context_pairs.size(0))]
        return sampled_context_pairs

    def _get_neg_context_prob(self, graph, anchors):
        prob = torch.ones((len(anchors), graph.num_nodes), dtype=self.dtype)
        for i, anchor in enumerate(anchors):
            neighbors = graph.edge_index[1, graph.edge_index[0] == anchor]
            prob[i][neighbors] = 0

        anchor_node_map = -1 * torch.ones(graph.num_nodes, dtype=torch.int64)
        anchor_node_map[anchors] = torch.arange(len(anchors), dtype=torch.int64)

        return prob, anchor_node_map
