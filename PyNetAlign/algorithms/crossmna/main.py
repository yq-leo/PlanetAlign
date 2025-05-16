from typing import List, Tuple, Union
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import degree
import time
import psutil
import os

from PyNetAlign.data import Dataset
from PyNetAlign.algorithms.base_model import BaseModel
from PyNetAlign.utils import merge_pyg_graphs_on_anchors, get_pairwise_anchor_pairs
from PyNetAlign.metrics import hits_ks_scores, mrr_score

from .model import MultiNetworkEmb
from .sampler import AliasSampler


class CrossMNA(BaseModel):
    """Embedding-based method CrossMNA for plain multi-network alignment.
    CrossMNA is proposed by the paper: "`Cross-Network Embedding for Multi-Network Alignment. <https://doi.org/10.1145/3308558.3313499>`_"
    in WWW 2019.

    Parameters
    ----------
    batch_size : int, optional
        Batch size for training. Default is 4096.
    neg_samples : int, optional
        Number of negative samples per positive sample. Default is 1.
    node_emb_dims : int, optional
        Dimensions of output node embeddings. Default is 200.
    graph_emb_dims : int, optional
        Dimensions of output graph embeddings. Default is 100.
    lr : float, optional
        Learning rate for the optimizer. Default is 0.02.
    dtype : torch.dtype, optional
        Data type of the tensors, choose from torch.float32 or torch.float64. Default is torch.float32.
    """

    def __init__(self,
                 batch_size: int = 512 * 8,
                 neg_samples: int = 1,
                 node_emb_dims: int = 200,
                 graph_emb_dims: int = 100,
                 lr: float = 0.02,
                 dtype: torch.dtype = torch.float32):
        super(CrossMNA, self).__init__(dtype=dtype)
        assert isinstance(batch_size, int), 'Batch size must be an integer'
        assert isinstance(neg_samples, int), 'Number of negative samples must be an integer'
        assert isinstance(node_emb_dims, int), 'Node embedding dimensions must be an integer'
        assert isinstance(graph_emb_dims, int), 'Graph embedding dimensions must be an integer'
        assert lr > 0, 'Learning rate must be positive'

        self.batch_size = batch_size
        self.neg_samples = neg_samples
        self.node_emb_dims = node_emb_dims
        self.graph_emb_dims = graph_emb_dims
        self.lr = lr

    def train(self,
              dataset: Dataset,
              gids: Union[List[int], Tuple[int, ...]],
              use_attr: bool = False,
              total_epochs: int = 400,
              save_log: bool = True,
              verbose: bool = True):
        """
        Parameters
        ----------
        dataset : Dataset
            The dataset containing the graphs to be aligned and the training/test data.
        gids : list or tuple
            The indices of the graphs in the dataset to be aligned.
        use_attr : bool, optional
            Whether to use node attributes for alignment. Default is True.
        total_epochs : int, optional
            Total number of training epochs. Default is 400.
        save_log : bool, optional
            Whether to save the training log. Default is True.
        verbose : bool, optional
            Whether to print training progress. Default is True.
        """

        self.check_inputs(dataset, gids, plain_method=True, use_attr=use_attr, pairwise=False, supervised=True)

        logger = self.init_training_logger(dataset, use_attr, additional_headers=['memory', 'infer_time'], save_log=save_log)
        process = psutil.Process(os.getpid())

        graphs = [dataset.pyg_graphs[gid] for gid in gids]
        anchor_links = dataset.train_data[:, gids]
        test_pairs_dict = get_pairwise_anchor_pairs(dataset.test_data[:, gids])

        # Initialization
        alias_samplers_list = []
        for gid in gids:
            graph = dataset.pyg_graphs[gid]
            alias_sampler = self._init_sampler(graph)
            alias_samplers_list.append(alias_sampler)

        _, _, id2node, node2id = merge_pyg_graphs_on_anchors(graphs, anchor_links)

        # Model initialization
        model = MultiNetworkEmb(num_of_nodes=len(id2node),
                                num_layer=len(graphs),
                                batch_size=self.batch_size,
                                K=self.neg_samples,
                                node_emb_dims=self.node_emb_dims,
                                layer_emb_dims=self.graph_emb_dims).to(self.dtype).to(self.device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=self.lr, alpha=0.99, eps=1.0, centered=True, momentum=0.0)

        # Training
        out_embs_dict = {}
        infer_time = 0
        for epoch in range(total_epochs):
            t0 = time.time()
            train_samples = self._generate_samples(graphs, alias_samplers_list)

            total_loss = 0
            for u_i, u_j, label, gid_vec in train_samples:
                u_i = u_i.cpu().numpy()
                u_j = u_j.cpu().numpy()
                gid_vec = gid_vec.cpu().numpy()
                label = label.to(self.dtype).to(self.device)

                mapped_u_i = np.array([node2id[(gid, u)] for gid, u in zip(gid_vec, u_i)])
                mapped_u_j = np.array([node2id[(gid, u)] for gid, u in zip(gid_vec, u_j)])

                optimizer.zero_grad()
                loss = model(mapped_u_i, mapped_u_j, gid_vec, label)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            t1 = time.time()
            infer_time += t1 - t0

            with torch.no_grad():
                mem_gb = process.memory_info().rss / 1024 ** 3
                out_embs_dict = {}
                embeddings = F.normalize(model.embedding, p=2, dim=1)
                for gid, graph in enumerate(graphs):
                    embs = embeddings[[node2id[(gid, u)] for u in range(graph.num_nodes)]]
                    out_embs_dict[gids[gid]] = embs

                for id1 in range(len(gids)):
                    for id2 in range(id1 + 1, len(gids)):
                        if verbose:
                            print(f'Graph {gids[id1]} vs Graph {gids[id2]}')
                        test_pairs = test_pairs_dict[(id1, id2)]
                        emb1 = out_embs_dict[gids[id1]]
                        emb2 = out_embs_dict[gids[id2]]
                        S = emb1 @ emb2.T
                        hits = hits_ks_scores(S, test_pairs, mode='mean')
                        mrr = mrr_score(S, test_pairs, mode='mean')
                        logger.log(epoch=epoch+1,
                                   loss=total_loss,
                                   epoch_time=t1-t0,
                                   mrr=mrr,
                                   hits=hits,
                                   memory=round(mem_gb, 4),
                                   infer_time=round(infer_time, 4),
                                   verbose=verbose)
        
        return out_embs_dict, logger
                        
    def _generate_samples(self, graphs, alias_samplers_list):
        all_edges = torch.empty((2, 0), dtype=torch.int64)
        all_gid_vec = torch.empty((0,), dtype=torch.int)
        all_neg_samples = torch.empty((self.neg_samples, 0), dtype=torch.int64)
        for gid, graph in enumerate(graphs):
            all_gid_vec = torch.cat([all_gid_vec, torch.tensor([gid] * graph.num_edges, dtype=torch.int)])
            all_edges = torch.hstack([all_edges, graph.edge_index])
            neg_samples = alias_samplers_list[gid].sample(num_samples=self.neg_samples * graph.num_edges).reshape(self.neg_samples, graph.num_edges)
            all_neg_samples = torch.hstack([all_neg_samples, torch.from_numpy(neg_samples)])

        # Shuffle sampled node pairs
        num_all_edges = all_edges.shape[1]
        perm = torch.randperm(num_all_edges)
        all_edges = all_edges[:, perm]
        all_gid_vec = all_gid_vec[perm]
        all_neg_samples = all_neg_samples[:, perm]

        ui_samples = torch.repeat_interleave(all_edges[0, :], repeats=self.neg_samples+1, dim=0)
        uj_samples = torch.vstack([all_edges[1, :], all_neg_samples]).T.flatten()
        gid_samples = torch.repeat_interleave(all_gid_vec, repeats=self.neg_samples+1)
        label_samples = torch.vstack([torch.ones(num_all_edges), -torch.ones(self.neg_samples, num_all_edges)]).T.flatten()

        assert ui_samples.shape == uj_samples.shape == gid_samples.shape == label_samples.shape, 'Shape mismatch'

        # Divide sampled node pairs into batches
        batched_samples = []
        num_batches = ui_samples.shape[0] // self.batch_size
        for i in range(num_batches):
            left = i * self.batch_size
            right = (i + 1) * self.batch_size
            u_i = ui_samples[left:right]
            u_j = uj_samples[left:right]
            label = label_samples[left:right]
            gid_vec = gid_samples[left:right]
            batched_samples.append((u_i, u_j, label, gid_vec))

        return batched_samples

    @staticmethod
    def _init_sampler(graph):
        node_positive_distribution = degree(graph.edge_index[0], num_nodes=graph.num_nodes) ** 0.75
        node_positive_distribution /= node_positive_distribution.sum()
        return AliasSampler(prob=node_positive_distribution)
