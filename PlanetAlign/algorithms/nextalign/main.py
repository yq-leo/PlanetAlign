import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import psutil
import os

from PlanetAlign.data import Dataset
from PlanetAlign.utils import get_anchor_pairs, get_batch_rwr_scores, balance_samples, get_anchor_embeddings
from PlanetAlign.metrics import hits_ks_scores, mrr_score
from PlanetAlign.algorithms.base_model import BaseModel

from .utils import load_walks, extract_pairs, merge_graphs, ContextDataset, negative_sampling_exact
from .model import Model


class NeXtAlign(BaseModel):
    """Embedding-based method NeXtAlign for pairwise network alignment.
    NeXtAlign is proposed by the paper: "`Balancing Consistency and Disparity in Network Alignment. <https://dl.acm.org/doi/abs/10.1145/3447548.3467331>`_"
    in KDD 2021

    Parameters
    ----------
    p : float, optional
        Hyperparameter in node2vec. Default is 1.
    q : float, optional
        Hyperparameter in node2vec. Default is 1.
    num_walks : int, optional
        Number of random walks during context pair generation. Default is 10.
    walk_length : int, optional
        Length of each random walk. Default is 80.
    rwr_restart_prob : float, optional
        Restart probability for random walk with restart. Default is 0.15.
    out_dim : int, optional
        Dimension of the output embeddings. Default is 128.
    dist : str, optional
        Distance metric for the similarity matrix. Default is 'L1'.
    batch_size : int, optional
        Batch size for training. Default is 300.
    neg_size : int, optional
        Number of negative samples. Default is 20.
    coeff1 : float, optional
        Coefficient for the within-network link prediction loss. Default is 1.
    coeff2 : float, optional
        Coefficient for the anchor link prediction loss. Default is 1.
    lr : float, optional
        Learning rate for the optimizer. Default is 0.01.
    dtype : torch.dtype, optional
        Data type of the tensors, choose from torch.float32 or torch.float64. Default is torch.float32.
    """

    def __init__(self,
                 p: float = 1,
                 q: float = 1,
                 num_walks: int = 10,
                 walk_length: int = 80,
                 rwr_restart_prob: float = 0.15,
                 out_dim: int = 128,
                 dist: str = 'L1',
                 batch_size: int = 300,
                 neg_size: int = 20,
                 coeff1: float = 1, 
                 coeff2: float = 1,
                 lr: float = 0.01,
                 dtype: torch.dtype = torch.float32):
        super(NeXtAlign, self).__init__(dtype=dtype)
        assert dist in ['L1', 'inner'], 'Invalid distance metric'

        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.rwr_restart_prob = rwr_restart_prob
        self.out_dim = out_dim
        self.dist = dist
        self.batch_size = batch_size
        self.neg_size = neg_size
        self.coeff1 = coeff1
        self.coeff2 = coeff2
        self.lr = lr

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
            The dataset containing the graphs to be aligned and the training/test data.
        gid1 : int
            The index of the first graph in the dataset to be aligned.
        gid2 : int
            The index of the second graph in the dataset to be aligned.
        use_attr : bool, optional
            Whether to use node attributes in the model. Default is True.
        total_epochs : int, optional
            Total number of epochs for training. Default is 50.
        save_log : bool, optional
            Whether to save the training log. Default is True.
        verbose : bool, optional
            Whether to print the training progress. Default is True.
        """

        self.check_inputs(dataset, (gid1, gid2), plain_method=False, use_attr=use_attr, pairwise=True, supervised=True)

        if dataset.pyg_graphs[gid1].num_nodes > dataset.pyg_graphs[gid2].num_nodes:
            gid1, gid2 = gid2, gid1

        logger = self.init_training_logger(dataset, use_attr, additional_headers=['memory', 'infer_time'], save_log=save_log)
        process = psutil.Process(os.getpid())

        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        anchor_links = get_anchor_pairs(dataset.train_data, gid1, gid2)
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)

        t0 = time.time()
        if verbose:
            print('Sampling positive context pairs...')
        context_pairs1, context_pairs2 = self._sample_positive_context_pairs(graph1, graph2, anchor_links, verbose=verbose)
        if verbose:
            print('Done, Time Spent: %.2f seconds' % (time.time() - t0))

        t0 = time.time()
        if verbose:
            print('Generating initial positional embeddings...', end=' ')
        input_emb1, input_emb2 = self._generate_initial_embeddings(graph1, graph2, anchor_links, use_attr)
        if verbose:
            print('Done, Time Spent: %.2f seconds' % (time.time() - t0))
        rwr_time = time.time() - t0

        t0 = time.time()
        if verbose:
            print('Merging graphs...', end=' ')
        num_nodes_merged = graph1.num_nodes + graph2.num_nodes - anchor_links.shape[0]
        edge_index, edge_types, x, node_mapping1, node_mapping2 = self.merge_two_graphs(graph1, graph2,
                                                                                        input_emb1, input_emb2,
                                                                                        anchor_links)

        if verbose:
            print('Done, Time Spent: %.2f seconds' % (time.time() - t0))

        onehot_emb = torch.arange(num_nodes_merged, dtype=torch.int64)
        x = (onehot_emb, x[0], x[1]) if use_attr else (onehot_emb, x)

        # Context dataset
        node2vec_context_dataset = ContextDataset(context_pairs1, context_pairs2)
        data_loader = DataLoader(dataset=node2vec_context_dataset, batch_size=self.batch_size, shuffle=True)

        # Model
        model = Model(num_nodes=num_nodes_merged,
                      out_features=self.out_dim,
                      anchor_nodes=anchor_links[:, 0],
                      distance=self.dist,
                      num_anchors=anchor_links.shape[0],
                      num_attrs=x[2].shape[1] if use_attr else 0)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Move to device
        model = model.to(self.dtype).to(self.device)
        x = tuple([sub.to(self.dtype).to(self.device) for sub in x])
        edge_index = edge_index.to(self.device)
        edge_types = edge_types.to(self.device)
        node_mapping1 = node_mapping1.to(self.device)
        node_mapping2 = node_mapping2.to(self.device)

        # Train
        if verbose:
            print('Training...')
        for epoch in range(total_epochs):
            t0 = time.time()
            model.train()
            epoch_loss = 0
            out_x = None
            infer_time = rwr_time
            for i, data in enumerate(data_loader):
                nodes1, nodes2 = data
                nodes1 = nodes1.to(self.device)
                nodes2 = nodes2.to(self.device)
                anchor_nodes1 = nodes1[:, 0].reshape((-1,))
                pos_context_nodes1 = nodes1[:, 1].reshape((-1,))
                anchor_nodes2 = nodes2[:, 0].reshape((-1,))
                pos_context_nodes2 = nodes2[:, 1].reshape((-1,))

                # forward pass
                optimizer.zero_grad()
                inf_t0 = time.time()
                out_x = model(edge_index.T, x, edge_types)
                infer_time += time.time() - inf_t0

                context_pos1_emb = out_x[node_mapping1[pos_context_nodes1]]
                context_pos2_emb = out_x[node_mapping2[pos_context_nodes2]]

                pn_examples1, _ = negative_sampling_exact(out_x, self.neg_size, anchor_nodes1, node_mapping1,
                                                          'p_n', 'g1')
                pn_examples2, _ = negative_sampling_exact(out_x, self.neg_size, anchor_nodes2, node_mapping2,
                                                          'p_n', 'g2')
                pnc_examples1, _ = negative_sampling_exact(out_x, self.neg_size, anchor_nodes1, node_mapping1,
                                                           'p_nc', 'g1', node_mapping2=node_mapping2)
                pnc_examples2, _ = negative_sampling_exact(out_x, self.neg_size, anchor_nodes2, node_mapping2,
                                                           'p_nc', 'g2', node_mapping2=node_mapping1)

                # get node embeddings
                pn_examples1 = torch.from_numpy(pn_examples1).reshape((-1,)).to(self.device)
                pn_examples2 = torch.from_numpy(pn_examples2).reshape((-1,)).to(self.device)
                pnc_examples1 = torch.from_numpy(pnc_examples1).reshape((-1,)).to(self.device)
                pnc_examples2 = torch.from_numpy(pnc_examples2).reshape((-1,)).to(self.device)

                anchor1_emb = out_x[node_mapping1[anchor_nodes1]]
                anchor2_emb = out_x[node_mapping2[anchor_nodes2]]
                context_neg1_emb = out_x[node_mapping1[pn_examples1]]
                context_neg2_emb = out_x[node_mapping2[pn_examples2]]
                anchor_neg1_emb = out_x[node_mapping2[pnc_examples1]]
                anchor_neg2_emb = out_x[node_mapping1[pnc_examples2]]

                input_embs = (anchor1_emb, anchor2_emb, context_pos1_emb, context_pos2_emb, context_neg1_emb,
                              context_neg2_emb, anchor_neg1_emb, anchor_neg2_emb)

                # compute loss
                loss1, loss2 = model.loss(input_embs)
                batch_loss = self.coeff1 * loss1 + self.coeff2 * loss2

                # if verbose:
                #     print(f'Epoch: {epoch + 1}/{total_epochs}, Iteration: {i + 1}/{len(data_loader)}, '
                #           f'Batch loss: {batch_loss.item():.4f}, Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}')

                # backward pass
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                optimizer.step()
            t1 = time.time()

            # Evaluation
            model.eval()
            with torch.no_grad():
                out_x = F.normalize(out_x, p=2, dim=1)
                emb1 = out_x[node_mapping1]
                emb2 = out_x[node_mapping2]

                S = self.get_cross_alignment_mat(emb1, emb2, model.score_lin.weight[0].detach())
                hits = hits_ks_scores(S, test_pairs, mode='mean')
                mrr = mrr_score(S, test_pairs, mode='mean')
                mem_gb = process.memory_info().rss / 1024 ** 3
                logger.log(epoch=epoch+1,
                           loss=epoch_loss,
                           epoch_time=t1-t0,
                           hits=hits,
                           mrr=mrr,
                           memory=round(mem_gb, 4),
                           infer_time=round(infer_time, 4),
                           verbose=verbose)

        return emb1, emb2, logger

    def _sample_positive_context_pairs(self, graph1, graph2, anchor_links, verbose=True):
        walks1 = load_walks(graph1, self.p, self.q, self.num_walks, self.walk_length, verbose=verbose)
        walks2 = load_walks(graph2, self.p, self.q, self.num_walks, self.walk_length, verbose=verbose)
        context_pairs1 = extract_pairs(walks1, anchor_links[:, 0])
        context_pairs2 = extract_pairs(walks2, anchor_links[:, 1])
        context_pairs1, context_pairs2 = balance_samples(context_pairs1, context_pairs2)
        return context_pairs1, context_pairs2

    def _generate_initial_embeddings(self, graph1, graph2, anchor_links, use_attr):
        rwr_emb1 = get_batch_rwr_scores(graph1, anchor_links[:, 0], self.rwr_restart_prob, device=self.device).cpu().to(self.dtype)
        rwr_emb2 = get_batch_rwr_scores(graph2, anchor_links[:, 1], self.rwr_restart_prob, device=self.device).cpu().to(self.dtype)
        anchor_emb1 = get_anchor_embeddings(graph1, anchor_links[:, 0])
        anchor_emb2 = get_anchor_embeddings(graph2, anchor_links[:, 1])
        rwr_emb1[anchor_links[:, 0], :] = 0
        rwr_emb2[anchor_links[:, 1], :] = 0
        pos_emb1 = anchor_emb1 + rwr_emb1
        pos_emb2 = anchor_emb2 + rwr_emb2
        if use_attr:
            input_emb1 = (pos_emb1, graph1.x)
            input_emb2 = (pos_emb2, graph2.x)
        else:
            input_emb1 = pos_emb1
            input_emb2 = pos_emb2
        return input_emb1, input_emb2

    @staticmethod
    def merge_two_graphs(graph1, graph2, input_emb1, input_emb2, anchor_links):
        node_mapping1 = torch.arange(graph1.num_nodes, dtype=torch.int64)
        edge_index, edge_types, x, node_mapping2 = merge_graphs(graph1, graph2, input_emb1, input_emb2, anchor_links)
        return edge_index, edge_types, x, node_mapping1, node_mapping2

    @torch.no_grad()
    def get_cross_alignment_mat(self, emb1, emb2, weights):
        dim = emb1.shape[1]
        emb1_1 = emb1[:, :dim // 2]
        emb1_2 = emb1[:, dim // 2: dim]
        emb2_1 = emb2[:, :dim // 2]
        emb2_2 = emb2[:, dim // 2: dim]
        if self.dist == 'inner':
            S = weights[0] * emb1_1.dot(emb2_1.T) + weights[1] * emb1_1.dot(emb2_2.T) + weights[2] * emb1_2.dot(emb2_1.T) + \
                weights[3] * emb1_2.dot(emb2_2.T)
        else:
            S = weights[0] * torch.cdist(emb1_1, emb2_1, p=1) + \
                weights[1] * torch.cdist(emb1_1, emb2_2, p=1) + \
                weights[2] * torch.cdist(emb1_2, emb2_1, p=1) + \
                weights[3] * torch.cdist(emb1_2, emb2_2, p=1)
            S = -S
        return torch.sigmoid(S)
