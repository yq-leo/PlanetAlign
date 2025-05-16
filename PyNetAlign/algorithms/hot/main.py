from typing import List, Tuple, Union
import time
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import ot
import psutil
import os

from PyNetAlign.data import Dataset
from PyNetAlign.utils import get_batch_rwr_scores, pairwise_cosine_similarity
from PyNetAlign.algorithms.base_model import BaseModel

from .mot import *
from .metrics import multi_align_hits_ks_scores, multi_align_mrr_score


class HOT(BaseModel):
    """OT-based method HOT for multi-network alignment.
    HOT is proposed by the paper "`Hierarchical Multi-Marginal Optimal Transport for Network Alignment <https://doi.org/10.1609/aaai.v38i15.29605>`_"
    in AAAI 2024.

    Parameters
    ----------
    alpha : float, optional
        The trade-off parameter between the RWR similarity and the attribute similarity. Default is 0.5.
    lp : float, optional
        The regularization parameter for the proximal regularizer. Default is 0.1.
    """

    def __init__(self,
                 alpha: float = 0.5,
                 lp: float = 0.1,
                 dtype: torch.dtype = torch.float32):
        super(HOT, self).__init__(dtype=dtype)
        assert 0 <= alpha <= 1, 'Alpha must be in the range [0, 1]'
        assert lp > 0, 'lp must be positive'

        self.alpha = alpha
        self.lp = lp

    def train(self,
              dataset: Dataset,
              gids: Union[List[int], Tuple[int, ...]],
              use_attr: bool = True,
              in_iters: int = 5,
              out_iters: int = 50,
              save_log: bool = True,
              verbose: bool = True):
        """
        Parameters
        ----------
        dataset : Dataset
            The dataset object containing the graphs to be aligned and the anchor links.
        gids : list of int or tuple of int
            The graph IDs of the graphs to be aligned.
        use_attr : bool, optional
            Whether to use node attributes for alignment. Default is True.
        in_iters : int, optional
            Number of inner iterations for the proximal point optimization. Default is 5.
        out_iters : int, optional
            Number of outer iterations for the proximal point optimization. Default is 50.
        save_log : bool, optional
            Whether to save the training log. Default is True.
        verbose : bool, optional
            Whether to print the training log. Default is True.
        """

        self.check_inputs(dataset, gids, plain_method=False, use_attr=use_attr, pairwise=False, supervised=True)

        logger = self.init_training_logger(dataset, use_attr, additional_headers=['memory', 'infer_time'], save_log=save_log)
        process = psutil.Process(os.getpid())

        # Initialization
        graphs = [dataset.pyg_graphs[gid] for gid in gids]
        anchor_links = dataset.train_data[:, gids][torch.sum(dataset.train_data == -1, dim=1) != len(gids), :]
        test_links = dataset.test_data[:, gids][torch.sum(dataset.test_data == -1, dim=1) != len(gids), :]
        self._check_anchor_availability(anchor_links)
        self._check_anchor_availability(test_links)

        inf_t0 = time.time()
        rwr_emb_list = []
        for i in range(len(gids)):
            rwr_emb = get_batch_rwr_scores(graphs[i], anchor_links[:, i], device=self.device).to(self.dtype)
            rwr_emb_list.append(rwr_emb)

        # HOT
        intra_cost_list = []
        for i, graph in enumerate(graphs):
            intra_cost = self._get_intra_cost(graph, rwr_emb_list[i], use_attr)
            intra_cost_list.append(intra_cost)

        max_num_nodes = max([graph.num_nodes for graph in graphs])
        num_clusters = max_num_nodes // 50
        c, cr, cx, ca = self._get_fgw_clusters(graphs, rwr_emb_list, intra_cost_list, num_clusters, use_attr)

        sim_tensor_dict = {}
        cluster_nodes_dict = {}
        loss_record_list = []

        t0 = time.time()
        for i in range(num_clusters):
            r = []
            x = []
            marginal_dists = []
            cluster_nodes_dict[i] = []
            A = []
            for j in range(len(graphs)):
                r.append(cr[j][i])
                marginal_dists.append(torch.ones(len(r[-1])).to(self.device).to(self.dtype) / len(r[-1]))
                cluster_nodes_dict[i].append(c[j][i].cpu())
                if use_attr:
                    x.append(cx[j][i])
                A.append(ca[j][i])
            cross_cost_tensor = self._get_cross_cost(r, x, use_attr)
            sim_tensor, loss_record = self._multi_fgw(cross_cost_tensor, A, marginal_dists, in_iters, out_iters)
            sim_tensor_dict[i] = sim_tensor.cpu()
            loss_record_list.append(loss_record)
        t1 = time.time()
        infer_time = time.time() - inf_t0

        final_loss = np.sum([record[-1].item() for record in loss_record_list])
        pairwise_hits_ks, _ = multi_align_hits_ks_scores(sim_tensor_dict, cluster_nodes_dict, test_links, mode='mean')
        mrr = multi_align_mrr_score(sim_tensor_dict, cluster_nodes_dict, test_links, mode='mean')
        mem_gb = process.memory_info().rss / 1024 ** 3

        logger.log(epoch=1,
                   loss=final_loss.item(),
                   epoch_time=t1-t0,
                   hits=pairwise_hits_ks,
                   mrr=mrr,
                   memory=round(mem_gb, 4),
                   infer_time=round(infer_time, 4),
                   verbose=verbose)

        return sim_tensor_dict, logger

    def _multi_fgw(self, cross_cost_tensor, A, marginal_dists, in_iters, out_iters, tau=0.5, eps=1e-5):
        num_dists = len(marginal_dists)
        N = [len(marginal_dists[gid]) for gid in range(num_dists)]
        u = []
        S = (np.ones((num_dists, num_dists)) - 2 * np.eye(num_dists)).astype(np.int32).tolist()
        W = A.copy()
        for gid in range(num_dists):  # row-normalized adjacency
            W[gid] = A[gid] + torch.eye(len(A[gid])).to(self.dtype).to(self.device)
            W[gid] = A[gid] / torch.sum(A[gid], dim=1, keepdim=True)
        log_T = torch.zeros(N, dtype=self.dtype).to(self.device)

        log_m_dists = [torch.log(marginal_dists[k]) for k in range(num_dists)]
        for k in range(num_dists):
            log_T += log_m_dists[k].reshape(S[k])
            u.append(torch.log(torch.ones(N[k]) / N[k]).to(self.dtype).to(self.device))

        res = torch.inf
        i = 0
        t = torch.exp(torch.clamp(log_T, min=-torch.inf, max=70))
        W_list = [torch.inf]
        loss_record = []
        while i < out_iters and res > eps:
            P_i = compute_p_i(log_T)
            P_ij = compute_p_ij(log_T)
            L = compute_l(A, P_i, P_ij, dtype=self.dtype, device=self.device)

            Q = - ((1 - self.alpha) * cross_cost_tensor + self.alpha * L - self.lp * log_T) / self.lp
            W_list.append(torch.sum(((1 - self.alpha) * cross_cost_tensor + self.alpha * L) * t))
            for j in range(in_iters):
                P_i = compute_p_i(log_T)
                U = torch.zeros(N).to(self.dtype).to(self.device)
                for k in range(num_dists):
                    u[k] = u[k] + log_m_dists[k] - P_i[f'{k}']
                    U = U + torch.reshape(u[k], S[k])
                log_T = (1 - tau) * log_T + tau * (Q + U)

            t = torch.exp(torch.clamp(log_T, min=-torch.inf, max=70))
            res = torch.abs(W_list[-1] - W_list[-2])
            loss_record.append(res)
            i = i + 1
        return t, torch.tensor(loss_record)

    def _get_fgw_clusters(self, graphs, rwr_emb_list, intra_cost_list, num_clusters, use_attr):
        Ys = []
        norm_attr_list = []
        for i in range(len(graphs)):
            normalized_rwr = F.normalize(rwr_emb_list[i], p=2, dim=1)
            if use_attr:
                normalized_attr = F.normalize(graphs[i].x, p=2, dim=1).to(self.dtype).to(self.device)
                norm_attr_list.append(normalized_attr)
                Y = torch.cat((normalized_rwr, normalized_attr), dim=1)
            else:
                Y = normalized_rwr
            Ys.append(Y)

        marginal_dists = [torch.from_numpy(ot.utils.unif(graph.num_nodes)).to(self.dtype).to(self.device) for graph in graphs]
        _, _, T_list = ot.gromov.fgw_barycenters(num_clusters, Ys, intra_cost_list, marginal_dists, alpha=self.alpha, log=True)

        c = []
        cr = []
        cx = []
        ca = []

        for i in range(len(graphs)):
            l = torch.argmax(T_list['T'][i], dim=0)
            ind = {}
            temp_r = {}
            temp_x = {}
            temp_a = {}
            for j in range(num_clusters):
                ind[j] = torch.where(l == torch.tensor(j))[0]
                temp_r[j] = rwr_emb_list[i][ind[j]]
                temp_a[j] = intra_cost_list[i][ind[j]][:, ind[j]]
                if use_attr:
                    temp_x[j] = norm_attr_list[i][ind[j]]
            c.append(ind)
            cr.append(temp_r)
            if use_attr:
                cx.append(temp_x)
            ca.append(temp_a)

        return c, cr, cx, ca

    def _get_intra_cost(self, graph, rwr_emb, use_attr):
        adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).squeeze(0).to(self.dtype).to(self.device)
        rwr_sim = pairwise_cosine_similarity(rwr_emb, rwr_emb)
        attr_sim = pairwise_cosine_similarity(graph.x, graph.x) if use_attr else torch.zeros_like(rwr_sim)
        attr_sim = attr_sim.to(self.dtype).to(self.device)
        return torch.clamp((self.alpha * rwr_sim + (1 - self.alpha) * attr_sim) * (1 - adj), min=0)

    def _get_cross_cost(self, rwr_emb_list, x_list, use_attr, p=2):
        emb_list = [F.normalize(rwr_emb, p=p, dim=1) for rwr_emb in rwr_emb_list]
        if use_attr:
            emb_list = [torch.cat((emb, F.normalize(x, p=p, dim=1)), dim=1) for emb, x in zip(emb_list, x_list)]

        tensor_shape = tuple([rwr_emb.shape[0] for rwr_emb in rwr_emb_list])
        cross_cost_tensor = torch.zeros(tensor_shape, dtype=self.dtype).to(self.device)
        num_graphs = len(rwr_emb_list)
        for i in range(num_graphs):
            for j in range(i + 1, num_graphs):
                shape = [1] * len(tensor_shape)
                shape[i] = tensor_shape[i]
                shape[j] = tensor_shape[j]
                cross_cost_tensor += ot.utils.dist(emb_list[i], emb_list[j], p=p).reshape(shape)

        return cross_cost_tensor

    @staticmethod
    def _check_anchor_availability(anchor_links):
        assert torch.all(anchor_links != -1), \
            'Anchor links contain missing values. HOT requires anchor nodes to be present across all networks to be aligned.'
