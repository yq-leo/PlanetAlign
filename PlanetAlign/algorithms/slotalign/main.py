from typing import Optional, Union, Tuple, List
import time
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import psutil
import os

from PlanetAlign.data import Dataset
from PlanetAlign.utils import get_anchor_pairs
from PlanetAlign.metrics import hits_ks_scores, mrr_score
from PlanetAlign.algorithms.base_model import BaseModel

from .model import ParamFreeGraphConv
from .utils import euclidean_proj_simplex


class SLOTAlign(BaseModel):
    """OT-based method SLOTAlign for unsupervised pairwise attributed network alignment.
    SLOTAlign is proposed by the paper "`Robust Attributed Graph Alignment via Joint Structure Learning and Optimal Transport. <https://doi.org/10.1109/ICDE55515.2023.00129>`_"
    in ICDE 2023.

    Parameters
    ----------
    bases : int, optional
        The base number for the model. The default is 4.
    truncate : bool, optional
        Whether to truncate the node attributes to the first 100 attributes. The default is False.
    epsilon : float, optional
        The entropy regularization parameter for the Sinkhorn iteration. The default is 1e-2.
    step_size : float, optional
        The step size for the embedding update. The default is 1.
    dtype : torch.dtype, optional
        The data type for the model. The default is torch.float32.
    """
    def __init__(self,
                 bases: int = 4,
                 truncate: bool = False,
                 epsilon: float = 1e-2,
                 step_size: float = 1,
                 dtype: torch.dtype = torch.float32):
        super(SLOTAlign, self).__init__(dtype=dtype)
        assert bases >= 2, 'Number of bases must be greater than or equal to 2'

        self.bases = bases
        self.num_layers = bases - 2
        self.truncate = truncate
        self.epsilon = epsilon
        self.step_size = step_size

    def train(self,
              dataset: Dataset,
              gids: Union[Tuple[int, int], List[int]],
              use_attr: bool = True,
              total_epochs: int = 900,
              joint_epochs: int = 100,
              save_log: bool = True,
              verbose: bool = True):
        """
        Parameters
        ----------
        dataset : Dataset
            The dataset object containing the graphs and anchor links.
        gids : tuple[int, int] or list[int]
            The indices of the two graphs in the dataset to be aligned.
        use_attr : bool, optional
            Whether to use node attributes for alignment. The default is True.
        total_epochs : int, optional
            The total number of epochs for the gromov-wasserstein optimization. The default is 900.
        joint_epochs : int, optional
            The number of epochs for the joint optimization. The default is 100.
        save_log : bool, optional
            Whether to save the log of the training process. The default is True.
        verbose : bool, optional
            Whether to print the progress of the optimization. The default is True.
        """

        self.check_inputs(dataset, gids, plain_method=False, use_attr=use_attr, pairwise=True, supervised=False)
        gid1, gid2 = gids

        logger = self.init_training_logger(dataset, use_attr, additional_headers=['memory', 'infer_time'], save_log=save_log)
        process = psutil.Process(os.getpid())

        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        n1, n2 = graph1.num_nodes, graph2.num_nodes
        anchor_links = get_anchor_pairs(dataset.train_data, gid1, gid2)
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)

        # Computing node structural similarity via a shared parameter-free GCN
        if verbose:
            print('Computing node structural similarity...', end=' ')
        t0 = time.time()
        shared_param_free_gcn = ParamFreeGraphConv().to(self.device)
        intra_sim_tensor1 = self.get_intra_graph_similarity(graph1, shared_param_free_gcn, anchor_links[:, 0], use_attr)
        intra_sim_tensor2 = self.get_intra_graph_similarity(graph2, shared_param_free_gcn, anchor_links[:, 1], use_attr)
        if verbose:
            print(f'Done. Time taken: {time.time() - t0:.2f}s')

        # Alternating optimization
        alpha0 = torch.ones(self.num_layers + 2, dtype=self.dtype).to(self.device) / (self.num_layers + 2)
        beta0 = torch.ones(self.num_layers + 2, dtype=self.dtype).to(self.device) / (self.num_layers + 2)
        p_s = torch.ones(n1, 1, dtype=self.dtype).to(self.device) / n1
        p_t = torch.ones(n2, 1, dtype=self.dtype).to(self.device) / n2
        S = torch.ones(n1, n2, dtype=self.dtype).to(self.device) / (n1 * n2)
        print('Starting joint optimization...')
        t0 = time.time()
        infer_time = 0
        for epoch in range(joint_epochs):
            alpha = torch.clone(alpha0).requires_grad_(True)
            beta = torch.clone(beta0).requires_grad_(True)

            inf_t0 = time.time()
            A = torch.sum(intra_sim_tensor1 * alpha, dim=2)
            B = torch.sum(intra_sim_tensor2 * beta, dim=2)
            infer_time += time.time() - inf_t0
            objective = (A ** 2).mean() + (B ** 2).mean() - torch.trace(A @ S @ B @ S.T)
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch + 1}, Loss: {objective.item(): .8f}')

            # Compute gradients for alpha and update
            alpha_grad = torch.autograd.grad(outputs=objective, inputs=alpha, retain_graph=True)[0]
            with torch.no_grad():
                alpha = alpha - self.step_size * alpha_grad
            alpha0 = alpha.detach().cpu().numpy()
            alpha0 = torch.from_numpy(euclidean_proj_simplex(alpha0)).to(self.dtype).to(self.device)

            # Compute gradients for beta and update
            beta_grad = torch.autograd.grad(outputs=objective, inputs=beta)[0]
            with torch.no_grad():
                beta = beta - self.step_size * beta_grad
            beta0 = beta.detach().cpu().numpy()
            beta0 = torch.from_numpy(euclidean_proj_simplex(beta0)).to(self.dtype).to(self.device)

            # Update S
            S = self.gw_step_optimize(A, B, p_s, p_t, S0=S, inner_iter=50)

        infer_time /= joint_epochs
        if verbose:
            print(f'Joint optimization done. Time taken: {time.time() - t0:.2f}s')

        A = torch.sum(intra_sim_tensor1 * alpha0, dim=2)
        B = torch.sum(intra_sim_tensor2 * beta0, dim=2)
        print('Starting GW OT optimization...')
        t0 = time.time()
        for epoch in range(total_epochs):
            t0_epoch = time.time()
            S0 = torch.clone(S)
            S = self.gw_step_optimize(A, B, p_s, p_t, S0=S, inner_iter=20)
            t1_epoch = time.time()
            diff = torch.norm(S - S0)
            hits, mrr = hits_ks_scores(S, test_pairs, mode='mean'), mrr_score(S, test_pairs, mode='mean')
            mem_gb = process.memory_info().rss / 1024 ** 3
            if verbose and (epoch + 1) % 50 == 0:
                logger.log(epoch=epoch+1,
                           loss=diff.item(),
                           epoch_time=t1_epoch-t0_epoch,
                           hits=hits,
                           mrr=mrr,
                           memory=round(mem_gb, 4),
                           infer_time=round(infer_time + t1_epoch - t0_epoch, 4),
                           verbose=verbose)
        print(f'GW OT optimization done. Time taken: {time.time() - t0:.2f}s')

        return S, logger

    def get_intra_graph_similarity(self, graph, gcn, anchor_nodes, use_attr):
        node_attr = graph.x if use_attr else torch.ones(graph.num_nodes, 1)
        if anchor_nodes.shape[0] > 0:
            anchor_emb = torch.zeros(graph.num_nodes, anchor_nodes.shape[0], dtype=self.dtype)
            anchor_emb[anchor_nodes, torch.arange(anchor_nodes.shape[0])] = 1
            node_attr = torch.hstack([node_attr, anchor_emb])

        node_attr -= node_attr.mean(dim=0, keepdim=True)
        if self.truncate:
            node_attr = node_attr[:100]
        node_attr = node_attr.to(self.dtype)

        # Generate node embeddings by a gcn
        node_attr_list = [node_attr.to(self.device)]
        edge_index = graph.edge_index.to(self.device)
        for i in range(self.num_layers):
            node_attr_list.append(gcn(node_attr_list[-1], edge_index).detach())

        adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).squeeze().to(self.dtype)
        intra_sim_list = [adj.to(self.device)]
        for i in range(self.num_layers + 1):
            node_attr = F.normalize(node_attr_list[i], p=2, dim=1)
            intra_sim = node_attr @ node_attr.T
            intra_sim_list.append(intra_sim)
        intra_sim_tensor = torch.stack(intra_sim_list, dim=2)

        return intra_sim_tensor

    @torch.no_grad()
    def gw_step_optimize(self,
                         cost_s: torch.Tensor,
                         cost_t: torch.Tensor,
                         p_s: torch.Tensor,
                         p_t: torch.Tensor,
                         S0: Optional[torch.Tensor] = None,
                         inner_iter: int = 50,
                         error_bound: float = 1e-10):
        if S0 is None:
            S0 = p_s @ p_t.T

        a = torch.ones_like(p_s) / p_s.shape[0]
        b = torch.ones_like(p_t) / p_t.shape[0]
        cost = -2 * (cost_s @ S0 @ cost_t.T)
        kernel = torch.exp(-cost / self.epsilon) * S0
        for i in range(inner_iter):
            a_old = torch.clone(a)
            b = p_t / (kernel.T @ a)
            a = p_s / (kernel @ b)
            relative_error = torch.sum(torch.abs(a - a_old)) / torch.sum(torch.abs(a))
            if relative_error < error_bound:
                break

        S = (a @ b.T) * kernel
        return S
    