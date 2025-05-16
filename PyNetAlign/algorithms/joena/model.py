import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from .utils import sinkhorn_stable


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(input_dim + hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x1, x2):
        pos_emd1 = torch.cat([x1, self.act(self.lin1(x1))], dim=1)
        pos_emd2 = torch.cat([x2, self.act(self.lin1(x2))], dim=1)
        pos_emd1 = self.lin2(pos_emd1)
        pos_emd2 = self.lin2(pos_emd2)
        pos_emd1 = F.normalize(pos_emd1, p=2, dim=1)
        pos_emd2 = F.normalize(pos_emd2, p=2, dim=1)
        return pos_emd1, pos_emd2


class FusedGWLoss(torch.nn.Module):
    def __init__(self, graph1, graph2, gw_weight=20, gamma_p=1e-2, init_lambda=1, lambda_step=1e-2, in_iter=5, out_iter=10, dtype=torch.float64):
        super().__init__()
        self.gw_weight = gw_weight
        self.gamma_p = gamma_p
        self.in_iter = in_iter
        self.out_iter = out_iter
        self.dtype = dtype
        self.device = 'cpu'

        self.n1, self.n2 = graph1.num_nodes, graph2.num_nodes
        self.threshold_lambda = init_lambda / (self.n1 * self.n2)
        self.lambda_step = lambda_step
        self.adj1 = to_dense_adj(graph1.edge_index, max_num_nodes=graph1.num_nodes).squeeze().to(self.dtype)
        self.adj2 = to_dense_adj(graph2.edge_index, max_num_nodes=graph2.num_nodes).squeeze().to(self.dtype)

    def forward(self, out1, out2):
        inter_c = torch.exp(-(out1 @ out2.T))
        intra_c1 = torch.exp(-(out1 @ out1.T)) * self.adj1
        intra_c2 = torch.exp(-(out2 @ out2.T)) * self.adj2
        with torch.no_grad():
            s = sinkhorn_stable(inter_c, intra_c1, intra_c2,
                                gw_weight=self.gw_weight,
                                gamma_p=self.gamma_p,
                                threshold_lambda=self.threshold_lambda,
                                in_iter=self.in_iter,
                                out_iter=self.out_iter,
                                dtype=self.dtype,
                                device=self.device)
            self.threshold_lambda = self.lambda_step * self._update_lambda(inter_c, intra_c1, intra_c2, s) + (1 - self.lambda_step) * self.threshold_lambda

        s_hat = s - self.threshold_lambda

        # Wasserstein Loss
        w_loss = torch.sum(inter_c * s_hat)

        # Gromov-Wasserstein Loss
        a = torch.sum(s_hat, dim=1)
        b = torch.sum(s_hat, dim=0)
        gw_loss = torch.sum(
            (intra_c1 ** 2 @ a.view(-1, 1) @ torch.ones((1, self.n2)).to(self.dtype).to(self.device) +
             torch.ones((self.n1, 1)).to(self.dtype).to(self.device) @ b.view(1, -1) @ intra_c2 ** 2 -
             2 * intra_c1 @ s_hat @ intra_c2.T) * s_hat)

        loss = w_loss + self.gw_weight * gw_loss + 20
        return loss, s, self.threshold_lambda

    def _update_lambda(self, inter_c, intra_c1, intra_c2, s):
        k1 = torch.sum(inter_c)

        one_mat = torch.ones((self.n1, self.n2), dtype=self.dtype).to(self.device)
        mid = intra_c1 ** 2 @ one_mat * self.n2 + one_mat @ intra_c2 ** 2 * self.n1 - 2 * intra_c1 @ one_mat @ intra_c2.T
        k2 = torch.sum(mid * s)
        k3 = torch.sum(mid)

        return (k1 + 2 * self.gw_weight * k2) / (2 * self.gw_weight * k3)

    def to(self, device):
        assert device in ['cpu', 'cuda'] or isinstance(device, torch.device), 'Invalid device'
        self.device = device
        self.adj1, self.adj2 = self.adj1.to(device), self.adj2.to(device)
        return self
