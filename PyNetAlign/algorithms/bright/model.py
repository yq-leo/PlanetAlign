import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv


class BrightUNet(torch.nn.Module):
    r"""BRIGHT-U model for plain network alignment."""
    def __init__(self, in_dim, out_dim):
        super(BrightUNet, self).__init__()
        self.lin1 = torch.nn.Linear(in_dim, out_dim)

    def forward(self, rwr_emb1, rwr_emb2, *args, **kwargs):
        pos_emd1 = self.lin1(rwr_emb1)
        pos_emd2 = self.lin1(rwr_emb2)
        pos_emd1 = F.normalize(pos_emd1, p=2, dim=1)
        pos_emd2 = F.normalize(pos_emd2, p=2, dim=1)
        return pos_emd1, pos_emd2


class BrightANet(torch.nn.Module):
    r"""BRIGHT-A model for attributed network alignment."""
    def __init__(self, rwr_dim, in_dim, out_dim):
        super(BrightANet, self).__init__()
        self.lin = torch.nn.Linear(rwr_dim, out_dim)
        self.shared_gcn = SharedGCN(in_dim, out_dim)
        self.combine = torch.nn.Linear(2 * out_dim, out_dim)

    def forward(self, rwr_emb1, rwr_emb2, x1, x2, edge_index1, edge_index2):
        pos_emd1 = self.lin(rwr_emb1)
        pos_emd2 = self.lin(rwr_emb2)
        gcn_emd1 = self.shared_gcn(x1, edge_index1)
        gcn_emd2 = self.shared_gcn(x2, edge_index2)
        pos_emd1 = F.normalize(pos_emd1, p=2, dim=1)
        pos_emd2 = F.normalize(pos_emd2, p=2, dim=1)
        gcn_emd1 = F.normalize(gcn_emd1, p=2, dim=1)
        gcn_emd2 = F.normalize(gcn_emd2, p=2, dim=1)
        emd1 = torch.cat([pos_emd1, gcn_emd1], 1)
        emd2 = torch.cat([pos_emd2, gcn_emd2], 1)
        emd1 = self.combine(emd1)
        emd1 = F.normalize(emd1, p=2, dim=1)
        emd2 = self.combine(emd2)
        emd2 = F.normalize(emd2, p=2, dim=1)
        return emd1, emd2


class SharedGCN(torch.nn.Module):
    def __init__(self, feat_dim, hid_dim):
        super(SharedGCN, self).__init__()
        self.conv1 = GCNConv(feat_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class MarginalRankingLoss(torch.nn.Module):
    def __init__(self, k, margin, **kwargs):
        super(MarginalRankingLoss, self).__init__()
        self.k = k
        self.margin = margin

    def neg_sampling(self, out1, out2, anchor1, anchor2):
        anchor_embeddings_1 = out1[anchor1]
        anchor_embeddings_2 = out2[anchor2]

        distances_1 = self.pairwise_cos_dist(anchor_embeddings_1, out2)
        ranks_1 = torch.argsort(distances_1, dim=1)
        neg_samples_1 = ranks_1[:, :self.k]

        distances_2 = self.pairwise_cos_dist(anchor_embeddings_2, out1)
        ranks_2 = torch.argsort(distances_2, dim=1)
        neg_samples_2 = ranks_2[:, :self.k]

        return neg_samples_1, neg_samples_2

    def forward(self, out1, out2, anchor_links):
        anchor1, anchor2 = anchor_links[:, 0], anchor_links[:, 1]

        with torch.no_grad():
            neg_samples_1, neg_samples_2 = self.neg_sampling(out1, out2, anchor1, anchor2)

        anchor_embeddings_1 = out1[anchor1]
        anchor_embeddings_2 = out2[anchor2]
        neg_embeddings_1 = out2[neg_samples_1, :]
        neg_embeddings_2 = out1[neg_samples_2, :]

        A = self.rowwise_cos_dist(anchor_embeddings_1, anchor_embeddings_2)
        D = A + self.margin
        B1 = -self.rowwise_cos_dist(
            anchor_embeddings_1.unsqueeze(1).repeat(1, self.k, 1).view(-1, anchor_embeddings_1.shape[-1]),
            neg_embeddings_1.view(-1, neg_embeddings_1.shape[-1]))
        L1 = torch.sum(F.relu(D.unsqueeze(-1) + B1.view(-1, self.k)))
        B2 = -self.rowwise_cos_dist(
            anchor_embeddings_2.unsqueeze(1).repeat(1, self.k, 1).view(-1, anchor_embeddings_2.shape[-1]),
            neg_embeddings_2.view(-1, neg_embeddings_2.shape[-1]))
        L2 = torch.sum(F.relu(D.unsqueeze(-1) + B2.view(-1, self.k)))

        return (L1 + L2) / (anchor1.shape[0] * self.k)

    @staticmethod
    def rowwise_cos_dist(emb1, emb2):
        return 1 - torch.sum(emb1 * emb2, dim=1) / (torch.norm(emb1, p=2, dim=1) * torch.norm(emb2, p=2, dim=1))

    @staticmethod
    def pairwise_cos_dist(emb1, emb2):
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        return 1 - emb1 @ emb2.T
    