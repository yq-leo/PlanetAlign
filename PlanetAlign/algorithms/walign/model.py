import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv, GCNConv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GATNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=512, heads=1):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_dim, hid_dim, heads=heads)
        self.conv2 = GATConv(hid_dim * heads, out_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, feature, edge_index):
        x = F.dropout(feature, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GCNNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=512):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, feature, edge_index):
        x = F.dropout(feature, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class LGCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=512, K=8):
        super(LGCN, self).__init__()
        self.conv1 = CombUnweighted(K=K)
        self.linear = torch.nn.Linear(in_dim * (K + 1), out_dim)

    def forward(self, feature, edge_index):
        x = self.conv1(feature, edge_index)
        x = self.linear(x)
        return x


class CombUnweighted(MessagePassing):
    def __init__(self, K=1, **kwargs):
        super(CombUnweighted, self).__init__(aggr='add', **kwargs)
        self.K = K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)
        xs = [x]
        for k in range(self.K):
            xs.append(self.propagate(edge_index, x=xs[-1], norm=norm))
        return torch.cat(xs, dim=1)

    def message(self, x_j, norm=None):
        if norm is not None:
            return norm.view(-1, 1) * x_j
        return x_j

    def edge_update(self):
        pass

    def message_and_aggregate(self, edge_index):
        pass

    def __repr__(self):
        return '{}({}, {}, K={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.K)


class TransLayer(torch.nn.Module):
    def __init__(self, out_dim=512, transform=True):
        super(TransLayer, self).__init__()
        self.transform = transform
        if transform:
            self.trans = torch.nn.Parameter(torch.eye(out_dim))

    def forward(self, x):
        if self.transform:
            return x @ self.trans
        return x


class WDiscriminator(torch.nn.Module):
    def __init__(self, hidden_sizes=(512, 512)):
        super(WDiscriminator, self).__init__()
        self.hidden1 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.hidden2 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[1])
        self.output = torch.nn.Linear(hidden_sizes[1], 1)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.hidden2(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        return self.output(x)


class ReconDNN(torch.nn.Module):
    def __init__(self, out_dim, in_dim, hid_dim=512):
        super(ReconDNN, self).__init__()
        self.hidden = torch.nn.Linear(out_dim, hid_dim)
        self.output = torch.nn.Linear(hid_dim, in_dim)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        return self.output(x)


class FeatureReconstructLoss(torch.nn.Module):
    def __init__(self):
        super(FeatureReconstructLoss, self).__init__()

    @staticmethod
    def forward(embd, x, recon_model):
        recon_x = recon_model(embd)
        return torch.norm(recon_x - x, dim=1, p=2).mean()
