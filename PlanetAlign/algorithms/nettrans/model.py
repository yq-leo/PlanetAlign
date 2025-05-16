import torch.nn as nn
from torch_geometric.nn.conv import GCNConv, GraphConv, GATConv, APPNP
from torch_geometric.utils.repeat import repeat
from .functions import *


class NetTransModel(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 pooling_ratio,
                 depth,
                 margin,
                 act=nn.ReLU()):
        super(NetTransModel, self).__init__()
        self.in_channels = in_dim
        self.nhid = hid_dim
        self.out_channels = out_dim
        self.pool_ratios = repeat(pooling_ratio, depth)
        self.depth = depth
        self.act = act
        self.down_convs = torch.nn.ModuleList()
        self.down_convs.append(GCN(self.in_channels, self.nhid, k=10))
        self.pools = torch.nn.ModuleList()
        # create encoder
        for i in range(depth):
            self.pools.append(TransPool(hid_dim, hid_dim, self.pool_ratios[i], non_linearity=self.act))
            self.down_convs.append(GCNConv(hid_dim, hid_dim))

        # create intermediate MLP
        self.MLP1 = nn.Linear(self.nhid, self.nhid, bias=True)

        # create decoder
        self.unpools = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.unpools.append(TransUnPool(self.nhid, self.nhid))

        self.last_layer = LastLayer(self.nhid, self.nhid)

        self.MLP2 = torch.nn.Linear(self.nhid, self.out_channels)

        self.adj_loss_func = nn.BCEWithLogitsLoss()
        self.align_loss_func = nn.MarginRankingLoss(margin=margin)
        self.mse_loss = nn.MSELoss()
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for unpool in self.unpools:
            unpool.reset_parameters()
        self.MLP1.reset_parameters()
        self.MLP2.reset_parameters()
        self.last_layer.reset_parameters()

    def forward(self, x, edge_index, edge_weight, y, edge_index_y, edge_weight_y, anchor_links, temperature):
        n1, n2 = x.shape[0], y.shape[0]
        batch = edge_index.new_zeros(x.shape[0])
        x = self.down_convs[0](x, edge_index, edge_weight)
        out_x = x

        x = self.act(x)

        xs, edge_indices, edge_weights, assign_mats = [out_x], [edge_index], [edge_weight], []
        num_sups = []

        for i in range(1, self.depth + 1):
            # TransPool layers to do the coarsening
            x, edge_index, edge_weight, batch, super_nodes, assign_index, assign_weight = self.pools[i - 1](
                x, edge_index, temperature, edge_weight, batch)
            assign_mats.append((assign_index, assign_weight))   # P matrices are sparse, so we store indices and values here
            num_sups.append(len(super_nodes))
            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)
            xs.append(x)
            if i < self.depth:
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
        sup1 = xs[1] if self.depth > 0 else []

        # intermediate layer
        x = self.act(self.MLP1(x))

        # decoder
        for i in range(self.depth - 1):
            j = self.depth - 1 - i
            x, edge_index, edge_weight = self.unpools[i](x, xs[j], edge_indices[j], edge_weights[j],
                                                         assign_mats[j][0], assign_mats[j][1])
            x = self.act(x)

        sup2 = x
        anchor_weight = torch.ones(len(anchor_links[0]), dtype=torch.float, device=x.device)
        q_index, q_weight = None, None
        if self.depth > 0:
            P1 = torch.sparse_coo_tensor(assign_mats[0][0], assign_mats[0][1], size=[num_sups[0], n1], device=x.device)
            L = torch.sparse_coo_tensor(anchor_links, anchor_weight, size=[n1, n2], device=x.device).to(P1.dtype)
            Q = mm(P1, L).coalesce()
            q_index, q_weight = Q.indices(), Q.values()

        out_y = self.down_convs[0](y, edge_index_y, edge_weight_y)
        out_y += self.last_layer(sup2, xs[0], q_index, q_weight, anchor_links, anchor_weight, n2)

        recon_y = self.MLP2(out_y)

        return out_x, sup1, out_y, sup2, assign_mats, recon_y

    @staticmethod
    def score(emb1, emb2):
        score = -torch.sum(torch.abs(emb1 - emb2), dim=1).reshape((-1, 1))
        return score

    def adj_loss(self, anchor1_emb, context_pos1_emb, context_neg1_emb):
        num_instance1 = anchor1_emb.shape[0]
        num_instance2 = context_neg1_emb.shape[0]
        N_negs = num_instance2 // num_instance1
        dim = anchor1_emb.shape[1]
        device = anchor1_emb.device

        term1 = self.score(anchor1_emb, context_pos1_emb)
        term2 = self.score(anchor1_emb.repeat(1, N_negs).reshape(-1, dim), context_neg1_emb)

        terms1 = torch.cat([term1, term2], dim=0).reshape((-1,))
        labels1 = torch.cat([torch.ones(num_instance1, device=device), torch.zeros(num_instance2, device=device)])
        loss = self.adj_loss_func(terms1, labels1)
        return loss

    def align_loss(self, pos_emb1, pos_emb2, neg_emb1, neg_emb2):
        num_instance1 = pos_emb1.shape[0]
        num_instance2 = neg_emb1.shape[0]
        dim = pos_emb1.shape[1]
        device = pos_emb1.device
        N_negs = num_instance2 // num_instance1
        term1 = self.score(pos_emb1, pos_emb2)
        term2 = self.score(pos_emb2, pos_emb1)
        term3 = self.score(pos_emb1.repeat(1, N_negs).reshape(-1, dim), neg_emb1)
        term4 = self.score(pos_emb2.repeat(1, N_negs).reshape(-1, dim), neg_emb2)
        loss = self.align_loss_func(term1.repeat(1, N_negs).reshape(-1, 1), term3, torch.ones_like(term3)) + \
                self.align_loss_func(term2.repeat(1, N_negs).reshape(-1, 1), term4, torch.ones_like(term4))
        return loss

    def recon_attr_loss(self, recon_y, y):
        return self.mse_loss(y, recon_y)


class TransUnPool(torch.nn.Module):
    def __init__(self, in_channels, out_channels, non_linearity=torch.tanh):
        super(TransUnPool, self).__init__()
        self.in_channels = in_channels
        self.non_linearity = non_linearity
        self.out_channels = out_channels
        self.BiConv = GraphConv((in_channels, in_channels), out_channels, bias=False)
        self.UniConv = GCNConv(in_channels, out_channels, bias=False, improved=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.BiConv.reset_parameters()
        self.UniConv.reset_parameters()

    def forward(self, sup_x, y, edge_index, edge_weight, assign_index, assign_weight):
        num_nodes1 = y.shape[0]
        num_nodes2 = sup_x.shape[0]

        x = self.BiConv((sup_x, None), assign_index, assign_weight, size=(num_nodes2, num_nodes1))
        x += self.UniConv(y, edge_index, edge_weight)

        return x, edge_index, edge_weight


class LastLayer(torch.nn.Module):
    def __init__(self, in_channels1, out_channels, bias=False):
        super(LastLayer, self).__init__()
        self.in_channels1 = in_channels1    # bipartite graph emb_dim
        self.out_channels = out_channels
        self.BiConv1 = GraphConv((in_channels1, in_channels1), out_channels, bias=bias)
        self.BiConv2 = GraphConv((in_channels1, in_channels1), out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.BiConv1.reset_parameters()
        self.BiConv2.reset_parameters()

    def forward(self, sup_x, y, assign_index, assign_weight, anchor_links, anchor_weight, num_nodes):
        # assign_index is of bipartite graph, edge_index is of unipartite graph
        # sup_x is embedding of supernodes, x0 is the node attribute of target network
        # y is the node embedding of source network
        num_src_nodes = y.shape[0]
        num_nodes1 = sup_x.shape[0]
        z = self.BiConv2((y, None), anchor_links, anchor_weight, size=(num_src_nodes, num_nodes))
        if assign_index is not None and assign_weight is not None:
            z += self.BiConv1((sup_x, None), assign_index, assign_weight, size=(num_nodes1, num_nodes))

        return z


class TransPool(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ratio=0.85, non_linearity=torch.relu):
        super(TransPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.score_layer = GCNConv(in_channels, 1)  # scoring layer for self-attention pooling
        self.non_linearity = non_linearity
        self.supAggr_layer = GCNConv(in_channels, out_channels) # aggregation layer from nodes to supernodes
        self.gumbel_weight = torch.empty(in_channels, out_channels)
        torch.nn.init.xavier_uniform_(self.gumbel_weight)
        self.gumbel_weight = nn.Parameter(self.gumbel_weight)
        self.reset_parameters()

    def reset_parameters(self):
        self.score_layer.reset_parameters()
        self.supAggr_layer.reset_parameters()

    def forward(self, x, edge_index, temperature, edge_weight, batch, layer_agg='avg'):
        num_nodes = x.shape[0]
        nodes = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        # select super nodes
        score = self.score_layer(x, edge_index, edge_weight).reshape(-1)
        super_nodes = self.topk(score, self.ratio, batch)
        num_supernodes = len(super_nodes)
        batch = batch[super_nodes]

        # build bipartite graph for node-to-supernode aggregation
        edge_index1, edge_weight1 = filter_target(edge_index, edge_weight, super_nodes, num_nodes)
        if isinstance(self.supAggr_layer, GCNConv):
            temp_x = self.supAggr_layer(x, edge_index1, edge_weight1)
        elif isinstance(self.supAggr_layer, GATConv):
            temp_x = self.supAggr_layer(x, edge_index1)
        else:
            raise ValueError("Currently only GAT and GCN are supported for node-to-supernode aggregation.")

        sup_x = self.non_linearity(temp_x[super_nodes])
        # re-map the index of supernodes to [0, num_supernodes-1]
        edge_index1, _ = remap_sup(edge_index1, super_nodes, num_nodes)
        out_edge_index, out_edge_weight = filter_source(edge_index1, edge_weight1, super_nodes, num_nodes)
        out_edge_index, _ = remap_sup(out_edge_index, super_nodes, num_nodes, mode='source')

        # candidate selection
        edge_index, edge_weight = cadicate_selection(edge_index, edge_weight, edge_index1, edge_weight1, nodes, super_nodes)

        assign_weight = torch.sigmoid(torch.sum(temp_x[edge_index[0]] * sup_x[edge_index[1]], dim=1))
        assign_weight = assign_weight * edge_weight
        edge_index, assign_weight = degree_normalize_sparse_tensor(edge_index, assign_weight, shape=[num_nodes, num_supernodes])
        edge_index, assign_weight = sparse_gumbel_softmax(assign_weight, edge_index, temperature, shape=[num_nodes, num_supernodes])
        edge_index, assign_weight = remove_by_threshold(edge_index, assign_weight, 0.5 / sup_x.shape[0])

        assign_index = torch.stack([edge_index[1], edge_index[0]])  # transpose
        if layer_agg == 'skip':
            x = sup_x
        else:
            assign = torch.sparse_coo_tensor(assign_index, assign_weight, size=[num_supernodes, num_nodes],
                                             device=edge_index.device)
            x = mm(assign, temp_x)
            if layer_agg == 'max':
                x = torch.max(sup_x, x)
            elif layer_agg == 'avg':
                x = (sup_x + x) / 2

        # get coarsened edges
        mapped_supernodes = torch.arange(len(super_nodes), dtype=torch.long, device=edge_index.device)
        isolated_nodes = isolated_source_nodes(out_edge_index, mapped_supernodes)
        if len(isolated_nodes) > 0:
            out_edge_index, out_edge_weight = connect_isolated_nodes(x, out_edge_index, out_edge_weight, isolated_nodes)

        out_edge_index, out_edge_weight = remove_self_loops(out_edge_index, out_edge_weight)

        return x, out_edge_index, out_edge_weight, batch, super_nodes, assign_index, assign_weight

    @staticmethod
    def topk(x, ratio, batch):
        # num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        num_nodes = torch.zeros(batch.max().item() + 1, dtype=batch.dtype, device=batch.device).index_add_(0, batch, batch.new_ones(x.size(0)))
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
        k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)
        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

        return perm


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, k=10, alpha=0.5):
        super(GCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.alpha = alpha

        self.conv = APPNP(self.k, self.alpha, add_self_loops=False)
        self.weight = torch.empty((self.in_channels, self.out_channels))
        torch.nn.init.uniform_(self.weight,  a=-out_channels**0.5, b=out_channels**0.5)
        self.weight = nn.Parameter(self.weight)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        x = x @ self.weight
        x = self.conv(x, edge_index, edge_weight)

        return x
