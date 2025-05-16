import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, num_nodes, out_features, anchor_nodes, distance='inner', num_anchors=None, num_attrs=0):
        """
        Model architecture.
        @param num_nodes: number of nodes in the merged graph.
        @param out_features: output feature dimension.
        @param anchor_nodes: anchor nodes in the merged graph.
        @param distance: distance function for scoring.
        @param num_anchors: number of anchor nodes.
        @param num_attrs: number of input node attributes.
        """
        super(Model, self).__init__()
        self.num_nodes = num_nodes
        self.out_features = out_features
        self.distance = distance
        self.num_attrs = num_attrs
        if num_attrs > 0:
            self.conv_attr = RelGCN(num_attrs, out_features, 2)
            self.combine = nn.Linear(num_anchors + out_features, out_features)
        else:
            self.combine = nn.Linear(num_anchors, out_features)

        self.conv_anchor = RelGCN(num_anchors, num_anchors, 2, bias=True, param=False)
        self.conv_one_hot = RelGCN(num_nodes, out_features, 2, bias=True)

        self.att1 = nn.Linear(out_features, 1, bias=False)
        self.att2 = nn.Linear(out_features, 1)
        self.score_lin = nn.Linear(4, 1)

        self.anchor_nodes = anchor_nodes

        self.loss_func1 = nn.BCEWithLogitsLoss()
        self.loss_func2 = nn.BCEWithLogitsLoss()

    def forward(self, edges, x, etype):
        """
        Forward pass of the whole model.

        @param edges: edge list of merged graph.
        @param x: input node attributes of merged graph. Either a tuple (one-hot encoding, pre-positioning)
                  for plain graph or (one-hot encoding, pre-positioning, node attributes) for attributed graph.
        @param etype: edge types of input merged graph.
        @return:
            out_x: output node embeddings.
        """
        x1, x2 = x[0], x[1]

        # out_x1 = self.conv_one_hot(g, x1, etype)
        # edges = torch.vstack(g.edges())
        out_x1 = self.conv_one_hot(edges, x1, etype)
        out_x1 = nn.functional.normalize(out_x1, p=1, dim=-1)

        anchor_emb = torch.zeros_like(x2)
        anchor_emb[self.anchor_nodes, torch.arange(len(self.anchor_nodes))] += 1
        # out_x2 = self.conv_anchor(g, anchor_emb, etype)
        out_x2 = self.conv_anchor(edges, anchor_emb, etype)
        out_x2 = nn.functional.normalize(out_x2, p=1, dim=-1)

        anchor_emb = out_x1[self.anchor_nodes]
        att1_score = self.att1(out_x1)
        att2_score = self.att2(anchor_emb)
        att1_score = att1_score.repeat(1, self.anchor_nodes.shape[0])
        att2_score = att2_score.reshape(1, -1).repeat(att1_score.shape[0], 1)
        att_score = att1_score + att2_score
        att_score = torch.softmax(att_score, dim=1)

        out_x = torch.multiply(out_x2, att_score)
        out_x = out_x + x2  # skip connections to encode pre-positioning.

        if self.num_attrs > 0:
            # out_x3 = self.conv_attr(g, x[2], etype)
            out_x3 = self.conv_attr(edges, x[2], etype)
            out_x3 = nn.functional.normalize(out_x3, p=1, dim=-1)
            out_x = torch.cat([out_x, out_x3], dim=1)

        out_x = self.combine(out_x)
        out_x = nn.functional.normalize(out_x, p=1, dim=-1)

        return out_x

    def score(self, emb1, emb2, graph_name):
        """
        Scoring function.

        @param emb1: node embeddings of graph G1.
        @param emb2: node embeddings of graph G2.
        @param graph_name: indicate whether G1 or G2.
        @return:
            predict_scores: alignment scores.
        """
        dim = emb1.shape[1]
        emb1_1, emb1_2 = emb1[:, 0: dim // 2], emb1[:, dim // 2: dim]
        emb2_1, emb2_2 = emb2[:, 0: dim // 2], emb2[:, dim // 2: dim]
        if self.distance == 'inner':
            score1 = torch.sum(torch.multiply(emb1_1, emb2_1), dim=1).reshape((-1, 1))
            score2 = torch.sum(torch.multiply(emb1_1, emb2_2), dim=1).reshape((-1, 1))
            score3 = torch.sum(torch.multiply(emb1_2, emb2_1), dim=1).reshape((-1, 1))
            score4 = torch.sum(torch.multiply(emb1_2, emb2_2), dim=1).reshape((-1, 1))
        else:
            score1 = -torch.sum(torch.abs(emb1_1 - emb2_1), dim=1).reshape((-1, 1))
            score2 = -torch.sum(torch.abs(emb1_1 - emb2_2), dim=1).reshape((-1, 1))
            score3 = -torch.sum(torch.abs(emb1_2 - emb2_1), dim=1).reshape((-1, 1))
            score4 = -torch.sum(torch.abs(emb1_2 - emb2_2), dim=1).reshape((-1, 1))
        if graph_name == 'g1':
            scores = torch.cat([score1, score2, score3, score4], dim=1)
        else:
            scores = torch.cat([score4, score3, score2, score1], dim=1)
        predict_scores = self.score_lin(scores)

        return predict_scores

    def loss(self, input_embs):
        """
        Loss functions for alignment.

        @param input_embs: a tuple of node embedding matrices.
            anchor1_emb, anchor2_emb: embeddings of anchors nodes in G1 and G2.
            context_pos1_emb, context_pos2_emb: embeddings of sampled context positive nodes in G1 and G2.
            context_neg1_emb, context_neg2_emb: embeddings of sampled context negative nodes in G1 and G2.
            anchor_neg1_emb, anchor_neg2_emb: embeddings of sampled negative alignment nodes in G1 and G2.
        @return:
            loss1: within-network link prediction loss.
            loss2: anchor link prediction loss.
        """
        (anchor1_emb, anchor2_emb, context_pos1_emb, context_pos2_emb, context_neg1_emb, context_neg2_emb,
         anchor_neg1_emb, anchor_neg2_emb) = input_embs

        device = anchor1_emb.device
        num_instance1 = anchor1_emb.shape[0]
        num_instance2 = context_neg1_emb.shape[0]
        N_negs = num_instance2 // num_instance1
        dim = anchor1_emb.shape[1]

        # loss for within-network
        term1 = self.score(anchor1_emb, context_pos1_emb, 'g1')
        term2 = self.score(anchor1_emb.repeat(1, N_negs).reshape(-1, dim), context_neg1_emb, 'g1')
        term3 = self.score(anchor2_emb, context_pos2_emb, 'g2')
        term4 = self.score(anchor2_emb.repeat(1, N_negs).reshape(-1, dim), context_neg2_emb, 'g2')

        terms1 = torch.cat([term1, term2], dim=0).reshape((-1,))
        labels1 = torch.cat([torch.ones(num_instance1, device=device), torch.zeros(num_instance2, device=device)])
        terms2 = torch.cat([term3, term4], dim=0).reshape((-1,))
        labels2 = torch.cat([torch.ones(num_instance1, device=device), torch.zeros(num_instance2, device=device)])
        loss1 = self.loss_func1(terms1, labels1) + self.loss_func1(terms2, labels2)

        # loss for cross-network
        term5 = self.score(anchor1_emb, anchor1_emb, 'g1')
        term7 = self.score(anchor2_emb, anchor2_emb, 'g2')
        term6 = self.score(anchor1_emb.repeat(1, N_negs).reshape(-1, dim), anchor_neg1_emb, 'g1')
        term8 = self.score(anchor2_emb.repeat(1, N_negs).reshape(-1, dim), anchor_neg2_emb, 'g2')

        terms3 = torch.cat([term5, term6], dim=0).reshape((-1,))
        labels3 = torch.cat([torch.ones(num_instance1, device=device), torch.zeros(num_instance2, device=device)])
        terms4 = torch.cat([term7, term8], dim=0).reshape((-1,))
        labels4 = torch.cat([torch.ones(num_instance1, device=device), torch.zeros(num_instance2, device=device)])

        loss2 = self.loss_func2(terms3, labels3) + self.loss_func2(terms4, labels4)

        return loss1, loss2


class RelGCN(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, bias=True, activation=None, self_loop=True, dropout=0.0, alpha=0.5,
                 param=True):
        super(RelGCN, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.dropout = dropout
        self.alpha = alpha
        self.param = param

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index, x, edge_type):
        """
        Forward pass of the RelGCN layer without DGL.

        @param x: Node features (num_nodes, in_feat).
        @param edge_index: Edge list (2, num_edges).
        @param edge_type: Edge types (num_edges).
        @return:
            node_repr: Node embedding matrix (num_nodes, out_feat).
        """

        # Sanity check for edge types
        if isinstance(edge_type, torch.Tensor):
            if edge_type.size(0) != edge_index.size(1):
                raise ValueError(f'"edge_type" tensor must have length equal to the number of edges. '
                                 f'Got {edge_type.size(0)} and {edge_index.size(1)}.')

        num_nodes = x.size(0)

        # Self-loop handling
        if self.self_loop:
            if self.param:
                if x.dtype == torch.int64:
                    loop_message = self.loop_weight.index_select(0, x[:num_nodes])
                else:
                    loop_message = torch.matmul(x[:num_nodes], self.loop_weight)
            else:
                loop_message = x[:num_nodes]
        else:
            loop_message = torch.zeros_like(x)

        # Message Passing
        src, dst = edge_index
        msg = self.message(x[src], edge_type)  # Call the message function for each edge

        # Aggregation (similar to DGL's fn.sum)
        # aggregated_msg = scatter_add(msg, dst, dim=0, dim_size=num_nodes)
        aggregated_msg = torch.zeros((num_nodes, *msg.shape[1:]), dtype=msg.dtype, device=msg.device)
        aggregated_msg.index_add_(0, dst, msg)

        # Feature Fusion (scaling with sqrt(alpha))
        # node_repr = aggregated_msg * math.sqrt(self.alpha)
        node_repr = aggregated_msg * self.alpha ** 0.5

        # Adding bias if present
        if self.bias:
            node_repr += self.h_bias

        # Adding self-loop contribution
        if self.self_loop:
            # node_repr += loop_message * math.sqrt(1 - self.alpha)
            node_repr += loop_message * (1 - self.alpha) ** 0.5

        # Applying activation function if specified
        if self.activation:
            node_repr = self.activation(node_repr)

        # Applying dropout
        node_repr = self.dropout(node_repr)

        return node_repr

    def message(self, x_j, edge_type):
        """
        Message passing function without DGL.

        @param x_j: Source node features (num_edges, in_feat).
        @param edge_type: Edge types (num_edges), indicating which relation the edge belongs to.
        @return:
            msg: Messages to be passed along edges (num_edges, out_feat).
        """
        weight = self.weight  # Shape: (num_rels, in_feat, out_feat)

        # Case 1: When input features are integer IDs (e.g., for embedding lookups)
        if x_j.ndim == 1:
            weight = weight.view(-1, weight.shape[2])  # Reshape to (num_rels * in_feat, out_feat)
            flat_idx = edge_type * weight.shape[1] + x_j.to(torch.int64)  # Compute flattened index
            msg = weight.index_select(0, flat_idx)  # Select corresponding weights
        else:
            # Case 2: When input features are real-valued (e.g., continuous node features)
            if self.param:
                selected_weight = weight[edge_type]  # Select relation-specific weights
                msg = torch.bmm(x_j.unsqueeze(1), selected_weight).squeeze(1)  # Batch matrix multiplication
            else:
                msg = x_j  # If no transformation, pass features as-is

        return msg
