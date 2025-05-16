import torch
import torch.nn.functional as F


class MultiNetworkEmb(torch.nn.Module):
    def __init__(self, num_of_nodes, batch_size, K, node_emb_dims, num_layer, layer_emb_dims):
        super(MultiNetworkEmb, self).__init__()
        self.batch_size = batch_size
        self.K = K

        # Parameters
        self.embedding = torch.nn.Parameter(torch.empty(num_of_nodes, node_emb_dims))
        self.L_embedding = torch.nn.Parameter(torch.empty(num_layer + 1, layer_emb_dims))
        self.W = torch.nn.Parameter(torch.empty(node_emb_dims, layer_emb_dims))

        # Initialize with truncated normal (approximation using normal distribution)
        torch.nn.init.trunc_normal_(self.embedding, mean=0.0, std=0.3)
        torch.nn.init.trunc_normal_(self.L_embedding, mean=0.0, std=0.3)
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=0.3)

        # Normalize embeddings
        self.embedding.data = F.normalize(self.embedding.data, p=2, dim=1)
        self.L_embedding.data = F.normalize(self.L_embedding.data, p=2, dim=1)
        self.W.data = F.normalize(self.W.data, p=2, dim=1)

    def forward(self, u_i, u_j, this_layer, label):
        # Step 1: Look up embeddings
        u_i_embedding = self.embedding[u_i]
        u_j_embedding = self.embedding[u_j]

        # Step 2: W * u
        u_i_embedding = torch.matmul(u_i_embedding, self.W)
        u_j_embedding = torch.matmul(u_j_embedding, self.W)

        # Step 3: Look up layer embedding
        l_i_embedding = self.L_embedding[this_layer]

        # Step 4: r_i = u_i * W + l
        r_i = u_i_embedding + l_i_embedding
        r_j = u_j_embedding + l_i_embedding

        # Step 5: Compute inner product
        inner_product = torch.sum(r_i * r_j, dim=1)

        # Loss function
        loss = -torch.sum(F.logsigmoid(label * inner_product))
        return loss
