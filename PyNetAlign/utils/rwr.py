import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


def get_batch_rwr_scores(graph, landmarks, restart_prob=0.15, max_iters=1000, tol=1e-6, connect_isolated=False, dtype=torch.float32, device='cpu'):
    r"""Compute Random Walk with Restart (RWR) scores for a batch of landmarks in a single graph."""
    batch_landmark_vecs = torch.zeros(graph.num_nodes, len(landmarks)).to(dtype).to(device)
    batch_landmark_vecs[landmarks, torch.arange(len(landmarks))] = 1
    batch_rwr_vecs = torch.ones(graph.num_nodes, len(landmarks)).to(dtype).to(device)

    adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).squeeze().to(dtype)
    if connect_isolated:
        adj[torch.where(~adj.sum(1).bool())] = torch.ones(graph.num_nodes, dtype=dtype)
    trans_mat = F.normalize(adj.to(device), p=1, dim=1).T
    for i in range(max_iters):
        batch_rwr_vecs_old = torch.clone(batch_rwr_vecs)
        batch_rwr_vecs = (1 - restart_prob) * trans_mat @ batch_rwr_vecs + restart_prob * batch_landmark_vecs
        diff = torch.max(torch.abs(batch_rwr_vecs - batch_rwr_vecs_old))
        if diff.item() < tol:
            break

    return batch_rwr_vecs
