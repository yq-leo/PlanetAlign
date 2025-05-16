import torch
import torch.nn as nn
import numpy as np
from collections import Counter, defaultdict
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from typing import List, Tuple, Union
from torch import Tensor


def get_anchor_based_embeddings(graph):
    # Original function: get_graph_label(network, init_dim, device)
    # Create an embedding of shape (num_nodes, num_nodes)
    init_embed = nn.Embedding(graph.num_nodes, graph.num_nodes)
    # Create an identity matrix (one-hot vectors)
    onehot_emb = torch.eye(graph.num_nodes)

    # Initialize all embeddings to zeros
    init_embed.weight.data.zero_()
    init_embed.weight.data[graph.anchors] = onehot_emb[graph.anchors]

    return init_embed


def get_degree_exp_distribution(node_samples, p=0.75):
    if isinstance(node_samples, torch.Tensor):
        node_samples = node_samples.cpu().numpy()

    # Count the frequency of each node in the samples
    node_counts = Counter(node_samples)
    total_count = len(node_samples)

    # Compute the frequency of each node as a fraction of the total count
    node_freqs = {node: count / total_count for node, count in node_counts.items()}

    # Compute the degree exponent distribution using the formula: p^(node_freq)
    degree_exp_dist = np.array(list(node_freqs.values())) ** p
    degree_exp_dist /= degree_exp_dist.sum()

    return degree_exp_dist


def get_pyg_successors(graph: Data, node: int):
    r"""Returns a tensor of nodes that are successors of the input node"""
    assert node < graph.num_nodes, f'Node index must be less than the number of nodes in the graph'
    return graph.edge_index[1][graph.edge_index[0] == node]


def get_pyg_predecessors(graph: Data, node: int):
    r"""Returns a tensor of nodes that are predecessors of the input node"""
    assert node < graph.num_nodes, f'Node index must be less than the number of nodes in the graph'
    return graph.edge_index[0][graph.edge_index[1] == node]


def single_hop_subgraph(node_idx: Union[int, List[int], Tensor],
                        edge_index: Tensor,
                        relabel_nodes: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Returns a subgraph of the input graph containing the input node and its neighbors"""
    subset, edge_index, mapping, _ = k_hop_subgraph(node_idx, 1, edge_index, relabel_nodes=relabel_nodes)
    src, dst = edge_index.numpy()
    src_mask = np.vectorize(lambda x: x in mapping)(src)
    dst_mask = np.vectorize(lambda x: x in mapping)(dst)
    edge_index = edge_index[:, src_mask | dst_mask]
    return subset, edge_index, mapping


def get_closest_cross_node_pairs(embeddings, nodes1, nodes2, device='cpu'):
    assert np.all(nodes1 >= 0) and np.all(nodes2 >= 0), 'Node indices should be non-negative.'
    emb1 = embeddings[nodes1].to(device)
    emb2 = embeddings[nodes2].to(device)

    src2dst_sim = emb1 @ emb2.T
    dst2src_sim = emb2 @ emb1.T

    src2dst_closest = torch.argmax(src2dst_sim, dim=1).cpu()
    dst2src_closest = torch.argmax(dst2src_sim, dim=1).cpu()

    src2dst_candidates = np.stack([nodes1, nodes2[src2dst_closest]], axis=1).tolist()
    dst2src_candidates = np.stack([nodes1[dst2src_closest], nodes2], axis=1).tolist()

    src2dst_candidates = set([tuple(pair) for pair in src2dst_candidates])
    dst2src_candidates = set([tuple(pair) for pair in dst2src_candidates])
    candidate = src2dst_candidates.intersection(dst2src_candidates)

    return np.array(list(candidate)).astype(np.int64)


def get_non_anchor_from_merged_graph(graph: Data,
                                     id2node: Union[dict, defaultdict],
                                     gid: int):
    r"""Get non-anchor nodes from the merged (sub)graph."""
    node_list = []
    for i in range(graph.num_nodes):
        if i in graph.anchors or len(id2node[i]) > 1:
            continue
        if id2node[i][0][0] == gid:
            node_list.append(i)
    return np.array(node_list)


def get_nodes_outside_subgraph(subgraph, graph):
    r"""Get nodes in a graph that are outside the given subgraph."""
    assert subgraph.num_nodes <= graph.num_nodes, 'Subgraph should be smaller than the original graph.'
    subset = subgraph.mapping if isinstance(subgraph.mapping, np.ndarray) else subgraph.mapping.cpu().numpy()
    return np.setdiff1d(np.arange(graph.num_nodes), subset)


