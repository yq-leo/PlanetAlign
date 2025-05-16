from typing import List, Tuple, Union
from collections import defaultdict
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import degree
import torch
import torch.nn.functional as F


def get_anchor_pairs(anchor_links, gid1, gid2):
    # TODO: Extend to multiple graphs (more than 2)
    potential_pairs = anchor_links[:, [gid1, gid2]]
    # TODO: potential_pairs[torch.sum(potential_pairs != -1, dim=1) >= 2]
    anchor_pairs = potential_pairs[torch.all(potential_pairs != -1, dim=1)]
    return anchor_pairs


def get_pairwise_anchor_pairs(anchor_links) -> dict:
    num_graphs = anchor_links.shape[1]
    anchor_pairs_dict = {}
    for gid1 in range(num_graphs):
        for gid2 in range(gid1 + 1, num_graphs):
            anchor_pairs = get_anchor_pairs(anchor_links, gid1, gid2)
            anchor_pairs_dict[(gid1, gid2)] = anchor_pairs
    return anchor_pairs_dict


def get_anchor_embeddings(graph, anchors):
    num_anchors = anchors.shape[0]
    anchor_embeddings = torch.zeros(graph.num_nodes, num_anchors, dtype=torch.float32)
    anchor_embeddings[anchors, torch.arange(num_anchors)] = 1
    return anchor_embeddings


def merge_pyg_graphs_on_anchors(pyg_graphs: Union[List[Data], Tuple[Data, ...]],
                                anchor_links: torch.Tensor):
    assert len(pyg_graphs) >= 2, 'At least two graphs are required for merging'
    assert len(pyg_graphs) == anchor_links.shape[1], 'Number of PyG graphs and anchor links dimension do not match'

    anchor_maps = dict()
    anchor_links = anchor_links.cpu().numpy()
    for anchor_link in anchor_links:
        true_anchor = None
        for gid, anchor in enumerate(anchor_link):
            if anchor > -1:
                if true_anchor is None:
                    true_anchor = (gid, int(anchor))
                else:
                    anchor_maps[(gid, int(anchor))] = true_anchor

    merged_node_cnt = 0
    id2node_dict, node2id_list, node2id_dict = defaultdict(list), list(), dict()
    merged_anchors = set()
    for gid, g in enumerate(pyg_graphs):
        node2id = np.zeros(g.num_nodes, dtype=np.int64)
        for node in range(g.num_nodes):
            if (gid, node) in anchor_maps:
                nid = node2id_list[anchor_maps[(gid, node)][0]][anchor_maps[(gid, node)][1]]
                merged_anchors.add(nid)
            else:
                nid = merged_node_cnt
                merged_node_cnt += 1
            node2id[node] = nid
            node2id_dict[(gid, node)] = nid
            id2node_dict[nid].append((gid, node))
        node2id_list.append(node2id)
    merged_anchors = torch.tensor(list(merged_anchors), dtype=torch.int64)

    # Build merged pyg graph
    assert all([g.num_node_features == pyg_graphs[0].num_node_features for g in pyg_graphs]), 'Node features must match'
    assert all([g.num_edge_features == pyg_graphs[0].num_edge_features for g in pyg_graphs]), 'Edge features must match'
    num_node_attr = pyg_graphs[0].num_node_features
    num_edge_attr = pyg_graphs[0].num_edge_features

    merged_edge_index = torch.empty((2, 0), dtype=torch.int64)
    merged_node_attr = torch.zeros((merged_node_cnt, num_node_attr), dtype=pyg_graphs[0].x.dtype) if pyg_graphs[0].x is not None else None
    merged_node_attr_cnt = torch.zeros(merged_node_cnt, dtype=torch.int)
    merged_edge_attr = torch.empty((0, num_edge_attr), dtype=pyg_graphs[0].edge_attr.dtype) if pyg_graphs[0].edge_attr is not None else None
    for gid, g in enumerate(pyg_graphs):
        lookup = torch.from_numpy(node2id_list[gid])
        edge_index_mapped = lookup[g.edge_index]
        merged_edge_index = torch.cat([merged_edge_index, edge_index_mapped], dim=1)
        if g.x is not None:
            merged_node_attr[lookup] += g.x
            merged_node_attr_cnt[lookup] += 1
        if g.edge_attr is not None:
            merged_edge_attr = torch.cat([merged_edge_attr, g.edge_attr], dim=0)
    if merged_node_attr is not None:
        merged_node_attr /= merged_node_attr_cnt.view(-1, 1)
    merged_graph = Data(x=merged_node_attr, edge_index=merged_edge_index, edge_attr=merged_edge_attr, num_nodes=merged_node_cnt)

    return merged_graph, merged_anchors, id2node_dict, node2id_dict


def infer_anchors_from_degree(dataset, topk_ratio=0.1):
    """
    Infer anchor node pairs based on degree similarity. The number of anchor pairs is calculated
    by taking the top k% of the minimum number of nodes in the two graphs.

    Parameters:
    ----------
    dataset : Dataset
        The dataset containing the graphs to be aligned.
    topk_ratio : float, optional
        The ratio of the number of anchor pairs to the minimum number of nodes in the two graphs. Default is 0.1.

    Returns:
    -------
    anchors : torch.Tensor
        The inferred anchor pairs.
    """

    g1, g2 = dataset.pyg_graphs[0], dataset.pyg_graphs[1]
    deg1 = degree(g1.edge_index[0], g1.num_nodes)
    deg2 = degree(g2.edge_index[0], g2.num_nodes)

    deg1 = deg1.unsqueeze(1)
    deg2 = deg2.unsqueeze(0)
    abs_diff = torch.abs(deg1 - deg2)
    sim_matrix = 1 / (1 + abs_diff.float())

    n1, n2 = sim_matrix.shape
    k = int(topk_ratio * min(n1, n2))

    sim_flat = sim_matrix.view(-1)
    _, topk_indices = torch.topk(sim_flat, k)

    node1_idx = topk_indices // n2
    node2_idx = topk_indices % n2

    anchors = torch.stack([node1_idx, node2_idx], dim=1)

    return anchors


def infer_anchors_from_attributes(dataset, topk_ratio=0.1):
    """
    Infer anchors based on attribute similarity. The number of anchor pairs is calculated
    by taking the top k% of the minimum number of nodes in the two graphs.

    Parameters:
    - dataset: The dataset containing the graphs and their attributes.
    - topk_ratio: The ratio of top k% to select as anchors.

    Returns:
    - anchors: A tensor containing the indices of the inferred anchors.
    """
    g1, g2 = dataset.pyg_graphs[0], dataset.pyg_graphs[1]
    assert g1.x is not None and g2.x is not None, "Graph attributes are required for anchor inference."
    assert g1.x.shape[1] == g2.x.shape[1], "Graph attributes must have the same dimension."

    x1 = F.normalize(g1.x, p=2, dim=1)
    x2 = F.normalize(g2.x, p=2, dim=1)

    sim_matrix = x1 @ x2.T

    n1, n2 = sim_matrix.shape
    k = int(topk_ratio * min(n1, n2))

    sim_flat = sim_matrix.view(-1)
    _, topk_indices = torch.topk(sim_flat, k)

    node1_idx = topk_indices // n2
    node2_idx = topk_indices % n2

    anchors = torch.stack([node1_idx, node2_idx], dim=1)

    return anchors
