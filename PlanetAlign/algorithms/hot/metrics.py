from typing import List, Tuple, Union
import numpy as np
import torch


def multi_align_hits_ks_scores(sim_tensor_dict: dict[int, torch.Tensor],
                               cluster_nodes_dict: dict[int, List[torch.Tensor]],
                               test_links: torch.Tensor,
                               ks: Union[list[int], tuple[int, ...]] = (1, 10, 30, 50),
                               mode: str = 'mean') -> Tuple[dict[int, float], dict[int, float]]:
    assert mode in ['mean', 'max'], 'Invalid mode.'
    num_graphs = test_links.shape[1]

    pairwise_hits_ks_arr = {k: [] for k in ks}
    high_order_hits_ks_arr = {k: [] for k in ks}

    for src_id in range(num_graphs):
        pairwise_hits, high_order_hits = multi_align_hits_ks_scores_from_src(sim_tensor_dict, cluster_nodes_dict, test_links, src_id, ks)
        for k in ks:
            pairwise_hits_ks_arr[k].append(pairwise_hits[k])
            high_order_hits_ks_arr[k].append(high_order_hits[k])

    pairwise_hits_ks = {}
    high_order_hits_ks = {}
    for k in ks:
        pairwise_hits_ks[k] = np.mean(pairwise_hits_ks_arr[k]) if mode == 'mean' else np.max(pairwise_hits_ks_arr[k])
        high_order_hits_ks[k] = np.mean(high_order_hits_ks_arr[k]) if mode == 'mean' else np.max(high_order_hits_ks_arr[k])

    return pairwise_hits_ks, high_order_hits_ks


def multi_align_mrr_score(sim_tensor_dict: dict[int, torch.Tensor],
                          cluster_nodes_dict: dict[int, List[torch.Tensor]],
                          test_links: torch.Tensor,
                          mode: str = 'mean') -> float:
    assert mode in ['mean', 'max'], 'Invalid mode.'
    num_graphs = test_links.shape[1]

    mrr_list = []
    for src_id in range(num_graphs):
        mrr = mutli_align_mrr_score_from_src(sim_tensor_dict, cluster_nodes_dict, test_links, src_id)
        mrr_list.append(mrr)
    mrr = np.mean(mrr_list) if mode == 'mean' else np.max(mrr_list)
    return mrr


def multi_align_hits_ks_scores_from_src(sim_tensor_dict: dict[int, torch.Tensor],
                                        cluster_nodes_dict: dict[int, List[torch.Tensor]],
                                        test_links: torch.Tensor,
                                        src_id: int,
                                        ks: Union[list[int], tuple[int, ...]] = (1, 10, 30, 50)) -> Tuple[dict[int, float], dict[int, float]]:
    assert src_id < test_links.shape[1], 'Source graph ID is out of range.'

    num_graphs = test_links.shape[1]
    num_clusters = len(cluster_nodes_dict)

    pairwise_hits_ks = {k: 0 for k in ks}
    high_order_hits_ks = {k: 0 for k in ks}
    for i in range(num_clusters):
        cluster_nodes = cluster_nodes_dict[i]
        sim_tensor = sim_tensor_dict[i]

        cluster_test_links = test_links[torch.isin(test_links[:, src_id], cluster_nodes[src_id])]
        node2id_src = {cluster_nodes[src_id][idx].item(): idx for idx in range(len(cluster_nodes[src_id]))}
        test2src = {idx: node2id_src[cluster_test_links[idx, src_id].item()] for idx in
                    range(cluster_test_links.shape[0])}
        for test_idx in range(cluster_test_links.shape[0]):
            inside = np.any(
                [cluster_test_links[test_idx, gid] in cluster_nodes[gid] for gid in range(num_graphs) if gid != src_id])
            if not inside:
                continue

            sim_tensor_slice = torch.index_select(sim_tensor, dim=src_id,
                                                  index=torch.tensor(test2src[test_idx])).squeeze(0)
            sorted_indices = torch.argsort(sim_tensor_slice.ravel(), descending=True)
            arg_rank_list = torch.stack(torch.unravel_index(sorted_indices, sim_tensor_slice.shape))
            rank_list = torch.empty_like(arg_rank_list).T
            for gid in range(num_graphs):
                if gid == src_id:
                    continue
                rank_list[:, gid - (gid > src_id)] = cluster_nodes[gid][arg_rank_list[gid - (gid > src_id)]]

            hh_ind = torch.where(
                (rank_list == cluster_test_links[test_idx][[gid for gid in range(num_graphs) if gid != src_id]]).all(
                    dim=1))[0]
            pp_ind = torch.where(
                (rank_list == cluster_test_links[test_idx][[gid for gid in range(num_graphs) if gid != src_id]]).any(
                    dim=1))[0]
            if len(hh_ind) != 0:
                for k in ks:
                    if k > hh_ind[0]:
                        high_order_hits_ks[k] += 1
            if len(pp_ind) != 0:
                for k in ks:
                    if k > pp_ind[0]:
                        pairwise_hits_ks[k] += 1

    for k in ks:
        pairwise_hits_ks[k] /= test_links.shape[0]
        high_order_hits_ks[k] /= test_links.shape[0]

    return pairwise_hits_ks, high_order_hits_ks


def mutli_align_mrr_score_from_src(sim_tensor_dict: dict[int, torch.Tensor],
                                   cluster_nodes_dict: dict[int, List[torch.Tensor]],
                                   test_links: torch.Tensor,
                                   src_id: int):
    assert src_id < test_links.shape[1], 'Source graph ID is out of range.'

    num_graphs = test_links.shape[1]
    max_num_nodes = torch.vstack([torch.tensor(sim_tensor.shape) for sim_tensor in sim_tensor_dict.values()]).sum(dim=0).max().item()
    num_clusters = len(cluster_nodes_dict)

    mrr = 0
    for i in range(num_clusters):
        cluster_nodes = cluster_nodes_dict[i]
        sim_tensor = sim_tensor_dict[i]

        cluster_test_links = test_links[torch.isin(test_links[:, src_id], cluster_nodes[src_id])]
        node2id_src = {cluster_nodes[src_id][idx].item(): idx for idx in range(len(cluster_nodes[src_id]))}
        test2src = {idx: node2id_src[cluster_test_links[idx, src_id].item()] for idx in
                    range(cluster_test_links.shape[0])}
        for test_idx in range(cluster_test_links.shape[0]):
            inside = np.any(
                [cluster_test_links[test_idx, gid] in cluster_nodes[gid] for gid in range(num_graphs) if gid != src_id])
            if not inside:
                mrr += 1 / max_num_nodes
                continue

            sim_tensor_slice = torch.index_select(sim_tensor, dim=src_id, index=torch.tensor(test2src[test_idx])).squeeze(0)
            sorted_indices = torch.argsort(sim_tensor_slice.ravel(), descending=True)
            arg_rank_list = torch.stack(torch.unravel_index(sorted_indices, sim_tensor_slice.shape))
            rank_list = torch.empty_like(arg_rank_list).T
            for gid in range(num_graphs):
                if gid == src_id:
                    continue
                rank_list[:, gid - (gid > src_id)] = cluster_nodes[gid][arg_rank_list[gid - (gid > src_id)]]

            hh_ind = torch.where((rank_list == cluster_test_links[test_idx][[gid for gid in range(num_graphs) if gid != src_id]]).all(dim=1))[0]
            if len(hh_ind) != 0:
                mrr += 1 / (hh_ind[0].item() + 1)

    mrr = mrr / test_links.shape[0]
    return mrr
