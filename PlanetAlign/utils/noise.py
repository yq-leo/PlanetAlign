from typing import Union, Optional, List, Tuple

import numpy as np
import torch
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data

from PlanetAlign.data import Dataset


def perturb_edges(graph: Data,
                  noise_ratio: float,
                  seed: Optional[int] = None) -> torch.Tensor:
    """
    Add structural noise by perturbing edges in a PyG dataset.

    Parameters
    ----------
    graph : PyG graph
        The input graph to perturb.
    noise_ratio : float
        The ratio of edges to perturb.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    torch.Tensor
        The perturbed edge index of the graph.
    """

    num_edges = graph.num_edges
    num_perturb_edges = int(num_edges * noise_ratio)
    num_nodes = graph.num_nodes

    edge_set = set()
    for i in range(graph.edge_index.size(1)):
        u, v = graph.edge_index[0, i].item(), graph.edge_index[1, i].item()
        if u != v:
            edge_set.add((min(u, v), max(u, v)))

    rng_state = None
    if seed is not None:
        rng_state = np.random.get_state()  # save current state
        np.random.seed(seed)  # set seed for reproducibility

    cnt = 0
    while cnt < num_perturb_edges:
        u, v = np.random.randint(0, num_nodes), np.random.randint(0, num_nodes)
        if u == v:
            continue

        if (min(u, v), max(u, v)) in edge_set:
            edge_set.remove((min(u, v), max(u, v)))
        else:
            edge_set.add((min(u, v), max(u, v)))
        cnt += 1

    if seed is not None:
        np.random.set_state(rng_state)

    # Convert edge_set back to edge_index
    new_edge_index = torch.tensor(list(edge_set), dtype=torch.int64).T
    new_edge_index = to_undirected(new_edge_index)

    return new_edge_index


def add_edge_noises(dataset: Dataset,
                    noise_ratio: float,
                    gids: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
                    seed: Optional[int] = None,
                    inplace: bool = False) -> Dataset:
    """
    Add structural noise to graphs in a PlanetAlign dataset by perturbing edges.

    Parameters
    ----------
    dataset : PyG dataset
        The input dataset containing graphs.
    noise_ratio : float
        The ratio of edges to perturb in each graph.
    gids : int, list of int, or tuple of int
        The graph IDs to perturb. If None, all graphs will be perturbed.
    seed : int, optional
        Random seed for reproducibility.
    inplace : bool, optional
        If True, modify the dataset in place. Otherwise, return a new dataset.

    Returns
    -------
    PyG dataset
        The dataset with perturbed edges.
    """
    assert 0 <= noise_ratio <= 1, "Noise ratio must be between 0 and 1."
    if gids is not None:
        if isinstance(gids, int):
            gids = [gids]
        elif isinstance(gids, list) or isinstance(gids, tuple):
            gids = list(gids)
        else:
            raise TypeError("gids must be an int, list of int, or tuple of int.")
    else:
        gids = list(range(len(dataset.pyg_graphs)))

    assert all(0 <= gid < len(dataset.pyg_graphs) for gid in gids), "Invalid graph IDs."

    if not inplace:
        dataset = dataset.clone()

    for gid in gids:
        graph = dataset.pyg_graphs[gid]
        edge_index = perturb_edges(graph, noise_ratio, seed)
        dataset.pyg_graphs[gid].edge_index = edge_index

    return dataset


def flip_attributes(graph: Data,
                   noise_ratio: float,
                   seed: Optional[int] = None) -> torch.Tensor:
    """
    Add attribute noise by flipping node attributes in a PyG graph.

    Parameters
    ----------
    graph : PyG graph
        The input graph to perturb.
    noise_ratio : float
        The ratio of attributes to flip.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    torch.Tensor
        The perturbed node attributes of the graph.
    """

    def is_binary_tensor(tensor: torch.Tensor) -> bool:
        """
        Check if a PyTorch tensor contains only binary values (0 and 1).

        Parameters
        ----------
        tensor : torch.Tensor
            The input tensor to check.

        Returns
        -------
        bool
            True if tensor contains only 0 and 1, False otherwise.
        """
        unique_vals = torch.unique(tensor)
        return torch.all((unique_vals == 0) | (unique_vals == 1)).item()

    assert is_binary_tensor(graph.x), "Node attributes must be binary (0 and 1)."
    num_nodes, num_attrs = graph.x.size()
    num_flip_attrs = int(num_attrs * noise_ratio)

    rng_state = None
    if seed is not None:
        rng_state = np.random.get_state()  # save current state
        np.random.seed(seed)  # set seed for reproducibility

    flipped_x = torch.clone(graph.x)
    for idx in range(num_nodes):
        perturbed_attr = np.random.choice(num_attrs, num_flip_attrs, replace=False)
        flipped_x[idx, perturbed_attr] = 1 - flipped_x[idx, perturbed_attr]

    if seed is not None:
        np.random.set_state(rng_state)

    return flipped_x


def perturb_attributes_gaussian(graph: Data,
                                std: float,
                                seed: Optional[int] = None) -> torch.Tensor:
    """
    Add Gaussian noise to node attributes in a PyG graph.

    Parameters
    ----------
    graph : PyG graph
        The input graph to perturb.
    std : float
        Standard deviation of the Gaussian noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    torch.Tensor
        The perturbed node attributes of the graph.
    """

    rng_state = None
    if seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)

    x = graph.x
    mean = x.mean(dim=0, keepdim=True)
    std_dev = x.std(dim=0, keepdim=True) + 1e-12
    x_norm = (x - mean) / std_dev

    noise = torch.randn_like(x_norm) * std
    if seed is not None:
        torch.set_rng_state(rng_state)

    x_noisy = (x_norm + noise) * std_dev + mean
    return x_noisy


def add_attr_noises(dataset: Dataset,
                    mode: str,
                    noise_ratio: float,
                    gids: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
                    seed: Optional[int] = None,
                    inplace: bool = False) -> Dataset:
    """
    Add attribute noise to graphs in a PlanetAlign dataset by flipping node attributes.
    
    Parameters
    ----------
    dataset : PyG dataset
        The input dataset containing graphs.
    mode: str
        The mode of noise to add. Options are 'flip' or 'gaussian'.
    noise_ratio : float
        The ratio of attributes to flip in each graph.
    gids : int, list of int, or tuple of int
        The graph IDs to perturb. If None, all graphs will be perturbed.
    seed : int, optional
        Random seed for reproducibility.
    inplace : bool, optional
        If True, modify the dataset in place. Otherwise, return a new dataset.

    Returns
    -------
    PyG dataset
        The dataset with perturbed attributes.
    """
    assert 0 <= noise_ratio <= 1, "Noise ratio must be between 0 and 1."
    if gids is not None:
        if isinstance(gids, int):
            gids = [gids]
        elif isinstance(gids, list) or isinstance(gids, tuple):
            gids = list(gids)
        else:
            raise TypeError("gids must be an int, list of int, or tuple of int.")
    else:
        gids = list(range(len(dataset.pyg_graphs)))

    assert all(0 <= gid < len(dataset.pyg_graphs) for gid in gids), "Invalid graph IDs."

    if not inplace:
        dataset = dataset.clone()

    for gid in gids:
        graph = dataset.pyg_graphs[gid]
        if mode == 'flip':
            x = flip_attributes(graph, noise_ratio, seed)
        elif mode == 'gaussian':
            x = perturb_attributes_gaussian(graph, noise_ratio, seed)
        else:
            raise ValueError("Invalid mode. Choose either 'flip' or 'gaussian'.")
        dataset.pyg_graphs[gid].x = x

    return dataset


def perturb_supervision(dataset: Dataset,
                        noise_ratio: float,
                        src_gid: int = 0,
                        dst_gid: int = 1,
                        seed: Optional[int] = None) -> torch.Tensor:
    """
    Add supervision noise to PyNetAlign dataset object

    Parameters
    ----------
    dataset : Dataset
        The input dataset containing graphs.
    noise_ratio: float
        The ratio of supervision to perturb.
    src_gid : int, optional
        The graph ID of the source graph. Default is 0.
    dst_gid : int, optional
        The graph ID of the destination graph. Default is 1.
    seed : int, optional
        Random seed for reproducibility.
    """

    assert 0 <= noise_ratio <= 1, "Noise ratio must be between 0 and 1."
    assert src_gid < len(dataset.pyg_graphs), f"Source graph ID {src_gid} is out of range."
    assert dst_gid < len(dataset.pyg_graphs), f"Destination graph ID {dst_gid} is out of range."
    assert src_gid != dst_gid, "Source and destination graph IDs must be different."

    rng_state = None
    if seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)

    dst_test_nodes = torch.unique(dataset.test_data[:, dst_gid])
    dst_nodes = torch.arange(dataset.pyg_graphs[dst_gid].num_nodes)
    candidate_noisy_dst_anchors = dst_nodes[~torch.isin(dst_nodes, dst_test_nodes)]

    noisy_train_data = dataset.train_data.clone()
    num_noisy_src_anchors = int(len(dataset.train_data) * noise_ratio)
    noisy_src_anchors_idx = torch.randperm(len(dataset.train_data))[:num_noisy_src_anchors]
    for noisy_src_anchor_idx in noisy_src_anchors_idx:
        dst_anchor = dataset.train_data[noisy_src_anchor_idx, dst_gid]
        noisy_anchor = dst_anchor
        while noisy_anchor == dst_anchor:
            noisy_anchor = candidate_noisy_dst_anchors[
                torch.randint(0, len(candidate_noisy_dst_anchors), (1,)).item()
            ]
        noisy_train_data[noisy_src_anchor_idx, dst_gid] = noisy_anchor

    if seed is not None:
        torch.set_rng_state(rng_state)

    return noisy_train_data


def add_sup_noises(dataset: Dataset,
                   noise_ratio: float,
                   src_gid: int = 0,
                   dst_gid: int = 1,
                   seed: Optional[int] = None,
                   inplace: bool = False) -> Dataset:
    """
    Add supervision noise to graphs in a PyNetAlign dataset by setting noisy anchors.

    Parameters
    ----------
    dataset : Dataset
        The input dataset containing graphs.
    noise_ratio: float
        The ratio of supervision to perturb.
    src_gid : int, optional
        The graph ID of the source graph. Default is 0.
    dst_gid : int, optional
        The graph ID of the destination graph. Default is 1.
    seed : int, optional
        Random seed for reproducibility.
    inplace : bool, optional
        If True, modify the dataset in place. Otherwise, return a new dataset.

    Returns
    -------
    PyG dataset
        The dataset with perturbed supervision.
    """

    if not inplace:
        dataset = dataset.clone()
    dataset.train_data = perturb_supervision(dataset, noise_ratio, src_gid, dst_gid, seed)
    return dataset

