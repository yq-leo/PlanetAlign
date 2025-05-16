from typing import Union
import torch


def hits_ks_scores(simiarity: torch.Tensor,
                   test_pairs: torch.Tensor,
                   ks: Union[list[int], tuple[int, ...]] = (1, 10, 30, 50),
                   mode: str = 'mean') -> dict[int, float]:
    r"""Hits@K scores of pairwise alignment results.

    Parameters
    ----------
    simiarity : torch.Tensor
        Similarity matrix of shape (n1, n2) where n1 and n2 are the number of nodes in graph1 and graph2.
    test_pairs : torch.Tensor
        Test pairs of shape (m, 2) where m is the number of test pairs.
    ks : list[int] or tuple[int, ...], optional
        List of k values for Hits@K scores. Default is (1, 10, 30, 50).
    mode : str, optional
        Mode for Hits@K scores. Options are 'mean', 'max', 'ltr' (left-to-right), 'rtl' (right-to-left). Default is 'mean'.
    """

    if mode == 'mean':
        hits_ks = hits_ks_mean_scores(simiarity, test_pairs, ks=ks)
    elif mode == 'max':
        hits_ks = hits_ks_max_scores(simiarity, test_pairs, ks=ks)
    elif mode == 'ltr':
        hits_ks = hits_ks_ltr_scores(simiarity, test_pairs, ks=ks)
    elif mode == 'rtl':
        hits_ks = hits_ks_rtl_scores(simiarity, test_pairs, ks=ks)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return hits_ks


def hits_ks_ltr_scores(similarity, test_pairs, ks=None):
    r"""Hits@K scores of graph1(left) to graph2(right) alignment."""
    test_pairs = test_pairs.to(similarity.device)
    hits_ks = {}
    ranks1 = torch.argsort(-similarity[test_pairs[:, 0]], dim=1)
    signal1_hit = ranks1 == test_pairs[:, 1].view(-1, 1)
    for k in ks:
        hits_ks[k] = (torch.sum(signal1_hit[:, :k]) / test_pairs.shape[0]).item()

    return hits_ks


def hits_ks_rtl_scores(similarity, test_pairs, ks=None):
    r"""Hits@K scores of graph2(right) to graph1(left) alignment."""
    test_pairs = test_pairs.to(similarity.device)
    hits_ks = {}
    ranks2 = torch.argsort(-similarity.T[test_pairs[:, 1]], dim=1)
    signal2_hit = ranks2 == test_pairs[:, 0].view(-1, 1)
    for k in ks:
        hits_ks[k] = (torch.sum(signal2_hit[:, :k]) / test_pairs.shape[0]).item()

    return hits_ks


def hits_ks_max_scores(similarity, test_pairs, ks=None):
    r"""Max Hits@K scores of left-to-right and right-to-left alignments."""
    hits_ks = {}

    hits_ks_ltr = hits_ks_ltr_scores(similarity, test_pairs, ks=ks)
    hits_ks_rtl = hits_ks_rtl_scores(similarity, test_pairs, ks=ks)
    for k in ks:
        hits_ks[k] = max(hits_ks_ltr[k], hits_ks_rtl[k])

    return hits_ks


def hits_ks_mean_scores(similarity, test_pairs, ks=None):
    r"""Mean Hits@K scores of left-to-right and right-to-left alignments."""
    hits_ks = {}

    hits_ks_ltr = hits_ks_ltr_scores(similarity, test_pairs, ks=ks)
    hits_ks_rtl = hits_ks_rtl_scores(similarity, test_pairs, ks=ks)
    for k in ks:
        hits_ks[k] = (hits_ks_ltr[k] + hits_ks_rtl[k]) / 2

    return hits_ks
