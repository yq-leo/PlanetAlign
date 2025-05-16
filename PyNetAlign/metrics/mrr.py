import torch


def mrr_score(similarity: torch.Tensor,
              test_pairs: torch.Tensor,
              mode: str = 'mean') -> float:
    r"""Mean Reciprocal Rank (MRR) score of pairwise alignment results.

    Parameters
    ----------
    similarity : torch.Tensor
        Similarity matrix of shape (n1, n2) where n1 and n2 are the number of nodes in graph1 and graph2.
    test_pairs : torch.Tensor
        Test pairs of shape (m, 2) where m is the number of test pairs.
    mode : str, optional
        Mode for MRR score. Options are 'mean', 'max', 'ltr' (left-to-right), 'rtl' (right-to-left). Default is 'mean'.

    """

    if mode == 'mean':
        mrr = mrr_mean_score(similarity, test_pairs)
    elif mode == 'max':
        mrr = mrr_max_score(similarity, test_pairs)
    elif mode == 'ltr':
        mrr = mrr_ltr_score(similarity, test_pairs)
    elif mode == 'rtl':
        mrr = mrr_rtl_score(similarity, test_pairs)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return mrr


def mrr_ltr_score(similarity, test_pairs):
    r"""Mean Reciprocal Rank (MRR) score of graph1(left) to graph2(right) alignment."""
    test_pairs = test_pairs.to(similarity.device)
    ranks1 = torch.argsort(-similarity[test_pairs[:, 0]], dim=1)
    signal1_hit = ranks1 == test_pairs[:, 1].view(-1, 1)
    mrr = torch.mean(1 / (torch.where(signal1_hit)[1].float() + 1)).item()
    return mrr


def mrr_rtl_score(similarity, test_pairs):
    r"""Mean Reciprocal Rank (MRR) score of graph2(right) to graph1(left) alignment."""
    test_pairs = test_pairs.to(similarity.device)
    ranks2 = torch.argsort(-similarity.T[test_pairs[:, 1]], dim=1)
    signal2_hit = ranks2 == test_pairs[:, 0].view(-1, 1)
    mrr = torch.mean(1 / (torch.where(signal2_hit)[1].float() + 1)).item()
    return mrr


def mrr_max_score(similarity, test_pairs):
    r"""Max Mean Reciprocal Rank (MRR) score of left-to-right and right-to-left alignments."""
    mrr_ltr = mrr_ltr_score(similarity, test_pairs)
    mrr_rtl = mrr_rtl_score(similarity, test_pairs)
    mrr = max(mrr_ltr, mrr_rtl)

    return mrr


def mrr_mean_score(similarity, test_pairs):
    r"""Mean Mean Reciprocal Rank (MRR) score of left-to-right and right-to-left alignments."""
    mrr_ltr = mrr_ltr_score(similarity, test_pairs)
    mrr_rtl = mrr_rtl_score(similarity, test_pairs)
    mrr = (mrr_ltr + mrr_rtl) / 2

    return mrr
