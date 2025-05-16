from typing import Optional, Union, List, Tuple
import warnings

from PlanetAlign.metrics import hits_ks_scores, mrr_score


def eval_and_print_basic(similarity, test_pairs,
                         title: Optional[str] = None,
                         modes: Union[List[str], Tuple[str, ...]] = ('mean', 'max'),
                         digits: int = 4):
    metrics_lines = []
    for mode in modes:
        hits_ks = hits_ks_scores(similarity, test_pairs, ks=[1, 10, 30, 50], mode=mode)
        mrr = mrr_score(similarity, test_pairs, mode=mode)
        metrics_line = get_hits_mrr_log(hits_ks, mrr, prefix=mode.capitalize(), digits=digits)
        metrics_lines.append(metrics_line)

    line_length = len(metrics_lines[0])
    if title:
        title_str = f" {title} "
        header_line = title_str.center(line_length, '-')
    else:
        header_line = "-" * line_length

    print(header_line)
    for line in metrics_lines:
        print(line)
    print("-" * line_length)


def get_hits_mrr_log(hits, mrr, prefix: Optional[str] = None, digits: int = 4) -> str:
    """
    Get formatted strings of Hits@K and MRR scores.

    <prefix> | Hits\@1: <value> | Hits\@10: <value> | Hits\@30: <value> | Hits\@50: <value> | MRR: <value>

    Parameters
    ----------
    hits : dict
        Dictionary containing Hits@K scores.
    mrr : float
        Mean Reciprocal Rank score.
    prefix : str, optional
        Prefix string to be included in the output. Default is None.
    digits : int, optional
        Number of decimal places to display for the scores. Default is 4.

    Returns
    -------
    str
        Formatted string containing Hits@K and MRR scores.
    """

    MAX_PREFIX_WIDTH = 5
    if prefix and len(prefix) > MAX_PREFIX_WIDTH:
        warnings.warn(f"Prefix length exceeds {MAX_PREFIX_WIDTH} characters; output may be misaligned.", UserWarning)
    metrics_line = " | ".join([f"Hits@{k}: {v:.{digits}f}" for k, v in hits.items()]) + f" | MRR: {mrr:.{digits}f}"
    if prefix:
        prefix_str = f"{prefix:<{MAX_PREFIX_WIDTH}}" if len(prefix) <= MAX_PREFIX_WIDTH else prefix
        full_line = f"{prefix_str} | {metrics_line}"
    else:
        full_line = metrics_line
    return full_line
