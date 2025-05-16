import torch


def balance_samples(sample_pairs1, sample_pairs2):
    if len(sample_pairs1) < len(sample_pairs2):
        len_diff = len(sample_pairs2) - len(sample_pairs1)
        idx = torch.randint(len(sample_pairs1), (len_diff,))
        imputes = sample_pairs1[idx]
        balanced_sample_pairs1 = torch.vstack([sample_pairs1, imputes])
        balanced_sample_pairs2 = sample_pairs2
    else:
        len_diff = len(sample_pairs1) - len(sample_pairs2)
        idx = torch.randint(len(sample_pairs2), (len_diff,))
        imputes = sample_pairs2[idx]
        balanced_sample_pairs2 = torch.vstack([sample_pairs2, imputes])
        balanced_sample_pairs1 = sample_pairs1
    return balanced_sample_pairs1, balanced_sample_pairs2
