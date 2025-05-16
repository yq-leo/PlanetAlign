import torch
import torch.nn.functional as F


def get_normalized_neg_exp_dist(emb1, emb2, p=2, device='cpu'):
    emb1 = emb1.to(device)
    emb2 = emb2.to(device)
    normalized_emb1 = F.normalize(emb1, p=p, dim=1)
    normalized_emb2 = F.normalize(emb2, p=p, dim=1)
    return torch.exp(-(normalized_emb1 @ normalized_emb2.T))


def pairwise_cosine_similarity(x, y, p=2):
    normalized_x = F.normalize(x, p=p, dim=1)
    normalized_y = F.normalize(y, p=p, dim=1)
    return normalized_x @ normalized_y.T
