import torch


def uniform_negative_sampling(anchor1, anchor2, n1, n2, n_negs):
    assert len(anchor1) == len(anchor2), "The number of anchors should be the same."

    anchor1, anchor2 = anchor1.cpu(), anchor2.cpu()
    num_anchors = len(anchor1)
    p1 = torch.ones((num_anchors, n2))
    p2 = torch.ones((num_anchors, n1))
    negs1, negs2 = [], []
    for i in range(num_anchors):
        p = p1[i]
        p[anchor2[i]] = 0
        p = p / torch.sum(p)
        samples = torch.multinomial(p, n_negs, replacement=True)
        negs1.append(samples)

        p = p2[i]
        p[anchor1[i]] = 0
        p = p / torch.sum(p)
        samples = torch.multinomial(p, n_negs, replacement=True)
        negs2.append(samples)

    negs1 = torch.vstack(negs1)
    negs2 = torch.vstack(negs2)

    return negs1, negs2


def negative_edge_sampling(prob, anchors, n_negs):
    anchors = anchors.cpu()
    prob = prob.cpu()
    prob = prob[anchors]
    negs = []
    for p in prob:
        p1 = p / torch.sum(p)
        neg_idx = torch.multinomial(p1, n_negs, replacement=True)
        negs.append(neg_idx)
    negs = torch.vstack(negs)
    return negs
