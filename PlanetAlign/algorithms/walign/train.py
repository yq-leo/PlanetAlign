import torch
import torch.nn.functional as F
from .model import FeatureReconstructLoss


def pred_anchor_links_from_embeddings(edge_index1, edge_index2, node_attr1, node_attr2, trans_layer, model, prior=None, prior_rate=0):
    emb1 = model(node_attr1, edge_index1)
    emb2 = trans_layer(model(node_attr2, edge_index2))

    normalied_emb1 = F.normalize(emb1, p=2, dim=1)
    normalied_emb2 = F.normalize(emb2, p=2, dim=1)
    cossim = normalied_emb2 @ normalied_emb1.T
    if prior is not None:
        cossim = (1 + cossim) / 2 * (1 - prior_rate) + prior * prior_rate
    ind = cossim.argmax(dim=1)

    anchor_links = torch.zeros(ind.size(0), 2, dtype=torch.long)
    anchor_links[:, 0] = ind.view(-1)
    anchor_links[:, 1] = torch.arange(ind.size(0))
    return anchor_links


def train_supervise_align(edge_index1, edge_index2, node_attr1, node_attr2, anchor_links, trans_layer, model, margin=0.2, batch_size=128):
    avg_loss = 0
    neg_sampling_matrix = get_neg_sampling_align(anchor_links, node_attr1.shape[0])
    for j in range(anchor_links.size(0)//batch_size + 1):
        neg_losses = []
        embd0 = model(node_attr1, edge_index1)
        embd1 = model(node_attr2, edge_index2)
        embd1_trans = trans_layer(embd1[anchor_links[j * batch_size: j * batch_size + batch_size, 1], :])
        embd0_curr = embd0[anchor_links[j * batch_size: j * batch_size + batch_size, 0], :]
        dist = 1 - F.cosine_similarity(embd1_trans, embd0_curr)
        for ik in range(2, neg_sampling_matrix.size(1)):
            neg_losses.append(1 - F.cosine_similarity(embd1_trans, embd0[neg_sampling_matrix[j * batch_size: j * batch_size + batch_size, ik], :]))
        neg_losses = torch.stack(neg_losses, dim=-1)
        loss_p = torch.max(margin - dist.unsqueeze(-1) + neg_losses, torch.FloatTensor([0])).sum()
        avg_loss += loss_p
    avg_loss /= anchor_links.size(0)
    return avg_loss


def get_neg_sampling_align(anchor_links, n, neg_sampling_num=5):
    neg_sampling_matrix = torch.zeros(anchor_links.size(0), neg_sampling_num + 2, dtype=torch.long)
    neg_sampling_matrix[:, 0] = anchor_links[:, 0]
    neg_sampling_matrix[:, 1] = anchor_links[:, 1]
    for i in range(anchor_links.size(0)):
        ss = torch.ones(n, dtype=torch.long)
        ss[anchor_links[i, 0]] = 0
        nz = torch.nonzero(ss).view(-1)
        neg_sampling_matrix[i, 2:] = nz[torch.randperm(nz.size(0))[:neg_sampling_num]]
    return neg_sampling_matrix


def train_wgan_adv_pseudo_self(edge_index1, edge_index2, node_attr1, node_attr2, trans_layer, model, w_discriminator, wd_optimizer, batch_d_per_iter=5):
    embd0 = model(node_attr1, edge_index1)
    embd1 = trans_layer(model(node_attr2, edge_index2))

    trans_layer.train()
    w_discriminator.train()
    model.train()

    for j in range(batch_d_per_iter):
        w0 = w_discriminator(embd0)
        w1 = w_discriminator(embd1)
        anchor1 = w1.view(-1).argsort(descending=True)[: embd1.size(0)]
        anchor0 = w0.view(-1).argsort(descending=False)[: embd1.size(0)]
        embd0_anchor = embd0[anchor0, :].clone().detach()
        embd1_anchor = embd1[anchor1, :].clone().detach()
        wd_optimizer.zero_grad()
        loss = -torch.mean(w_discriminator(embd0_anchor)) + torch.mean(w_discriminator(embd1_anchor))
        loss.backward()
        wd_optimizer.step()
        for p in w_discriminator.parameters():
            p.data.clamp_(-0.1, 0.1)
    w0 = w_discriminator(embd0)
    w1 = w_discriminator(embd1)
    anchor1 = w1.view(-1).argsort(descending=True)[: embd1.size(0)]
    anchor0 = w0.view(-1).argsort(descending=False)[: embd1.size(0)]
    embd0_anchor = embd0[anchor0, :]
    embd1_anchor = embd1[anchor1, :]
    loss = -torch.mean(w_discriminator(embd1_anchor))
    return loss


def train_feature_recon(edge_index1, edge_index2, node_attr1, node_attr2, trans_layer, model, recon_models, recon_optimizers, batch_r_per_iter=10):
    recon_model0, recon_model1 = recon_models
    recon1_optimizer, recon2_optimizer = recon_optimizers
    embd0 = model(node_attr1, edge_index1)
    embd1 = trans_layer(model(node_attr2, edge_index2))

    recon_model0.train()
    recon_model1.train()
    trans_layer.train()
    model.train()
    embd0_copy = embd0.clone().detach()
    embd1_copy = embd1.clone().detach()
    criterion = FeatureReconstructLoss()
    for t in range(batch_r_per_iter):
        recon1_optimizer.zero_grad()
        loss = criterion(embd0_copy, node_attr1, recon_model0)
        loss.backward()
        recon1_optimizer.step()
    for t in range(batch_r_per_iter):
        recon2_optimizer.zero_grad()
        loss = criterion(embd1_copy, node_attr2, recon_model1)
        loss.backward()
        recon2_optimizer.step()
    loss = 0.5 * criterion(embd0, node_attr1, recon_model0) + 0.5 * criterion(embd1, node_attr2, recon_model1)

    return loss
