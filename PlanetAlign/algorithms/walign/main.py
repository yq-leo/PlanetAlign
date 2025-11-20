from typing import Tuple, List, Union
import itertools
import torch
import torch.nn.functional as F
import time
import psutil
import os

from PlanetAlign.data import Dataset
from PlanetAlign.utils import get_anchor_pairs, pairwise_cosine_similarity
from PlanetAlign.metrics import hits_ks_scores, mrr_score
from PlanetAlign.algorithms.base_model import BaseModel

from .model import GCNNet, GATNet, LGCN, TransLayer, WDiscriminator, ReconDNN
from .train import pred_anchor_links_from_embeddings, train_supervise_align, train_wgan_adv_pseudo_self, \
    train_feature_recon


class WAlign(BaseModel):
    """Embedding-based method WAlign for unsupervised pairwise attributed network alignment.
    WAlign is proposed by the paper "`Unsupervised Graph Alignment with Wasserstein Distance Discriminator. <https://dl.acm.org/doi/10.1145/3447548.3467332>`_"
    in KDD 2021.

    Parameters
    ----------
    setup : int, optional
        The setup of the model. Choose from [1, 2, 3, 4]. Default is 4.
    prior_rate : float, optional
        The rate of prior knowledge. Default is 0.02.
    alpha : float, optional
        The hyperparameter balancing the Wasserstein distance loss and the reconstruction loss. Default is 1e-2.
    out_dim : int, optional
        The dimension of the output embeddings. Default is 512.
    lr : float, optional
        The learning rate for the model. Default is 1e-3.
    lr_wd : float, optional
        The learning rate for the Wasserstein discriminator. Default is 1e-2.
    lr_recon : float, optional
        The learning rate for the reconstruction model. Default is 1e-2.
    transform : bool, optional
        Whether to use the transformation layer. Default is True.
    dtype : torch.dtype, optional
        Data type of the tensors, choose from torch.float32 or torch.float64. Default is torch.float32.
    """

    def __init__(self,
                 setup: int = 4,
                 prior_rate: float = 0.02,
                 alpha: float = 1e-2,
                 out_dim: int = 512,
                 lr: float = 1e-3,
                 lr_wd: float = 1e-2,
                 lr_recon: float = 1e-2,
                 transform: bool = True,
                 dtype: torch.dtype = torch.float32):
        super(WAlign, self).__init__(dtype=dtype)
        assert setup in [1, 2, 3, 4], 'Invalid setup value, choose from [1, 2, 3, 4]'

        self.setup = setup
        self.prior_rate = prior_rate
        self.alpha = alpha
        self.out_dim = out_dim
        self.lr = lr
        self.lr_wd = lr_wd
        self.lr_recon = lr_recon
        self.transform = transform

    def train(self,
              dataset: Dataset,
              gids: Union[Tuple[int, int], List[int]],
              total_epochs: int = 20,
              use_attr: bool = True,
              save_log: bool = True,
              verbose: bool = True):
        """
        Parameters
        ----------
        dataset : Dataset
            The dataset containing the graphs to be aligned and the training/test data.
        gids : tuple[int, int] or list[int]
            The indices of the graphs in the dataset to be aligned.
        use_attr : bool, optional
            Whether to use node and edge attributes for alignment. Default is True.
        total_epochs : int, optional
            The total number of training epochs. Default is 20.
        save_log : bool, optional
            Whether to save the training log. Default is True.
        verbose : bool, optional
            Whether to print the progress during training. Default is True.
        """
        
        self.check_inputs(dataset, gids, plain_method=False, use_attr=use_attr, pairwise=True, supervised=False)
        gid1, gid2 = gids

        logger = self.init_training_logger(dataset, use_attr, additional_headers=['memory', 'infer_time'], save_log=save_log)
        process = psutil.Process(os.getpid())

        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        anchor_links = get_anchor_pairs(dataset.train_data, gid1, gid2)
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)

        prior = torch.zeros(graph2.num_nodes, graph1.num_nodes, dtype=self.dtype).to(self.device)
        prior[anchor_links[:, 1], anchor_links[:, 0]] = 1

        edge_index1, edge_index2 = graph1.edge_index.to(self.device), graph2.edge_index.to(self.device)
        if use_attr:
            node_attr1, node_attr2 = graph1.x.to(self.dtype), graph2.x.to(self.dtype)
        else:
            node_attr1 = torch.ones(graph1.num_nodes, 1, dtype=self.dtype)
            node_attr2 = torch.ones(graph2.num_nodes, 1, dtype=self.dtype)

        if anchor_links.shape[0] > 0:
            anchor_emb1 = torch.zeros(graph1.num_nodes, anchor_links.shape[0], dtype=self.dtype)
            anchor_emb2 = torch.zeros(graph2.num_nodes, anchor_links.shape[0], dtype=self.dtype)
            anchor_emb1[anchor_links[:, 0], torch.arange(anchor_links.shape[0])] = 1
            anchor_emb2[anchor_links[:, 1], torch.arange(anchor_links.shape[0])] = 1
            node_attr1 = torch.hstack((node_attr1, anchor_emb1))
            node_attr2 = torch.hstack((node_attr2, anchor_emb2))

        node_attr1, node_attr2 = node_attr1.to(self.device), node_attr2.to(self.device)

        in_dim = node_attr1.shape[1]
        trans_layer = TransLayer(out_dim=self.out_dim,
                                 transform=self.transform and self.setup != 2 and self.setup != 3).to(self.dtype).to(self.device)
        model = self._get_model(in_dim, self.out_dim, self.setup).to(self.dtype).to(self.device)
        trans_optimizer = torch.optim.Adam(itertools.chain(trans_layer.parameters(), model.parameters()),
                                           lr=self.lr, weight_decay=5e-4)

        w_discriminator = WDiscriminator((self.out_dim, self.out_dim)).to(self.dtype).to(self.device)
        wd_optimizer = torch.optim.Adam(w_discriminator.parameters(), lr=self.lr_wd, weight_decay=5e-4)

        recon_model1 = ReconDNN(self.out_dim, in_dim).to(self.dtype).to(self.device)
        recon_model2 = ReconDNN(self.out_dim, in_dim).to(self.dtype).to(self.device)
        recon1_optimizer = torch.optim.Adam(recon_model1.parameters(), lr=self.lr_recon, weight_decay=5e-4)
        recon2_optimizer = torch.optim.Adam(recon_model2.parameters(), lr=self.lr_recon, weight_decay=5e-4)

        for epoch in range(total_epochs):
            t0 = time.time()
            trans_layer.train()
            model.train()

            trans_optimizer.zero_grad()
            if self.setup == 1 or self.setup == 2 or self.setup == 3:
                anchor_links_pred = pred_anchor_links_from_embeddings(edge_index1=edge_index1, edge_index2=edge_index2,
                                                                      node_attr1=node_attr1, node_attr2=node_attr2,
                                                                      trans_layer=trans_layer, model=model,
                                                                      prior=prior, prior_rate=self.prior_rate)
                loss = train_supervise_align(edge_index1=edge_index1, edge_index2=edge_index2,
                                             node_attr1=node_attr1, node_attr2=node_attr2,
                                             anchor_links=anchor_links_pred, trans_layer=trans_layer, model=model)
            else:
                loss = train_wgan_adv_pseudo_self(edge_index1=edge_index1, edge_index2=edge_index2,
                                                  node_attr1=node_attr1, node_attr2=node_attr2,
                                                  trans_layer=trans_layer, model=model,
                                                  w_discriminator=w_discriminator, wd_optimizer=wd_optimizer)

            infer_time = time.time() - t0

            loss_feature = train_feature_recon(edge_index1=edge_index1, edge_index2=edge_index2,
                                               node_attr1=node_attr1, node_attr2=node_attr2,
                                               trans_layer=trans_layer, model=model,
                                               recon_models=(recon_model1, recon_model2),
                                               recon_optimizers=(recon1_optimizer, recon2_optimizer))
            loss = (1 - self.alpha) * loss + self.alpha * loss_feature

            loss.backward()
            trans_optimizer.step()
            t1 = time.time()

            trans_layer.eval()
            model.eval()
            with torch.no_grad():
                emb1 = model(node_attr1, edge_index1)
                emb2 = model(node_attr2, edge_index2)
                S = pairwise_cosine_similarity(emb1, emb2)
                hits = hits_ks_scores(S, test_pairs, mode='mean')
                mrr = mrr_score(S, test_pairs, mode='mean')
                mem_gb = process.memory_info().rss / 1024 ** 3
                logger.log(epoch=epoch+1,
                           loss=loss.item(),
                           epoch_time=t1-t0,
                           hits=hits,
                           mrr=mrr,
                           memory=round(mem_gb, 4),
                           infer_time=round(infer_time, 4),
                           verbose=verbose)

        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        return emb1, emb2, logger

    @staticmethod
    def _get_model(in_dim, out_dim, setup):
        if setup == 1:
            return GCNNet(in_dim, out_dim)
        elif setup == 2:
            return GATNet(in_dim, out_dim)
        elif setup == 3 or setup == 4:
            return LGCN(in_dim, out_dim)
        else:
            raise ValueError('Invalid setup value, choose from [1, 2, 3, 4]')

