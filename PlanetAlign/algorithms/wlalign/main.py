import time
from typing import List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import psutil
import os

from PlanetAlign.data import Dataset
from PlanetAlign.utils import get_anchor_pairs, merge_pyg_graphs_on_anchors, pairwise_cosine_similarity
from PlanetAlign.metrics import hits_ks_scores, mrr_score
from PlanetAlign.algorithms.base_model import BaseModel

from .model import aggregate_label, EmbeddingModel, NegativeSamplingLoss
from .utils import (get_anchor_based_embeddings, get_degree_exp_distribution, get_pyg_successors, get_pyg_predecessors,
                    single_hop_subgraph, get_closest_cross_node_pairs, get_non_anchor_from_merged_graph, get_nodes_outside_subgraph)
from .data import ContextDataset


class WLAlign(BaseModel):
    """Embedding-based method WLAlign for pairwise plain network alignment.
    WLAlign is proposed by the paper "`WL-Align: Weisfeiler-Lehman Relabeling for Aligning Users Across Networks via Regularized Representation Learning. <https://doi.org/10.1109/TKDE.2023.3277843>`_"
    in TKDE 2023.

    Parameters
    ----------
    emb_dim : int, optional
        The dimension of the node embeddings. Default is 128.
    struct_lr : float, optional
        The learning rate for the structural model. Default is 5e-3.
    batch_size : int, optional
        The batch size for training. Default is 1000.
    neg_sample_size : int, optional
        The number of negative samples for training. Default is 20.
    dtype : torch.dtype, optional
        Data type of the tensors, choose from torch.float32 or torch.float64. Default is torch.float32.
    """

    def __init__(self,
                 emb_dim: int = 128,
                 struct_lr: float = 5e-3,
                 batch_size: int = 1000,
                 neg_sample_size: int = 20,
                 dtype: torch.dtype = torch.float32):
        super(WLAlign, self).__init__(dtype=dtype)

        self.emb_dim = emb_dim
        self.struct_lr = struct_lr
        self.batch_size = batch_size
        self.neg_sample_size = neg_sample_size

        self.model = None
        self.criterion = None
        self.cos = None
        self.optimizer = None

        self.verbose = True

    def train(self,
              dataset: Dataset,
              gids: Union[Tuple[int, int], List[int]],
              use_attr: bool = False,
              total_epochs: int = 50,
              struct_epochs: int = 100,
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
            Whether to use node and edge attributes for alignment. Default is False.
        total_epochs : int, optional
            The maximum number of epochs for the optimization. Default is 50.
        struct_epochs : int, optional
            The number of epochs for the structural model training. Default is 100.
        save_log : bool, optional
            Whether to save the evaluation logs. Default is True.
        verbose : bool, optional
            Whether to print the progress during training. Default is True.
        """

        self.check_inputs(dataset, gids, plain_method=True, use_attr=use_attr, pairwise=True, supervised=True)
        gid1, gid2 = gids

        logger = self.init_training_logger(dataset, use_attr, additional_headers=['memory', 'infer_time'], save_log=save_log)
        process = psutil.Process(os.getpid())

        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        anchor_links = get_anchor_pairs(dataset.train_data, gid1, gid2)
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)
        self.verbose = verbose

        inf_t0 = time.time()
        merged_graph, merged_anchors, id2node, node2id = merge_pyg_graphs_on_anchors([graph1, graph2], anchor_links)
        gnd_induced_subset, gnd_induced_edge_index, gnd_induced_mapping = single_hop_subgraph(merged_anchors,
                                                                                              edge_index=merged_graph.edge_index,
                                                                                              relabel_nodes=True)
        gnd_induced_subgraph = Data(edge_index=gnd_induced_edge_index,
                                    num_nodes=len(gnd_induced_subset),
                                    mapping=gnd_induced_subset.numpy(),
                                    anchors=gnd_induced_mapping)
        gnd_induced_subgraph.anchor_successors = {anchor: get_pyg_successors(gnd_induced_subgraph, anchor).numpy() for anchor in gnd_induced_subgraph.anchors.numpy()}
        gnd_induced_subgraph.anchor_predecessors = {anchor: get_pyg_predecessors(gnd_induced_subgraph, anchor).numpy() for anchor in gnd_induced_subgraph.anchors.numpy()}
        gnd_sub_id2node = {i: id2node[nid] for i, nid in enumerate(gnd_induced_subset.numpy())}

        gnd_subgraph_nodes1 = get_non_anchor_from_merged_graph(gnd_induced_subgraph, gnd_sub_id2node, gid1)
        gnd_subgraph_nodes2 = get_non_anchor_from_merged_graph(gnd_induced_subgraph, gnd_sub_id2node, gid2)

        current_anchors = torch.clone(merged_anchors)

        infer_time = time.time() - inf_t0

        # Model initialization
        self.model = EmbeddingModel(merged_graph.num_nodes, self.emb_dim).to(self.dtype).to(self.device)
        self.criterion = NegativeSamplingLoss()
        self.cos = nn.CosineEmbeddingLoss(margin=0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.struct_lr, betas=(0.9, 0.999), eps=1e-8)

        all_candidate_pairs = np.empty((0, 2), dtype=np.int64)
        output_emb1, output_emb2 = None, None
        for epoch in range(total_epochs):
            t0 = time.time()
            subset, edge_index, mapping = single_hop_subgraph(current_anchors, edge_index=merged_graph.edge_index, relabel_nodes=True)
            anchor_induced_subgraph = Data(edge_index=edge_index, num_nodes=len(subset), mapping=subset.numpy(), anchors=mapping)
            if verbose:
                print('Number of nodes in the anchor-induced subgraph:', anchor_induced_subgraph.num_nodes)
            sub_id2node = {i: id2node[nid] for i, nid in enumerate(subset.numpy())}
            merged2sub_dict = {nid: i for i, nid in enumerate(subset.numpy())}
            merged2sub = np.vectorize(lambda x: merged2sub_dict[x] if x in merged2sub_dict else -1)(np.arange(merged_graph.num_nodes))

            subgraph_nodes1 = get_non_anchor_from_merged_graph(anchor_induced_subgraph, sub_id2node, gid1)
            subgraph_nodes2 = get_non_anchor_from_merged_graph(anchor_induced_subgraph, sub_id2node, gid2)

            onehot_embs = get_anchor_based_embeddings(anchor_induced_subgraph).to(self.dtype).to(self.device)
            layer_embs_list = aggregate_label(onehot_embs.weight.data, anchor_induced_subgraph, num_layers=1, device=self.device)

            layer_emb = F.normalize(layer_embs_list[0], p=2, dim=1)
            cross_candidate_pairs = get_closest_cross_node_pairs(layer_emb, subgraph_nodes1, subgraph_nodes2, device=self.device)
            cross_candidate_pairs_by_gnd = get_closest_cross_node_pairs(layer_emb,
                                                                        merged2sub[gnd_induced_subgraph.mapping[gnd_subgraph_nodes1]],
                                                                        merged2sub[gnd_induced_subgraph.mapping[gnd_subgraph_nodes2]],
                                                                        device=self.device)
            candidate_pairs = np.vstack([cross_candidate_pairs_by_gnd, cross_candidate_pairs])
            candidate_pairs_mapped = anchor_induced_subgraph.mapping[candidate_pairs]  # mapped node idx from the anchor induced subgraph to the original merged graph
            all_candidate_pairs = np.unique(np.vstack([all_candidate_pairs, candidate_pairs_mapped]), axis=0)

            # Train the anchor embeddings
            output_embeddings, loss = self.train_anchor(merged_graph, gnd_induced_subgraph, all_candidate_pairs, struct_epochs, curr_epoch=epoch)
            new_anchors = torch.unique(torch.from_numpy(candidate_pairs_mapped.flatten()))
            current_anchors = torch.unique(torch.cat([current_anchors, new_anchors]))
            t1 = time.time()
            infer_time += t1 - t0

            # Evaluate the model
            with torch.no_grad():
                output_embeddings = F.normalize(output_embeddings, p=2, dim=1)
                output_emb1 = output_embeddings[[node2id[(0, node)] for node in range(graph1.num_nodes)]]
                output_emb2 = output_embeddings[[node2id[(1, node)] for node in range(graph2.num_nodes)]]
                S = pairwise_cosine_similarity(output_emb1, output_emb2)
                hits = hits_ks_scores(S, test_pairs, mode='mean')
                mrr = mrr_score(S, test_pairs, mode='mean')
                mem_gb = process.memory_info().rss / 1024 ** 3
                logger.log(epoch=epoch+1,
                           loss=loss,
                           epoch_time=t1-t0,
                           hits=hits,
                           mrr=mrr,
                           memory=round(mem_gb, 4),
                           infer_time=round(infer_time, 4),
                           verbose=verbose)

        return output_emb1, output_emb2, logger

    def train_anchor(self, merged_graph, gnd_subgraph, candidate_pairs, struct_epochs, curr_epoch):
        num_candidates = candidate_pairs.shape[0]
        candidate_nodes1 = torch.from_numpy(candidate_pairs[:, 0]).to(self.device)
        candidate_nodes2 = torch.from_numpy(candidate_pairs[:, 1]).to(self.device)

        node_noise = get_nodes_outside_subgraph(gnd_subgraph, merged_graph)
        nodes_dist = torch.from_numpy(get_degree_exp_distribution(merged_graph.edge_index.flatten())).to(self.device)
        noise_dist = torch.from_numpy(get_degree_exp_distribution(node_noise)).to(self.device)
        if noise_dist.shape[0] == 0:
            noise_dist = nodes_dist

        t0 = time.time()
        overall_loss = 0
        for epoch in range(struct_epochs):
            total_loss = 0
            sampled_src, sampled_tgt = self._get_training_samples(gnd_subgraph)
            sampled_src_mapped, sampled_tgt_mapped = gnd_subgraph.mapping[sampled_src], gnd_subgraph.mapping[sampled_tgt]
            sampled_dataset = ContextDataset(sampled_src_mapped, sampled_tgt_mapped)
            sampled_loader = torch.utils.data.DataLoader(sampled_dataset, batch_size=self.batch_size, shuffle=True)
            for batch_src, batch_tgt in sampled_loader:
                curr_batch_size = batch_src.shape[0]

                batch_src = batch_src.to(self.device)
                batch_tgt = batch_tgt.to(self.device)

                self.optimizer.zero_grad()
                target_input_vecs = self.model.forward_input(batch_tgt)
                source_output_vecs = self.model.forward_output(batch_src)
                self_input_vecs = self.model.forward_self(batch_src)
                self_output_vecs = self.model.forward_self(batch_tgt)

                self_left = self.model.forward_self(candidate_nodes1)
                self_right = self.model.forward_self(candidate_nodes2)

                noise_vecs_self1, noise_vecs_input1, noise_vecs_output1 = self.model.forward_noise(curr_batch_size,
                                                                                                   self.neg_sample_size,
                                                                                                   nodes_dist)
                noise_vecs_self2, noise_vecs_input2, noise_vecs_output2 = self.model.forward_noise(curr_batch_size,
                                                                                                   self.neg_sample_size,
                                                                                                   noise_dist)

                loss = self.criterion(self_input_vecs, self_output_vecs, target_input_vecs, source_output_vecs,
                                      noise_vecs_self1, noise_vecs_input1, noise_vecs_output1,
                                      noise_vecs_self2, noise_vecs_input2, noise_vecs_output2)
                loss += self.cos(self_left, self_right, torch.ones(num_candidates, dtype=self.dtype).to(self.device))
                total_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            if epoch % 10 == 9 and self.verbose:
                print(f'Epoch {curr_epoch + 1}, Struct epoch {epoch + 1}/{struct_epochs}, Loss: {total_loss:.4f}')
            overall_loss += total_loss

        overall_loss /= struct_epochs
        if self.verbose:
            print(f'Epoch {curr_epoch + 1}, Time: {time.time() - t0:.2f}s')

        return self.model.self_embed.weight.data, overall_loss

    def _get_training_samples(self, gnd_subgraph, num_batches=1):
        edge_index = gnd_subgraph.edge_index
        num_edges = gnd_subgraph.num_edges

        # Sample edges
        batch_indices = []
        for _ in range(num_batches):
            indices = np.random.choice(num_edges, size=self.batch_size, replace=True)
            batch_indices.append(indices)
        all_sampled_indices = np.concatenate(batch_indices)

        sample_indices = torch.from_numpy(all_sampled_indices).to(edge_index.device)
        sampled_edges = edge_index[:, sample_indices]  # shape: [2, num_batches * batch_size]

        # Convert sampled edge indices to numpy arrays for vectorized string operations.
        sampled_edges = sampled_edges.cpu().numpy()
        src, tgt = sampled_edges[0], sampled_edges[1]

        # Replace anchor nodes with their successors/predecessors
        def replace_anchor_with_successor(node):
            if node in gnd_subgraph.anchors:
                anchor_successors = gnd_subgraph.anchor_successors
                return anchor_successors[node][np.random.choice(len(anchor_successors[node]))]
            return node

        def replace_anchor_with_predecessor(node):
            if node in gnd_subgraph.anchors:
                anchor_predecessors = gnd_subgraph.anchor_predecessors
                return anchor_predecessors[node][np.random.choice(len(anchor_predecessors[node]))]
            return node

        replaced_src = np.vectorize(replace_anchor_with_successor)(src)
        replaced_tgt = np.vectorize(replace_anchor_with_predecessor)(tgt)

        # Resample edges that at least one of the nodes is an anchor
        filtered_src_indices = np.vectorize(lambda x: x in gnd_subgraph.anchors)(src)
        filtered_tgt_indices = np.vectorize(lambda x: x in gnd_subgraph.anchors)(tgt)
        filtered_src = src[filtered_src_indices | filtered_tgt_indices]
        filtered_tgt = tgt[filtered_src_indices | filtered_tgt_indices]

        # Concatenate replaced and filtered edges
        sampled_src = np.concatenate([replaced_src, filtered_src])
        sampled_tgt = np.concatenate([replaced_tgt, filtered_tgt])
        assert len(sampled_src) == len(sampled_tgt), 'The number of source and target nodes should be equal.'

        return torch.from_numpy(sampled_src).to(torch.int64), torch.from_numpy(sampled_tgt).to(torch.int64)
