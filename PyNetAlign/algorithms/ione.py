import time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
import psutil
import os

from PyNetAlign.data import Dataset
from PyNetAlign.metrics import hits_ks_scores, mrr_score
from PyNetAlign.utils import get_anchor_pairs
from .base_model import BaseModel


class IONE(BaseModel):
    """Embedding-based method IONE for pairwise plain network alignment.
    IONE is proposed by the paper "`Aligning Users Across Social Networks Using Network Embedding <https://www.ijcai.org/Proceedings/16/Papers/254.pdf>`_"
    in IJCAI 2016.

    Parameters
    ----------
    out_dim: int, optional
        The output dimension of the embeddings. Default is 100.
    dtype: torch.dtype, optional
        Data type of the tensors, choose from torch.float32 or torch.float64. Default is torch.float32.
    """
    def __init__(self,
                 out_dim: int = 100,
                 dtype: torch.dtype = torch.float32):
        super(IONE, self).__init__(dtype=dtype)
        assert out_dim > 0, 'Output dimension must be a positive integer'

        self.out_dim = out_dim
        self.base_epochs = 100000

    def train(self,
              dataset: Dataset,
              gid1: int,
              gid2: int,
              use_attr: bool = False,
              total_epochs: int = 100,
              save_log: bool = True,
              verbose: bool = True):
        """
        Parameters
        ----------
        dataset : Dataset
            The dataset containing graphs to be aligned and the training/test data.
        gid1 : int
            The graph id of the first graph to be aligned.
        gid2 : int
            The graph id of the second graph to be aligned.
        use_attr : bool, optional
            Flag for using attributes. **Must be False for IONE**. Default is False.
        total_epochs : int, optional
            Maximum number of training epochs. Default is 10000000.
        save_log : bool, optional
            Flag for saving the log. Default is True.
        verbose : bool, optional
            Flag for printing the log. Default is True.
        """

        assert total_epochs > 0, 'Total epochs must be a positive integer'
        self.check_inputs(dataset, (gid1, gid2), plain_method=True, use_attr=use_attr, pairwise=True, supervised=True)

        logger = self.init_training_logger(dataset, use_attr, additional_headers=['memory', 'infer_time'], save_log=save_log)
        process = psutil.Process(os.getpid())

        graph1, graph2 = dataset.pyg_graphs[gid1], dataset.pyg_graphs[gid2]
        anchor_links = get_anchor_pairs(dataset.train_data, gid1, gid2)
        test_pairs = get_anchor_pairs(dataset.test_data, gid1, gid2)

        two_order_x = IONEUpdate(graph1, self.out_dim, self.dtype).to(self.device)
        two_order_y = IONEUpdate(graph2, self.out_dim, self.dtype).to(self.device)
        
        anchor_map1 = {anchor[0].item(): anchor[1].item() for anchor in anchor_links}
        anchor_map2 = {anchor[1].item(): anchor[0].item() for anchor in anchor_links}

        infer_time = 0
        S = torch.zeros(graph1.num_nodes, graph2.num_nodes, dtype=self.dtype).to(self.device)
        for epoch in range(total_epochs):
            t0 = time.time()
            for _ in range(self.base_epochs):
                two_order_x(i=epoch,
                            iter_count=total_epochs,
                            two_order_embeddings=two_order_x.embeddings,
                            two_order_emb_context_input=two_order_x.emb_context_input,
                            two_order_emb_context_output=two_order_x.emb_context_output,
                            anchors=anchor_map1,
                            same_network=True)
                two_order_y(i=epoch,
                            iter_count=total_epochs,
                            two_order_embeddings=two_order_x.embeddings,
                            two_order_emb_context_input=two_order_x.emb_context_input,
                            two_order_emb_context_output=two_order_x.emb_context_output,
                            anchors=anchor_map2,
                            same_network=False)
            t1 = time.time()
            infer_time += t1 - t0

            S_old = S.clone()
            emb_x = F.normalize(two_order_x.embeddings, p=2, dim=1)
            emb_y = F.normalize(two_order_y.embeddings, p=2, dim=1)
            S = emb_x @ emb_y.T
            diff = torch.norm(S - S_old)
            hits = hits_ks_scores(S, test_pairs, mode='mean')
            mrr = mrr_score(S, test_pairs, mode='mean')
            mem_gb = process.memory_info().rss / 1024 ** 3
            logger.log(epoch=epoch+1,
                       loss=diff.item(),
                       epoch_time=t1-t0,
                       mrr=mrr,
                       hits=hits,
                       memory=round(mem_gb, 4),
                       infer_time=round(infer_time, 4),
                       verbose=verbose)

        return S, logger


class IONEUpdate:
    def __init__(self,
                 graph: Data,
                 out_dim: int,
                 dtype: torch.dtype = torch.float32):
        assert dtype in [torch.float32, torch.float64], 'Invalid floating point dtype'
        self.dtype = dtype
        self.device = 'cpu'

        self.graph = graph
        self.dimension = out_dim

        self.embeddings = torch.empty((self.graph.num_nodes, self.dimension), dtype=self.dtype).uniform_(
            -0.5 / self.dimension, 0.5 / self.dimension)
        self.emb_context_input = torch.zeros(self.graph.num_nodes, self.dimension, dtype=self.dtype)
        self.emb_context_output = torch.zeros(self.graph.num_nodes, self.dimension, dtype=self.dtype)
        self.vertex = (degree(self.graph.edge_index[0], num_nodes=self.graph.num_nodes) +
                       degree(self.graph.edge_index[1], num_nodes=self.graph.num_nodes))

        self.init_rho = 0.025
        self.rho = 0
        self.num_negative = 5
        self.neg_table_size = 10000000

        self.edge_weight = []
        self.prob = torch.zeros(self.graph.num_edges, dtype=self.dtype)
        self.alias = torch.zeros(self.graph.num_edges, dtype=torch.int64)
        self.neg_table = torch.zeros(self.neg_table_size, dtype=torch.int64)

        # Initialize tables
        start = time.time()
        self.init_alias_table()
        print(f'{self.graph.name}: alias table initialized in {time.time() - start:.2f} seconds')
        start = time.time()
        self.init_neg_table()
        print(f'{self.graph.name}: negative table initialized in {time.time() - start:.2f} seconds')

    def forward(self, i, iter_count, two_order_embeddings, two_order_emb_context_input, two_order_emb_context_output,
                anchors, same_network=True):
        vec_error = torch.zeros(self.dimension, dtype=self.dtype).to(self.device)
        if i % int(iter_count / 10) == 0:
            self.rho = self.init_rho * (1.0 - i / iter_count)
            if self.rho < self.init_rho * 0.0001:
                self.rho = self.init_rho * 0.0001

        edge_id = self.sample_edge(torch.rand(1).item(), torch.rand(1).item())
        uid_1, uid_2 = self.graph.edge_index[:, edge_id].cpu()
        uid_1, uid_2 = uid_1.item(), uid_2.item()

        d = 0
        while d < self.num_negative + 1:
            if d == 0:
                label = 1
                target = uid_2
            else:
                neg_index = torch.randint(0, self.neg_table_size, (1,)).item()
                target = self.neg_table[neg_index].cpu().item()
                assert not isinstance(target, torch.Tensor), 'Target should not be a tensor'
                if target == uid_1 or target == uid_2:
                    continue
                label = 0

            vec_error += self.update(vec_u=self.embeddings[uid_1],
                                     vec_v=self.emb_context_input[target],
                                     label=label,
                                     source=uid_1,
                                     target=target,
                                     two_order_embeddings=two_order_embeddings,
                                     two_order_emb_context=two_order_emb_context_input,
                                     anchors=anchors,
                                     same_network=same_network)
            self.update_reverse(vec_u=self.embeddings[target],
                                vec_v=self.emb_context_output[uid_1],
                                label=label,
                                source=target,
                                target=uid_1,
                                two_order_embeddings=two_order_embeddings,
                                two_order_emb_context=two_order_emb_context_output,
                                anchors=anchors,
                                same_network=same_network)
            d = d + 1

        if uid_1 in anchors:
            vec_u = two_order_embeddings[anchors[uid_1]] if not same_network else None
            if vec_u is None:
                self.embeddings[uid_1] += vec_error
            else:
                two_order_embeddings[anchors[uid_1]] += vec_error

        else:
            self.embeddings[uid_1] += vec_error

    def init_alias_table(self):
        self.edge_weight = torch.ones(self.graph.num_edges, dtype=self.dtype)
        norm_prob = F.normalize(self.edge_weight, p=1, dim=0) * self.graph.num_edges

        small_block = torch.flip(torch.argwhere(norm_prob < 1).flatten(), dims=[0])
        large_block = torch.flip(torch.argwhere(norm_prob >= 1).flatten(), dims=[0])

        num_small_block = len(small_block)
        num_large_block = len(large_block)
        while num_small_block > 0 and num_large_block > 0:
            num_small_block = num_small_block - 1
            cur_small_block = small_block[num_small_block]
            num_large_block = num_large_block - 1
            cur_large_block = large_block[num_large_block]

            self.prob[cur_small_block] = norm_prob[cur_small_block]
            self.alias[cur_small_block] = cur_large_block

            norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] - 1
            if norm_prob[cur_large_block] < 1:
                small_block[num_small_block] = cur_large_block
                num_small_block = num_small_block + 1
            else:
                large_block[num_large_block] = cur_large_block
                num_large_block = num_large_block + 1

        while num_large_block > 0:
            num_large_block = num_large_block - 1
            self.prob[large_block[num_large_block]] = 1

        while num_small_block > 0:
            num_small_block = num_small_block - 1
            self.prob[small_block[num_small_block]] = 1

    def sample_edge(self, rand1: float, rand2: float) -> int:
        k = int(len(self.edge_weight) * rand1)
        return k if rand2 < self.prob[k] else self.alias[k]

    def init_neg_table(self):
        total_sum = torch.sum(self.vertex ** 0.75).cpu().item()

        cumulative_sum = 0
        por = 0
        perm_node_list = torch.randperm(self.graph.num_nodes).numpy().tolist()
        list_iter = iter(perm_node_list)
        current = next(list_iter)

        vertex = self.vertex.cpu().numpy().tolist()
        self.neg_table = []
        for i in range(self.neg_table_size):
            if (i + 1) / self.neg_table_size > por:
                cumulative_sum += vertex[current] ** 0.75
                por = cumulative_sum / total_sum
                if por >= 1:
                    self.neg_table.append(current)
                    continue
                if i != 0:
                    current = next(list_iter)
            self.neg_table.append(current)
        self.neg_table = torch.tensor(self.neg_table, dtype=torch.int64).to(self.vertex.device)

    def update(self, vec_u, vec_v, label, source, target, two_order_embeddings, two_order_emb_context,
               anchors, same_network=True):
        if source in anchors:
            vec_u = two_order_embeddings[anchors[source]] if not same_network else two_order_embeddings[source]
        if target in anchors:
            vec_v = two_order_emb_context[anchors[target]] if not same_network else two_order_emb_context[target]

        x = vec_u @ vec_v
        g = (label - torch.sigmoid(x)) * self.rho

        vec_error = g * vec_v
        if target in anchors:
            if same_network:
                vec_v += g * vec_u
            else:
                two_order_emb_context[anchors[target]] += g * vec_u
        else:
            vec_v += g * vec_u

        return vec_error

    def update_reverse(self, vec_u, vec_v, label, source, target, two_order_embeddings, two_order_emb_context,
                       anchors, same_network=True):
        if source in anchors:
            vec_u = two_order_embeddings[anchors[source]] if not same_network else two_order_embeddings[source]
        if target in anchors:
            vec_v = two_order_emb_context[anchors[target]] if not same_network else two_order_emb_context[target]

        x = vec_u @ vec_v
        g = (label - torch.sigmoid(x)) * self.rho

        vec_error = g * vec_v
        if target in anchors:
            if same_network:
                vec_v += g * vec_u
            else:
                two_order_emb_context[anchors[target]] += g * vec_u
        else:
            vec_v += g * vec_u

        uid_1 = source
        if uid_1 in anchors:
            if same_network:
                self.embeddings[uid_1] += vec_error
            else:
                two_order_embeddings[anchors[uid_1]] += vec_error
        else:
            self.embeddings[uid_1] += vec_error

    def to(self, device):
        assert device in ['cpu', 'cuda'] or isinstance(device, torch.device), 'Invalid device'
        self.device = device
        self.embeddings = self.embeddings.to(device)
        self.emb_context_input = self.emb_context_input.to(device)
        self.emb_context_output = self.emb_context_output.to(device)
        self.vertex = self.vertex.to(device)
        self.prob = self.prob.to(device)
        self.alias = self.alias.to(device)
        self.neg_table = self.neg_table.to(device)
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
