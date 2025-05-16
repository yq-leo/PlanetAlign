import numpy as np
import torch
from torch.utils.data import Dataset


class ContextDataset(Dataset):
    def __init__(self, context_pairs1, context_pairs2):
        # data loading
        self.context_pairs1 = context_pairs1
        self.context_pairs2 = context_pairs2
        self.n_samples = context_pairs1.shape[0]

    def __getitem__(self, index):
        return self.context_pairs1[index], self.context_pairs2[index]

    def __len__(self):
        return self.n_samples


def merge_graphs(graph1, graph2, x1, x2, anchor_links):
    visit = 0
    n1, n2 = graph1.num_nodes, graph2.num_nodes
    anchor_links = anchor_links.detach().cpu().numpy()
    node_mapping = {}

    for i, (node1, node2) in enumerate(anchor_links):
        node_mapping[node2] = node1

    ########################################################
    # merge node features
    x, x_pos, x_attr = [], [], []
    for i in range(n1):
        if not isinstance(x1, tuple):
            x.append(x1[i])
        else:
            x_pos.append(x1[0][i])
            x_attr.append(x1[1][i])

    for i in range(n2):
        if i not in node_mapping:
            node_mapping[i] = i + n1 - visit
            if not isinstance(x2, tuple):
                x.append(x2[i])
            else:
                x_pos.append(x2[0][i])
                x_attr.append(x2[1][i])
        else:
            visit += 1
            if not isinstance(x2, tuple):
                x[node_mapping[i]] = torch.max(x1[node_mapping[i]], x2[i])
            else:
                x_pos[node_mapping[i]] = torch.max(x1[0][node_mapping[i]], x2[0][i])
                x_attr[node_mapping[i]] = torch.max(x1[1][node_mapping[i]], x2[1][i])
    if not isinstance(x1, tuple) and not isinstance(x2, tuple):
        x = torch.stack(x)
    else:
        x = (torch.stack(x_pos), torch.stack(x_attr))

    ########################################################
    # merge edges
    edge_index1 = np.copy(graph1.edge_index.detach().cpu().numpy().T)
    edge_index2 = np.copy(graph2.edge_index.detach().cpu().numpy().T)
    for i in range(len(edge_index2)):
        edge_index2[i][0] = node_mapping[edge_index2[i][0]]
        edge_index2[i][1] = node_mapping[edge_index2[i][1]]
    edge_index = np.vstack([edge_index1, edge_index2])
    edge_types = np.concatenate([np.zeros(len(edge_index1)), np.ones(len(edge_index2))]).astype(np.int64)

    node_map = np.zeros(n2, dtype=np.int64)
    for k, v in node_mapping.items():
        node_map[k] = v

    # return edge_index, edge_types, x, node_map
    return torch.from_numpy(edge_index), torch.from_numpy(edge_types), x, torch.from_numpy(node_map)
