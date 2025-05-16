from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class ParamFreeGraphConv(MessagePassing):
    def __init__(self):
        super(ParamFreeGraphConv, self).__init__(aggr='add')  # using sum aggregation

    def forward(self, x, edge_index):
        # Add self-loops to mimic dgl.add_self_loop()
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])

        # Compute symmetric normalization (norm='both')
        row, col = edge_index
        deg = degree(row, x.shape[0], dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Propagate messages without applying any weight
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm=None):
        if norm is not None:
            return norm.view(-1, 1) * x_j
        return x_j

    def edge_update(self):
        pass

    def message_and_aggregate(self, edge_index):
        pass