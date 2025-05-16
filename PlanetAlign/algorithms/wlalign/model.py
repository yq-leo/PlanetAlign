import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import to_dense_adj


def aggregate_label(x, graph, num_layers=1, device='cpu'):
    label_list = []
    adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).squeeze().to(x.dtype).to(device)
    for i in range(num_layers):
        one_layer_label = adj @ x
        # Threshold the output at 0 and convert to a binary (float) tensor
        one_layer_label = (one_layer_label > 0).float()
        label_list.append(one_layer_label - x)
        x = one_layer_label
    return label_list


class EmbeddingModel(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed

        self.self_embed = nn.Embedding(n_vocab, n_embed)
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)

        self.self_embed.weight.data.uniform_(-1, 1)
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors

    def forward_output(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors

    def forward_self(self, words):
        self_vectors = self.self_embed(words)
        return self_vectors

    def forward_noise(self, size, N_SAMPLES, noise_dist):
        noise_words = torch.multinomial(noise_dist, size * N_SAMPLES, replacement=True)
        noise_vectors_self = self.self_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        noise_vectors_in = self.in_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        noise_vectors_out = self.out_embed(noise_words).view(size, N_SAMPLES, self.n_embed)
        return noise_vectors_self, noise_vectors_in, noise_vectors_out


class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super(NegativeSamplingLoss, self).__init__()

    @staticmethod
    def forward(self_in_vectors, self_out_vectors, target_input_vectors, source_output_vectors,
                noise_vectors_self, noise_vectors_input, noise_vectors_output,
                noise_vectors_self2, noise_vectors_input2, noise_vectors_output2):
        BATCH_SIZE, embed_size = target_input_vectors.shape

        self_in_vectors_T = self_in_vectors.view(BATCH_SIZE, embed_size, 1)
        input_vectors_I = target_input_vectors.view(BATCH_SIZE, 1, embed_size)
        output_vectors_I = source_output_vectors.view(BATCH_SIZE, 1, embed_size)
        self_out_vectors_T = self_out_vectors.view(BATCH_SIZE, embed_size, 1)

        input_vectors_T = target_input_vectors.view(BATCH_SIZE, embed_size, 1)
        output_vectors_T = source_output_vectors.view(BATCH_SIZE, embed_size, 1)

        p1_loss = torch.bmm(input_vectors_I, self_in_vectors_T).sigmoid().log()
        p2_loss = torch.bmm(output_vectors_I, self_out_vectors_T).sigmoid().log()

        p1_noise_loss = torch.bmm(noise_vectors_input.neg(), self_in_vectors_T).sigmoid().log()
        p2_nose_loss = torch.bmm(noise_vectors_self.neg(), output_vectors_T).sigmoid().log()
        p1_noise_loss = p1_noise_loss.squeeze().sum(1)
        p2_nose_loss = p2_nose_loss.squeeze().sum(1)

        p1_noise_loss_ex = torch.bmm(noise_vectors_self.neg(), input_vectors_T).sigmoid().log()
        p2_nose_loss_ex = torch.bmm(noise_vectors_output.neg(), self_out_vectors_T).sigmoid().log()
        p1_noise_loss_ex = p1_noise_loss_ex.squeeze().sum(1)
        p2_nose_loss_ex = p2_nose_loss_ex.squeeze().sum(1)

        p1_noise_loss2 = torch.bmm(noise_vectors_input2.neg(), self_in_vectors_T).sigmoid().log()
        p2_nose_loss2 = torch.bmm(noise_vectors_self2.neg(), output_vectors_T).sigmoid().log()
        p1_noise_loss2 = p1_noise_loss2.squeeze().sum(1)
        p2_nose_loss2 = p2_nose_loss2.squeeze().sum(1)

        p1_noise_loss_ex2 = torch.bmm(noise_vectors_self2.neg(), input_vectors_T).sigmoid().log()
        p2_nose_loss_ex2 = torch.bmm(noise_vectors_output2.neg(), self_out_vectors_T).sigmoid().log()
        p1_noise_loss_ex2 = p1_noise_loss_ex2.squeeze().sum(1)
        p2_nose_loss_ex2 = p2_nose_loss_ex2.squeeze().sum(1)

        return -(p1_loss + p2_loss + p1_noise_loss + p2_nose_loss + p1_noise_loss_ex + p2_nose_loss_ex +
                 p1_noise_loss2 + p2_nose_loss2 + p1_noise_loss_ex2 + p2_nose_loss_ex2).mean()
