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
