import numpy as np


class AliasSampler:
    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U.tolist()):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sample(self, num_samples=1):
        x = np.random.rand(num_samples)
        i = np.floor(self.n * x).astype(np.int32)
        y = self.n * x - i
        samples = np.array([i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(num_samples)])
        return samples