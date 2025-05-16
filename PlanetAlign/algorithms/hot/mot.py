import torch


def compute_p_i(log_T):
    num_dists = len(log_T.shape)
    p_i = {}
    for i in range(num_dists):
        tmp_s = list(range(num_dists))
        tmp_s.remove(i)
        p_i[f'{i}'] = torch.logsumexp(log_T, dim=tuple(tmp_s))
    return p_i


def compute_p_ij(log_T):
    num_dists = len(log_T.shape)
    if num_dists == 2:
        return {'0,1': log_T}

    p_ij = {}
    for i in range(num_dists):
        tmp_s = list(range(num_dists))
        tmp_s.remove(i)
        for j in range(i + 1, num_dists):
            tmp_ss = tmp_s.copy()
            tmp_ss.remove(j)
            p_ij[f'{i},{j}'] = torch.logsumexp(log_T, dim=tuple(tmp_ss))
    return p_ij


def compute_l(A, p_i, p_ij, dtype, device='cpu'):
    num_dists = len(A)
    s = []
    for i in range(num_dists):
        s.append(len(A[i]))
    L = torch.zeros(s, dtype=dtype).to(device)
    for i in range(num_dists):
        tmp_s = [1] * num_dists
        tmp_s[i] = s[i]
        L += (num_dists-1) * ((A[i]**2) @ torch.exp(torch.clamp(p_i[f'{i}'], min=-torch.inf, max=70))).reshape(tmp_s)
        for j in range(i+1, num_dists):
            tmp_ss = tmp_s.copy()
            tmp_ss[j] = s[j]
            L += -2 * (A[i] @ torch.exp(torch.clamp(p_ij[f'{i},{j}'], min=-torch.inf, max=70)) @ A[j].T).reshape(tmp_ss)
    return L
