import torch


def sinkhorn_stable(inter_c, intra_c1, intra_c2, threshold_lambda=0, in_iter=5, out_iter=10, gw_weight=20, gamma_p=1e-2,
                    dtype=torch.float32, device='cpu'):
    n1, n2 = inter_c.shape
    # marginal distribution
    a = torch.ones(n1).to(dtype).to(device) / n1
    b = torch.ones(n2).to(dtype).to(device) / n2
    # lagrange multiplier
    f = torch.ones(n1).to(dtype).to(device) / n1
    g = torch.ones(n2).to(dtype).to(device) / n2
    # transport plan
    s = torch.ones((n1, n2)).to(dtype).to(device) / (n1 * n2)

    def soft_min_row(z_in, eps):
        hard_min = torch.min(z_in, dim=1, keepdim=True)[0]
        soft_min = hard_min - eps * torch.log(torch.sum(torch.exp(-(z_in - hard_min) / eps), dim=1, keepdim=True))
        return soft_min.squeeze(-1)

    def soft_min_col(z_in, eps):
        hard_min = torch.min(z_in, dim=0, keepdim=True)[0]
        soft_min = hard_min - eps * torch.log(torch.sum(torch.exp(-(z_in - hard_min) / eps), dim=0, keepdim=True))
        return soft_min.squeeze(0)

    for i in range(out_iter):
        a_hat = torch.sum(s - threshold_lambda, dim=1)
        b_hat = torch.sum(s - threshold_lambda, dim=0)
        temp = (intra_c1 ** 2 @ a_hat.view(-1, 1) @ torch.ones((1, n2)).to(dtype).to(device) +
                torch.ones((n1, 1)).to(dtype).to(device) @ b_hat.view(1, -1) @ intra_c2 ** 2)
        L = temp - 2 * intra_c1 @ (s - threshold_lambda) @ intra_c2.T
        cost = inter_c + gw_weight * L

        Q = cost
        for j in range(in_iter):
            # log-sum-exp stabilization
            f = soft_min_row(Q - g.view(1, -1), gamma_p) + gamma_p * torch.log(a)
            g = soft_min_col(Q - f.view(-1, 1), gamma_p) + gamma_p * torch.log(b)
        s = 0.05 * s + 0.95 * torch.exp((f.view(-1, 1) + g.view(-1, 1).T - Q) / gamma_p)

    return s
