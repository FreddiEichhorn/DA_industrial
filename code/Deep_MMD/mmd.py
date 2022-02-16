import torch


def mmd3(xs, xt, kernel='linear', sigma=1):

    def linear_kernel(a, b):
        helper = a @ b.T
        return (-torch.diag(a @ a.T).expand_as(helper).T + 2 * helper - torch.diag(b @ b.T).expand_as(helper)) / sigma

    def rbf_kernel(a, b):
        return torch.exp(linear_kernel(a, b) / sigma)

    if kernel == 'linear':
        k1 = linear_kernel(xs, xs)
        k2 = linear_kernel(xs, xt)
        k3 = linear_kernel(xt, xt)
    elif kernel == 'rbf':
        k1 = rbf_kernel(xs, xs)
        k2 = rbf_kernel(xs, xt)
        k3 = rbf_kernel(xt, xt)
    else:
        k1 = None
        k2 = None
        k3 = None

    return torch.mean(k1 - 2 * k2 + k3)
