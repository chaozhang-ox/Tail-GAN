"""
Neural Sorting
"""
import torch

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def deterministic_NeuralSort(s, tau):
    """
    s: input elements to be sorted. Shape: batch_size x n x 1
    tau: temperature for relaxation. Scalar.
    """
    n = s.size()[1]
    one = torch.ones((n, 1)).type(Tensor)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))
    scaling = (n + 1 - 2 * (torch.arange(n) + 1)).type(Tensor)
    C = torch.matmul(s, scaling.unsqueeze(0))
    P_max = (C - B).permute(0, 2, 1)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat


def sample_gumbel(samples_shape, eps = 1e-10):
    U = torch.rand(samples_shape)
    rt = -torch.log(-torch.log(U + eps) + eps)
    return rt.type(Tensor)


def stochastic_NeuralSort(s, n_samples, tau):
    """
    s: parameters of the PL distribution. Shape: batch_size x n x 1.
    n_samples: number of samples from the PL distribution. Scalar.
    tau: temperature for the relaxation. Scalar.
    """
    batch_size = s.size()[0]
    n = s.size()[1]
    log_s_perturb = torch.log(s) + sample_gumbel([n_samples, batch_size, n, 1])
    log_s_perturb = log_s_perturb.view(n_samples * batch_size, n, 1)
    P_hat = deterministic_NeuralSort(log_s_perturb, tau)
    P_hat = P_hat.view(n_samples, batch_size, n, n)
    return P_hat