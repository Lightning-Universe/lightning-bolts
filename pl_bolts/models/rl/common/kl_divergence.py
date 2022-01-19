from typing import Tuple

import torch
from torch import Tensor


def kl_divergence_between_discrete_distributions(p: Tensor, q: Tensor) -> float:
    """
    Calculates mean KL divergence between two tensors of distributions. Which is represented by equation:
    KL(p||q): sum p(x) log(p(x)/q(x))

    Args:
        p: Two dimensional vector of probability distributions: [NUM_SAMPLES, NUM_ACTIONS]
        q: Two dimensional vector of probability distributions: [NUM_SAMPLES, NUM_ACTIONS]

    Returns:
        Mean KL divergence
    """
    p.detach()
    p, q = p.squeeze(), q.squeeze()
    kl = (p * (torch.log(p) - torch.log(q))).sum(-1)
    return kl.mean()


def kl_divergence_between_continuous_distributions(p: Tuple[Tensor, Tensor], q: Tuple[Tensor, Tensor]) -> float:
    """
    Calculates mean KL divergence between two gaussian distributions. Which is represented by equation:
    KL(p||q): sum p(x) log(p(x)/q(x))
    The solution comes from https://stats.stackexchange.com/a/60699

    Args:
        p: Tuple of two tensors, first of shape [NUM_SAMPLES, NUM_ACTIONS] which represents means, while second one
            represents standard deviations and has shape [NUM_ACTIONS]
        q: Tuple of two tensors, first of shape [NUM_SAMPLES, NUM_ACTIONS] which represents means, while second one
            represents standard deviations and has shape [NUM_ACTIONS]

    Returns:
        Mean KL divergence
    """
    p_mean, p_std = p
    q_mean, q_std = q

    p_mean.detach()
    p_std.detach()

    p_var, q_var = p_std.pow(2), q_std.pow(2)

    d = q_mean.shape[1]
    diff = q_mean - p_mean

    log_quot_frac = torch.log(q_var).sum() - torch.log(p_var).sum()
    tr = (p_var / q_var).sum()
    quadratic = ((diff / q_var) * diff).sum(dim=1)

    kl_sum = 0.5 * (log_quot_frac - d + tr + quadratic)
    assert kl_sum.shape == (p_mean.shape[0],)
    return kl_sum.mean()
