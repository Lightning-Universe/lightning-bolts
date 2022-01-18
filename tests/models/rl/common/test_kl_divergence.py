import torch

from pl_bolts.models.rl.common.kl_divergence import (
    kl_divergence_between_discrete_distributions,
    kl_divergence_between_continuous_distributions,
)


def test_discrete_kl_div_when_distributions_are_same():
    a = torch.tensor([[0.1, 0.5, 0.2, 0.2]])
    b = torch.tensor([[0.1, 0.5, 0.2, 0.2]])
    assert kl_divergence_between_discrete_distributions(a, b) == 0


def test_continuous_kl_div_when_distributions_are_same():
    means = torch.tensor([[0.1], [0.5]])
    stds = torch.tensor([0.1])
    assert kl_divergence_between_continuous_distributions((means, stds), (means, stds)) == 0
