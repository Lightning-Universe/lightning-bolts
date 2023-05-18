"""Distributions used in some continuous RL algorithms."""
import torch

from pl_bolts.utils.stability import under_review


@under_review()
class TanhMultivariateNormal(torch.distributions.MultivariateNormal):
    """The distribution of X is an affine of tanh applied on a normal distribution.

    X = action_scale * tanh(Z) + action_bias
    Z ~ Normal(mean, variance)
    """

    def __init__(self, action_bias, action_scale, **kwargs):
        super().__init__(**kwargs)

        self.action_bias = action_bias
        self.action_scale = action_scale

    def rsample_with_z(self, sample_shape=torch.Size()):
        """Samples X using reparametrization trick with the intermediate variable Z.

        Returns:
            Sampled X and Z
        """
        z = super().rsample()
        return self.action_scale * torch.tanh(z) + self.action_bias, z

    def log_prob_with_z(self, value, z):
        """Computes the log probability of a sampled X.

        Refer to the original paper of SAC for more details in equation (20), (21)

        Args:
            value: the value of X
            z: the value of Z
        Returns:
            Log probability of the sample
        """
        value = (value - self.action_bias) / self.action_scale
        z_logprob = super().log_prob(z)
        correction = torch.log(self.action_scale * (1 - value**2) + 1e-7).sum(1)
        return z_logprob - correction

    def rsample_and_log_prob(self, sample_shape=torch.Size()):
        """Samples X and computes the log probability of the sample.

        Returns:
            Sampled X and log probability
        """
        z = super().rsample()
        z_logprob = super().log_prob(z)
        value = torch.tanh(z)
        correction = torch.log(self.action_scale * (1 - value**2) + 1e-7).sum(1)
        return self.action_scale * value + self.action_bias, z_logprob - correction

    def rsample(self, sample_shape=torch.Size()):
        fz, z = self.rsample_with_z(sample_shape)
        return fz

    def log_prob(self, value):
        value = (value - self.action_bias) / self.action_scale
        z = torch.log(1 + value) / 2 - torch.log(1 - value) / 2
        return self.log_prob_with_z(value, z)
