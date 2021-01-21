"""Series of networks used
Based on implementations found here:
"""
import math
from typing import Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F


class CNN(nn.Module):
    """
    Simple MLP network
    """

    def __init__(self, input_shape: Tuple[int], n_actions: int):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
        """
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.head = nn.Sequential(nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def _get_conv_out(self, shape) -> int:
        """
        Calculates the output size of the last conv layer

        Args:
            shape: input dimensions
        Returns:
            size of the conv output
        """
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))

    def forward(self, input_x) -> Tensor:
        """
        Forward pass through network

        Args:
            x: input to network
        Returns:
            output of network
        """
        conv_out = self.conv(input_x).view(input_x.size()[0], -1)
        return self.head(conv_out)


class MLP(nn.Module):
    """
    Simple MLP network
    """

    def __init__(self, input_shape: Tuple[int], n_actions: int, hidden_size: int = 128):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, input_x):
        """
        Forward pass through network

        Args:
            x: input to network

        Returns:
            output of network
        """
        return self.net(input_x.float())


class DuelingMLP(nn.Module):
    """
    MLP network with duel heads for val and advantage
    """

    def __init__(self, input_shape: Tuple[int], n_actions: int, hidden_size: int = 128):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super(DuelingMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_shape[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.head_adv = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.head_val = nn.Sequential(nn.Linear(hidden_size, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, input_x):
        """
        Forward pass through network. Calculates the Q using the value and advantage

        Args:
            x: input to network

        Returns:
            Q value
        """
        adv, val = self.adv_val(input_x)
        q_val = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_val

    def adv_val(self, input_x) -> Tuple[Tensor, Tensor]:
        """
        Gets the advantage and value by passing out of the base network through the
        value and advantage heads

        Args:
            input_x: input to network

        Returns:
            advantage, value
        """
        float_x = input_x.float()
        base_out = self.net(float_x)
        return self.fc_adv(base_out), self.fc_val(base_out)


class DuelingCNN(nn.Module):
    """
    CNN network with duel heads for val and advantage
    """

    def __init__(self, input_shape: Tuple[int], n_actions: int, _: int = 128):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        # advantage head
        self.head_adv = nn.Sequential(nn.Linear(conv_out_size, 256), nn.ReLU(), nn.Linear(256, n_actions))

        # value head
        self.head_val = nn.Sequential(nn.Linear(conv_out_size, 256), nn.ReLU(), nn.Linear(256, 1))

    def _get_conv_out(self, shape) -> int:
        """
        Calculates the output size of the last conv layer

        Args:
            shape: input dimensions

        Returns:
            size of the conv output
        """
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))

    def forward(self, input_x):
        """
        Forward pass through network. Calculates the Q using the value and advantage

        Args:
            input_x: input to network

        Returns:
            Q value
        """
        adv, val = self.adv_val(input_x)
        q_val = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_val

    def adv_val(self, input_x):
        """
        Gets the advantage and value by passing out of the base network through the
        value and advantage heads

        Args:
            input_x: input to network

        Returns:
            advantage, value
        """
        float_x = input_x.float()
        base_out = self.conv(input_x).view(float_x.size()[0], -1)
        return self.head_adv(base_out), self.head_val(base_out)


class NoisyCNN(nn.Module):
    """
    CNN with Noisy Linear layers for exploration
    """

    def __init__(self, input_shape: Tuple[int], n_actions: int):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.head = nn.Sequential(NoisyLinear(conv_out_size, 512), nn.ReLU(), NoisyLinear(512, n_actions))

    def _get_conv_out(self, shape) -> int:
        """
        Calculates the output size of the last conv layer

        Args:
            shape: input dimensions

        Returns:
            size of the conv output
        """
        conv_out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(conv_out.size()))

    def forward(self, input_x) -> Tensor:
        """
        Forward pass through network

        Args:
            x: input to network

        Returns:
            output of network
        """
        conv_out = self.conv(input_x).view(input_x.size()[0], -1)
        return self.head(conv_out)


###################
#  Custom Layers  #
###################


class NoisyLinear(nn.Linear):
    """
    Noisy Layer using Independent Gaussian Noise.
    based on https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/
    Chapter08/lib/dqn_extra.py#L19
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017, bias: bool = True):
        """
        Args:
            in_features: number of inputs
            out_features: number of outputs
            sigma_init: initial fill value of noisy weights
            bias: flag to include bias to linear layer
        """
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)

        weights = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(weights)
        epsilon_weight = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", epsilon_weight)

        if bias:
            bias = torch.full((out_features, ), sigma_init)
            self.sigma_bias = nn.Parameter(bias)
            epsilon_bias = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", epsilon_bias)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """initializes or resets the paramseter of the layer"""
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input_x: Tensor) -> Tensor:
        """
        Forward pass of the layer

        Args:
            input_x: input tensor

        Returns:
            output of the layer
        """
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data

        noisy_weights = self.sigma_weight * self.epsilon_weight.data + self.weight

        return F.linear(input_x, noisy_weights, bias)
