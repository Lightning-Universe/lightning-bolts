"""Test RL Loss Functions."""

from unittest import TestCase

import numpy as np
import torch
from torch import Tensor

from pl_bolts.losses.rl import double_dqn_loss, dqn_loss, per_dqn_loss
from pl_bolts.models.rl.common.gym_wrappers import make_environment
from pl_bolts.models.rl.common.networks import CNN


class TestRLLoss(TestCase):
    def setUp(self) -> None:

        self.state = torch.rand(32, 4, 84, 84)
        self.next_state = torch.rand(32, 4, 84, 84)
        self.action = torch.ones([32])
        self.reward = torch.ones([32])
        self.done = torch.zeros([32]).long()

        self.batch = (self.state, self.action, self.reward, self.done, self.next_state)

        self.env = make_environment("PongNoFrameskip-v4")
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.net = CNN(self.obs_shape, self.n_actions)
        self.target_net = CNN(self.obs_shape, self.n_actions)

    def test_dqn_loss(self):
        """Test the dqn loss function."""

        loss = dqn_loss(self.batch, self.net, self.target_net)
        self.assertIsInstance(loss, Tensor)

    def test_double_dqn_loss(self):
        """Test the double dqn loss function."""

        loss = double_dqn_loss(self.batch, self.net, self.target_net)
        self.assertIsInstance(loss, Tensor)

    def test_per_dqn_loss(self):
        """Test the double dqn loss function."""
        prios = torch.ones([32])

        loss, batch_weights = per_dqn_loss(self.batch, prios, self.net, self.target_net)
        self.assertIsInstance(loss, Tensor)
        self.assertIsInstance(batch_weights, np.ndarray)
