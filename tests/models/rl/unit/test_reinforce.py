import argparse
from unittest import TestCase

import gym
import numpy as np
import torch

from pl_bolts.datamodules.experience_source import DiscountedExperienceSource
from pl_bolts.models.rl.common.agents import Agent
from pl_bolts.models.rl.common.gym_wrappers import ToTensor
from pl_bolts.models.rl.common.networks import MLP
from pl_bolts.models.rl.reinforce_model import Reinforce


class TestReinforce(TestCase):

    def setUp(self) -> None:
        self.env = ToTensor(gym.make("CartPole-v0"))
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.net = MLP(self.obs_shape, self.n_actions)
        self.agent = Agent(self.net)
        self.exp_source = DiscountedExperienceSource(self.env, self.agent)

        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser = Reinforce.add_model_specific_args(parent_parser)
        args_list = [
            "--env",
            "CartPole-v0",
            "--batch_size",
            "32",
            "--gamma",
            "0.99",
        ]
        self.hparams = parent_parser.parse_args(args_list)
        self.model = Reinforce(**vars(self.hparams))

        self.rl_dataloader = self.model.train_dataloader()

    def test_loss(self):
        """Test the reinforce loss function"""

        batch_states = torch.rand(16, 4)
        batch_actions = torch.rand(16).long()
        batch_qvals = torch.rand(16)

        loss = self.model.loss(batch_states, batch_actions, batch_qvals)

        self.assertIsInstance(loss, torch.Tensor)

    def test_get_qvals(self):
        """Test that given an batch of episodes that it will return a list of qvals for each episode"""

        batch_qvals = []
        rewards = np.ones(32)
        out = self.model.calc_qvals(rewards)
        batch_qvals.append(out)

        self.assertIsInstance(batch_qvals[0][0], float)
        self.assertEqual(batch_qvals[0][0], (batch_qvals[0][1] * self.hparams.gamma) + 1.0)

    def test_calc_q_vals(self):
        rewards = np.ones(4)
        gt_qvals = [3.9403989999999998, 2.9701, 1.99, 1.0]

        qvals = self.model.calc_qvals(rewards)

        self.assertEqual(gt_qvals, qvals)
