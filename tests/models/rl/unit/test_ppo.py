import argparse
from unittest import TestCase

import numpy as np
import torch

from pl_bolts.models.rl.ppo_model import PPO


class TestPPOCategorical(TestCase):

    def setUp(self) -> None:

        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser = PPO.add_model_specific_args(parent_parser)
        args_list = [
            "--env",
            "CartPole-v0",
            "--batch_size",
            "16",
            "--gamma",
            "0.99",
        ]
        self.hparams = parent_parser.parse_args(args_list)

        self.model = PPO(**vars(self.hparams))
        self.obs_dim = self.model.env.observation_space.shape[0]
        self.rl_dataloader = self.model.train_dataloader()

    def test_actor_loss(self):
        """Test the actor loss function"""

        batch_states = torch.rand(8, self.obs_dim)
        batch_actions = torch.rand(8).long()
        batch_logp_old = torch.rand(8)
        batch_adv = torch.rand(8)

        loss = self.model.actor_loss(batch_states, batch_actions, batch_logp_old, batch_adv)

        self.assertIsInstance(loss, torch.Tensor)

    def test_critic_loss(self):
        """Test the critic loss function"""

        batch_states = torch.rand(8, self.obs_dim)
        batch_qvals = torch.rand(8)

        loss = self.model.critic_loss(batch_states, batch_qvals)

        self.assertIsInstance(loss, torch.Tensor)

    def test_discount_rewards(self):
        rewards = np.ones(4)
        gt_qvals = [3.9403989999999998, 2.9701, 1.99, 1.0]

        qvals = self.model.discount_rewards(rewards, discount=0.99)

        self.assertEqual(gt_qvals, qvals)


class TestPPOContinous(TestCase):

    def setUp(self) -> None:

        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser = PPO.add_model_specific_args(parent_parser)
        args_list = [
            "--env",
            "MountainCarContinuous-v0",
            "--batch_size",
            "16",
            "--gamma",
            "0.99",
        ]
        self.hparams = parent_parser.parse_args(args_list)

        self.model = PPO(**vars(self.hparams))
        self.obs_dim = self.model.env.observation_space.shape[0]
        self.action_dim = self.model.env.action_space.shape[0]

    def test_actor_loss(self):
        """Test the actor loss function"""

        batch_states = torch.rand(8, self.obs_dim)
        batch_actions = torch.rand(8, self.action_dim)
        batch_logp_old = torch.rand(8)
        batch_adv = torch.rand(8)

        loss = self.model.actor_loss(batch_states, batch_actions, batch_logp_old, batch_adv)

        self.assertIsInstance(loss, torch.Tensor)
