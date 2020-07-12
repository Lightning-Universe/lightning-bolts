import argparse
from unittest import TestCase
from unittest.mock import Mock

import gym
import numpy as np
import torch

from pl_bolts.models.rl.common import cli
from pl_bolts.models.rl.common.agents import Agent
from pl_bolts.models.rl.common.memory import Experience
from pl_bolts.models.rl.common.networks import MLP
from pl_bolts.models.rl.common.wrappers import ToTensor
from pl_bolts.models.rl.vanilla_policy_gradient_model import PolicyGradient


class TestPolicyGradient(TestCase):

    def setUp(self) -> None:
        self.env = ToTensor(gym.make("CartPole-v0"))
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.net = MLP(self.obs_shape, self.n_actions)
        self.agent = Agent(self.net)

        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser = cli.add_base_args(parent=parent_parser)
        parent_parser = PolicyGradient.add_model_specific_args(parent_parser)
        args_list = [
            "--episode_length", "100",
            "--env", "CartPole-v0",
        ]
        self.hparams = parent_parser.parse_args(args_list)
        self.model = PolicyGradient(**vars(self.hparams))

    def test_calc_q_vals(self):
        rewards = [torch.tensor(1), torch.tensor(1), torch.tensor(1), torch.tensor(1)]
        gt_qvals = np.array([1.4652743, 0.49497533, -0.4851246, -1.4751246])

        qvals = self.model.calc_qvals(rewards)
        qvals = torch.stack(qvals).numpy()

        self.assertEqual(gt_qvals.all(), qvals.all())

    def test_loss(self):
        """Test the PolicyGradient loss function"""
        self.model.net = self.net
        self.model.agent = self.agent
        self.model.logger = Mock()
        xp_dataloader = self.model.train_dataloader()

        for i_batch, batch in enumerate(xp_dataloader):
            states, actions, scales = batch

            loss = self.model.loss(scales, states, actions)

            self.assertIsInstance(loss, torch.Tensor)
            break

    def test_train_batch(self):
        state = np.random.rand(4, 84, 84)
        self.source = Mock()
        exp = Experience(state=state, action=0, reward=5, done=False, new_state=state)
        self.source.step = Mock(return_value=(exp, 1, False))
        self.model.source = self.source

        xp_dataloader = self.model.train_dataloader()

        for i_batch, batch in enumerate(xp_dataloader):
            self.assertEqual(len(batch), 3)
            self.assertEqual(len(batch[0]), self.model.batch_size)
            self.assertTrue(isinstance(batch, list))
            self.assertEqual(self.model.baseline, 5)
            self.assertIsInstance(batch[0], torch.Tensor)
            self.assertIsInstance(batch[1], torch.Tensor)
            self.assertIsInstance(batch[2], torch.Tensor)
            break
