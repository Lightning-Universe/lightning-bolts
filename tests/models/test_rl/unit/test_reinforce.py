import argparse
from unittest import TestCase
from unittest.mock import Mock

import gym
import torch
from torch.utils.data import DataLoader

from pl_bolts.models.rl.common import cli
from pl_bolts.models.rl.common.agents import Agent
from pl_bolts.models.rl.common.experience import EpisodicExperienceStream
from pl_bolts.models.rl.common.networks import MLP
from pl_bolts.models.rl.common.wrappers import ToTensor
from pl_bolts.models.rl.dqn_model import DQN
from pl_bolts.models.rl.reinforce_model import Reinforce


class TestReinforce(TestCase):

    def setUp(self) -> None:
        self.env = ToTensor(gym.make("CartPole-v0"))
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        self.net = MLP(self.obs_shape, self.n_actions)
        self.agent = Agent(self.net)
        self.xp_stream = EpisodicExperienceStream(self.env, self.agent, Mock(), episodes=4)
        self.rl_dataloader = DataLoader(self.xp_stream)

        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser = cli.add_base_args(parent=parent_parser)
        parent_parser = DQN.add_model_specific_args(parent_parser)
        args_list = [
            "--algo", "dqn",
            "--warm_start_steps", "500",
            "--episode_length", "100",
            "--env", "CartPole-v0",
        ]
        self.hparams = parent_parser.parse_args(args_list)
        self.model = Reinforce(**vars(self.hparams))

    def test_loss(self):
        """Test the reinforce loss function"""
        self.model.net = self.net
        self.model.agent = self.agent

        for i_batch, batch in enumerate(self.rl_dataloader):
            exp_batch = batch

            batch_qvals, batch_states, batch_actions, _ = self.model.process_batch(exp_batch)

            loss = self.model.loss(batch_qvals, batch_states, batch_actions)

            self.assertIsInstance(loss, torch.Tensor)
            break

    def test_get_qvals(self):
        """Test that given an batch of episodes that it will return a list of qvals for each episode"""
        batch_qvals = []
        for i_batch, batch in enumerate(self.rl_dataloader):

            for episode in batch:
                rewards = [step[2] for step in episode]
                batch_qvals.append(self.model.calc_qvals(rewards))

            self.assertEqual(len(batch_qvals), len(batch))
            self.assertIsInstance(batch_qvals[0][0], torch.Tensor)
            self.assertEqual(batch_qvals[0][0], (batch_qvals[0][1] * self.hparams.gamma) + 1.0)
            break

    def test_process_batch(self):
        """Test that given a batch of episodes that it will return the q_vals, the states and the actions"""
        batch_len = 0

        for i_batch, batch in enumerate(self.rl_dataloader):
            for mini_batch in batch:
                batch_len += len(mini_batch)

            q_vals, states, actions, rewards = self.model.process_batch(batch)

            self.assertEqual(len(q_vals), batch_len)
            self.assertEqual(len(states), batch_len)
            self.assertEqual(len(actions), batch_len)
            self.assertEqual(len(rewards), batch_len)

            self.assertEqual(len(q_vals.shape), 1)
            self.assertEqual(len(states.shape), 2)
            self.assertEqual(len(actions.shape), 1)
            self.assertEqual(len(rewards.shape), 1)

    def test_calc_q_vals(self):
        rewards = [1, 1, 1, 1]
        gt_qvals = [3.9403989999999998, 2.9701, 1.99, 1.0]

        qvals = self.model.calc_qvals(rewards)

        self.assertEqual(gt_qvals, qvals)
