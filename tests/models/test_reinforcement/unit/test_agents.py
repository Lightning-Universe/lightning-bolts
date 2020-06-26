"""Tests that the agent module works correctly"""
from unittest import TestCase
from unittest.mock import Mock

import gym
import torch

from pl_bolts.models.rl.common.agents import Agent, PolicyAgent, ValueAgent


class TestAgents(TestCase):

    def setUp(self) -> None:
        self.env = gym.make("CartPole-v0")
        self.state = self.env.reset()
        self.net = Mock()

    def test_base_agent(self):
        agent = Agent(self.net)
        action = agent(self.state, 'cuda:0')
        self.assertIsInstance(action, int)


class TestValueAgent(TestCase):

    def setUp(self) -> None:
        self.env = gym.make("CartPole-v0")
        self.net = Mock(return_value=torch.Tensor([[0.0, 100.0]]))
        self.state = torch.tensor(self.env.reset())
        self.device = self.state.device
        self.value_agent = ValueAgent(self.net, self.env.action_space.n)

    def test_value_agent(self):

        action = self.value_agent(self.state, self.device)
        self.assertIsInstance(action, int)

    def test_value_agent_GET_ACTION(self):
        action = self.value_agent.get_action(self.state, self.device)
        self.assertIsInstance(action, int)
        self.assertEqual(action, 1)

    def test_value_agent_RANDOM(self):
        action = self.value_agent.get_random_action()
        self.assertIsInstance(action, int)


class TestPolicyAgent(TestCase):

    def setUp(self) -> None:
        self.env = gym.make("CartPole-v0")
        self.net = Mock(return_value=torch.Tensor([0.0, 100.0]))
        self.state = torch.tensor(self.env.reset())
        self.device = self.state.device

    def test_policy_agent(self):
        policy_agent = PolicyAgent(self.net)
        action = policy_agent(self.state, self.device)
        self.assertIsInstance(action, int)
        self.assertEqual(action, 1)
