from collections import deque
from unittest import TestCase
from unittest.mock import Mock
import numpy as np
import gym
import torch
from torch.utils.data import DataLoader

from pl_bolts.datamodules.experience_source import BaseExperienceSource, ExperienceSource, ExperienceSourceDataset, \
    Experience
from pl_bolts.models.rl.common.agents import Agent


class DummyAgent(Agent):
    def __call__(self, states):
        return [0 for s in states]


class DummyExperienceSource(BaseExperienceSource):
    def __iter__(self):
        yield torch.ones(3)


class TestExperienceSourceDataset(TestCase):

    def train_batch(self):
        return iter([i for i in range(100)])

    def test_iterator(self):
        source = ExperienceSourceDataset(self.train_batch)
        batch_size = 10
        data_loader = DataLoader(source, batch_size=batch_size)

        for idx, batch in enumerate(data_loader):
            self.assertEqual(len(batch), batch_size)
            self.assertEqual(batch[0], 0)
            self.assertEqual(batch[5], 5)
            break


class TestBaseExperienceSource(TestCase):

    def setUp(self) -> None:
        self.net = Mock()
        self.agent = DummyAgent(net=self.net)
        self.env = gym.make("CartPole-v0")
        self.device = torch.device('cpu')
        self.source = DummyExperienceSource(self.env, self.agent)

    def test_base_class(self):
        """Tests that base class is initialized correctly"""
        self.assertTrue(isinstance(self.source.env, gym.Env))
        self.assertTrue(isinstance(self.source.agent, Agent))
        out = next(iter(self.source))
        self.assertTrue(torch.all(out.eq(torch.ones(3))))


class TestExperienceSource(TestCase):

    def setUp(self) -> None:
        self.net = Mock()
        self.agent = DummyAgent(net=self.net)
        self.env = [gym.make("CartPole-v0") for _ in range(2)]
        self.device = torch.device('cpu')
        self.source = ExperienceSource(self.env, self.agent)

    def test_init(self):
        """Test that experience source is setup correctly"""
        self.assertEqual(self.source.n_steps, 1)
        self.assertIsInstance(self.source.pool, list)

        self.assertEqual(len(self.source.states), len(self.source.pool))
        self.assertEqual(len(self.source.histories), len(self.source.pool))
        self.assertEqual(len(self.source.cur_rewards), len(self.source.pool))
        self.assertEqual(len(self.source.cur_steps), len(self.source.pool))

    def test_env_actions(self):
        """Assert that a list of actions of shape [num_envs, action_len] is returned"""
        actions = self.source.env_actions()
        self.assertEqual(len(actions), len(self.env))
        self.assertTrue(isinstance(actions[0], list))

    def test_env_step(self):
        """Assert that taking a step through a single environment yields a list of history steps"""
        actions = [[1], [1]]
        env = self.env[0]
        exp = self.source.env_step(0, env, actions[0])

        self.assertTrue(isinstance(exp, Experience))

    def test_source_next_single_env_single_step(self):
        """Test that steps are executed correctly with one environment and 1 step"""

        self.env = [gym.make("CartPole-v0") for _ in range(1)]
        self.device = torch.device('cpu')
        self.source = ExperienceSource(self.env, self.agent, n_steps=1)

        for idx, exp in enumerate(self.source):
            self.assertTrue(isinstance(exp, deque))
            break

    def test_source_next_single_env_multi_step(self):
        """Test that steps are executed correctly with one environment and 2 step"""

        self.env = [gym.make("CartPole-v0") for _ in range(1)]
        self.device = torch.device('cpu')
        n_steps = 4
        self.source = ExperienceSource(self.env, self.agent, n_steps=n_steps)

        for idx, exp in enumerate(self.source):
            self.assertTrue(isinstance(exp, deque))
            self.assertTrue(len(exp) == n_steps)
            break

    def test_source_next_multi_env_single_step(self):
        """Test that steps are executed correctly with 2 environment and 1 step"""

        self.env = [gym.make("CartPole-v0") for _ in range(2)]
        self.device = torch.device('cpu')
        self.source = ExperienceSource(self.env, self.agent, n_steps=1)

        for idx, exp in enumerate(self.source):
            self.assertTrue(isinstance(exp, deque))
            self.assertTrue(len(exp) == self.source.n_steps)
            break

    def test_source_next_multi_env_multi_step(self):
        """Test that steps are executed correctly with 2 environment and 2 step"""

        self.env = [gym.make("CartPole-v0") for _ in range(2)]
        self.device = torch.device('cpu')
        self.source = ExperienceSource(self.env, self.agent, n_steps=2)

        for idx, exp in enumerate(self.source):
            self.assertTrue(isinstance(exp, deque))
            self.assertTrue(len(exp) == self.source.n_steps)
            break

    def test_source_update_state(self):
        """Test that after a step the state is updated"""

        self.env = [gym.make("CartPole-v0") for _ in range(1)]
        self.device = torch.device('cpu')
        self.source = ExperienceSource(self.env, self.agent, n_steps=2)

        for idx, exp in enumerate(self.source):
            self.assertTrue(isinstance(exp, deque))
            new = np.asarray(exp[-1].new_state)
            old = np.asarray(self.source.states[0])
            self.assertTrue(np.array_equal(new, old))
            break






