from unittest import TestCase
from unittest.mock import Mock

import gym
import numpy as np
import torch
from torch.utils.data import DataLoader

from pl_bolts.datamodules.experience_source import (
    BaseExperienceSource,
    DiscountedExperienceSource,
    Experience,
    ExperienceSource,
    ExperienceSourceDataset,
)
from pl_bolts.models.rl.common.agents import Agent


class DummyAgent(Agent):
    def __call__(self, states, device):
        return [0] * len(states)


class DummyExperienceSource(BaseExperienceSource):
    def __iter__(self):
        yield torch.ones(3)


class TestExperienceSourceDataset(TestCase):
    def train_batch(self):
        """Returns an iterator used for testing."""
        return iter([i for i in range(100)])

    def test_iterator(self):
        """Tests that the iterator returns batches correctly."""
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.source = DummyExperienceSource(self.env, self.agent)
        self.s1 = torch.ones(3)
        self.s2 = torch.zeros(3)

    def test_dummy_base_class(self):
        """Tests that base class is initialized correctly."""
        self.assertTrue(isinstance(self.source.env, gym.Env))
        self.assertTrue(isinstance(self.source.agent, Agent))
        out = next(iter(self.source))
        self.assertTrue(torch.all(out.eq(torch.ones(3))))


class TestExperienceSource(TestCase):
    def setUp(self) -> None:
        self.net = Mock()
        self.agent = DummyAgent(net=self.net)
        self.env = [gym.make("CartPole-v0") for _ in range(2)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.source = ExperienceSource(self.env, self.agent, n_steps=1)

        self.s1 = torch.ones(3)
        self.s2 = torch.zeros(3)

        self.mock_env = Mock()
        self.mock_env.step = Mock(return_value=(self.s1, 1, False, Mock()))

        self.exp1 = Experience(state=self.s1, action=1, reward=1, done=False, new_state=self.s2)
        self.exp2 = Experience(state=self.s1, action=1, reward=1, done=False, new_state=self.s2)

    def test_init_source(self):
        """Test that experience source is setup correctly."""
        self.assertEqual(self.source.n_steps, 1)
        self.assertIsInstance(self.source.pool, list)

        self.assertEqual(len(self.source.states), len(self.source.pool))
        self.assertEqual(len(self.source.histories), len(self.source.pool))
        self.assertEqual(len(self.source.cur_rewards), len(self.source.pool))
        self.assertEqual(len(self.source.cur_steps), len(self.source.pool))

    def test_init_single_env(self):
        """Test that if a single env is passed that it is wrapped in a list."""
        self.source = ExperienceSource(self.mock_env, self.agent)
        self.assertIsInstance(self.source.pool, list)

    def test_env_actions(self):
        """Assert that a list of actions of shape [num_envs, action_len] is returned."""
        actions = self.source.env_actions(self.device)
        self.assertEqual(len(actions), len(self.env))
        self.assertTrue(isinstance(actions[0], list))

    def test_env_step(self):
        """Assert that taking a step through a single environment yields a list of history steps."""
        actions = [[1], [1]]
        env = self.env[0]
        exp = self.source.env_step(0, env, actions[0])

        self.assertTrue(isinstance(exp, Experience))

    def test_source_next_single_env_single_step(self):
        """Test that steps are executed correctly with one environment and 1 step."""

        self.env = [gym.make("CartPole-v0") for _ in range(1)]
        self.source = ExperienceSource(self.env, self.agent, n_steps=1)

        for idx, exp in enumerate(self.source.runner(self.device)):
            self.assertTrue(isinstance(exp, tuple))
            break

    def test_source_next_single_env_multi_step(self):
        """Test that steps are executed correctly with one environment and 2 step."""

        self.env = [gym.make("CartPole-v0") for _ in range(1)]
        n_steps = 4
        self.source = ExperienceSource(self.env, self.agent, n_steps=n_steps)

        for idx, exp in enumerate(self.source.runner(self.device)):
            self.assertTrue(isinstance(exp, tuple))
            self.assertTrue(len(exp) == n_steps)
            break

    def test_source_next_multi_env_single_step(self):
        """Test that steps are executed correctly with 2 environment and 1 step."""

        for idx, exp in enumerate(self.source.runner(self.device)):
            self.assertTrue(isinstance(exp, tuple))
            self.assertTrue(len(exp) == self.source.n_steps)
            break

    def test_source_next_multi_env_multi_step(self):
        """Test that steps are executed correctly with 2 environment and 2 step."""
        self.source = ExperienceSource(self.env, self.agent, n_steps=2)

        for idx, exp in enumerate(self.source.runner(self.device)):
            self.assertTrue(isinstance(exp, tuple))
            self.assertTrue(len(exp) == self.source.n_steps)
            break

    def test_source_update_state(self):
        """Test that after a step the state is updated."""

        self.env = [gym.make("CartPole-v0") for _ in range(1)]
        self.source = ExperienceSource(self.env, self.agent, n_steps=2)

        for idx, exp in enumerate(self.source.runner(self.device)):
            self.assertTrue(isinstance(exp, tuple))
            new = np.asarray(exp[-1].new_state)
            old = np.asarray(self.source.states[0])
            self.assertTrue(np.array_equal(new, old))
            break

    def test_source_is_done_short_episode(self):
        """Test that when done and the history is not full, to return the partial history."""

        self.mock_env.step = Mock(return_value=(self.s1, 1, True, Mock))

        env = [self.mock_env for _ in range(1)]
        self.source = ExperienceSource(env, self.agent, n_steps=2)

        for idx, exp in enumerate(self.source.runner(self.device)):
            self.assertTrue(isinstance(exp, tuple))
            self.assertTrue(len(exp) == 1)
            break

    def test_source_is_done_2step_episode(self):
        """Test that when done and the history is full, return the full history, then start to return the tail of
        the history."""

        self.env = [self.mock_env]
        self.source = ExperienceSource(self.env, self.agent, n_steps=2)

        self.mock_env.step = Mock(return_value=(self.s1, 1, True, Mock))

        self.source.histories[0].append(self.exp1)

        for idx, exp in enumerate(self.source.runner(self.device)):

            self.assertTrue(isinstance(exp, tuple))

            if idx == 0:
                self.assertTrue(len(exp) == self.source.n_steps)
            elif idx == 1:
                self.assertTrue(len(exp) == self.source.n_steps - 1)
                self.assertTrue(torch.equal(exp[0].new_state, self.s1))

                break

    def test_source_is_done_metrics(self):
        """Test that when done and the history is full, return the full history."""

        n_steps = 3
        n_envs = 2

        self.mock_env.step = Mock(return_value=(self.s1, 1, True, Mock))

        self.env = [self.mock_env for _ in range(2)]
        self.source = ExperienceSource(self.env, self.agent, n_steps=3)

        history = self.source.histories[0]
        history += [self.exp1, self.exp2, self.exp2]

        for idx, exp in enumerate(self.source.runner(self.device)):

            if idx == n_steps - 1:
                self.assertEqual(self.source._total_rewards[0], 1)
                self.assertEqual(self.source.total_steps[0], 1)
                self.assertEqual(self.source.cur_rewards[0], 0)
                self.assertEqual(self.source.cur_steps[0], 0)
            elif idx == (3 * n_envs) - 1:
                self.assertEqual(self.source.iter_idx, 1)
                break

    def test_pop_total_rewards(self):
        """Test that pop rewards returns correct rewards."""
        self.source._total_rewards = [10, 20, 30]

        rewards = self.source.pop_total_rewards()

        self.assertEqual(rewards, [10, 20, 30])


class TestDiscountedExperienceSource(TestCase):
    def setUp(self) -> None:
        self.net = Mock()
        self.agent = DummyAgent(net=self.net)
        self.env = [gym.make("CartPole-v0") for _ in range(2)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_steps = 3
        self.gamma = 0.9
        self.source = DiscountedExperienceSource(self.env, self.agent, n_steps=self.n_steps, gamma=self.gamma)

        self.state = torch.ones(3)
        self.next_state = torch.zeros(3)
        self.reward = 1

        self.exp1 = Experience(
            state=self.state,
            action=1,
            reward=self.reward,
            done=False,
            new_state=self.next_state,
        )
        self.exp2 = Experience(
            state=self.next_state,
            action=1,
            reward=self.reward,
            done=False,
            new_state=self.state,
        )

        self.env1 = Mock()
        self.env1.step = Mock(return_value=(self.next_state, self.reward, True, self.state))

    def test_init(self):
        """Test that experience source is setup correctly."""
        self.assertEqual(self.source.n_steps, self.n_steps + 1)
        self.assertEqual(self.source.steps, self.n_steps)
        self.assertEqual(self.source.gamma, self.gamma)

    def test_source_step(self):
        """Tests that the source returns a single experience."""

        for idx, exp in enumerate(self.source.runner(self.device)):
            self.assertTrue(isinstance(exp, Experience))
            break

    def test_source_step_done(self):
        """Tests that the source returns a single experience."""

        self.source = DiscountedExperienceSource(self.env1, self.agent, n_steps=self.n_steps)

        self.source.histories[0].append(self.exp1)
        self.source.histories[0].append(self.exp2)

        for idx, exp in enumerate(self.source.runner(self.device)):
            self.assertTrue(isinstance(exp, Experience))
            self.assertTrue(torch.all(torch.eq(exp.new_state, self.next_state)))
            break

    def test_source_discounted_return(self):
        """Tests that the source returns a single experience with discounted rewards.

        discounted returns: G(t) = R(t+1) + γ*R(t+2) + γ^2*R(t+3) ... + γ^N-1*R(t+N)
        """

        self.source = DiscountedExperienceSource(self.env1, self.agent, n_steps=self.n_steps)

        self.source.histories[0] += [self.exp1, self.exp2]

        discounted_reward = (
            self.exp1.reward + (self.source.gamma * self.exp2.reward) + (self.source.gamma * self.reward) ** 2
        )

        for idx, exp in enumerate(self.source.runner(self.device)):
            self.assertTrue(isinstance(exp, Experience))
            self.assertEqual(exp.reward, discounted_reward)
            break
