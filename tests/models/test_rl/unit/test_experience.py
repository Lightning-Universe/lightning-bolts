from unittest import TestCase
from unittest.mock import Mock

import gym
import numpy as np
import torch
from torch.utils.data import DataLoader

from pl_bolts.models.rl.common.agents import Agent
from pl_bolts.models.rl.common.experience import EpisodicExperienceStream, RLDataset, ExperienceSource, \
    NStepExperienceSource
from pl_bolts.models.rl.common.memory import Experience
from pl_bolts.models.rl.common.wrappers import ToTensor


class DummyAgent(Agent):
    def __call__(self, states, agent_states):
        return 0


class TestEpisodicExperience(TestCase):
    """Test the standard experience stream"""

    def setUp(self) -> None:
        self.env = ToTensor(gym.make("CartPole-v0"))
        self.net = Mock()
        self.agent = Agent(self.net)
        self.xp_stream = EpisodicExperienceStream(self.env, self.agent, device=Mock(), episodes=4)
        self.rl_dataloader = DataLoader(self.xp_stream)

    def test_experience_stream_SINGLE_EPISODE(self):
        """Check that the experience stream gives 1 full episode per batch"""
        self.xp_stream.episodes = 1

        for i_batch, batch in enumerate(self.rl_dataloader):
            self.assertEqual(len(batch), 1)
            self.assertIsInstance(batch[0][0], Experience)
            self.assertEqual(batch[0][-1].done, True)

    def test_experience_stream_MULTI_EPISODE(self):
        """Check that the experience stream gives 4 full episodes per batch"""
        self.xp_stream.episodes = 4

        for i_batch, batch in enumerate(self.rl_dataloader):
            self.assertEqual(len(batch), 4)
            self.assertIsInstance(batch[0][0], Experience)
            self.assertEqual(batch[0][-1].done, True)


class TestExperienceSource(TestCase):

    def setUp(self) -> None:
        self.net = Mock()
        self.agent = DummyAgent(net=self.net)
        self.env = gym.make("CartPole-v0")
        self.source = ExperienceSource(self.env, self.agent, Mock())

    def test_step(self):
        exp, reward, done = self.source.step()
        self.assertEqual(len(exp), 5)

    def test_episode(self):
        total_reward = self.source.run_episode()
        self.assertIsInstance(total_reward, float)


class TestNStepExperienceSource(TestCase):

    def setUp(self) -> None:
        self.net = Mock()
        self.agent = DummyAgent(net=self.net)
        self.env = gym.make("CartPole-v0")
        self.n_step = 2
        self.source = NStepExperienceSource(self.env, self.agent, Mock(), n_steps=self.n_step)

        self.state = np.zeros([32, 32])
        self.state_02 = np.ones([32, 32])
        self.next_state = np.zeros([32, 32])
        self.next_state_02 = np.ones([32, 32])
        self.action = np.zeros([1])
        self.action_02 = np.ones([1])
        self.reward = np.zeros([1])
        self.reward_02 = np.ones([1])
        self.done = np.zeros([1])
        self.done_02 = np.zeros([1])

        self.experience01 = Experience(self.state, self.action, self.reward, self.done, self.next_state)
        self.experience02 = Experience(self.state_02, self.action_02, self.reward_02, self.done_02, self.next_state_02)
        self.experience03 = Experience(self.state_02, self.action_02, self.reward_02, self.done_02, self.next_state_02)

    def test_step(self):
        self.assertEqual(len(self.source.n_step_buffer), 0)
        exp, reward, done = self.source.step()
        self.assertEqual(len(exp), 5)
        self.assertEqual(len(self.source.n_step_buffer), self.n_step)

    def test_multi_step(self):
        self.source.env.step = Mock(return_value=(self.next_state_02, self.reward_02, self.done_02, Mock()))
        self.source.n_step_buffer.append(self.experience01)
        self.source.n_step_buffer.append(self.experience01)

        exp, reward, done = self.source.step()

        next_state = exp[4]
        self.assertEqual(next_state.all(), self.next_state_02.all())

    def test_discounted_transition(self):
        self.source = NStepExperienceSource(self.env, self.agent, Mock(), n_steps=3)

        self.source.n_step_buffer.append(self.experience01)
        self.source.n_step_buffer.append(self.experience02)
        self.source.n_step_buffer.append(self.experience03)

        reward, next_state, done = self.source.get_transition_info()

        reward_01 = self.experience02.reward + 0.9 * self.experience03.reward * (1 - done)
        reward_gt = self.experience01.reward + 0.9 * reward_01 * (1 - done)

        self.assertEqual(reward, reward_gt)
        self.assertEqual(next_state.all(), self.next_state_02.all())
        self.assertEqual(self.experience03.done, done)

    def test_multi_step_discount(self):
        self.source = NStepExperienceSource(self.env, self.agent, Mock(), n_steps=3)
        self.source.env.step = Mock(return_value=(self.next_state_02, self.reward_02, self.done_02, Mock()))

        self.source.n_step_buffer.append(self.experience01)
        self.source.n_step_buffer.append(self.experience02)

        reward_gt = 1.71

        exp, reward, done = self.source.step()

        self.assertEqual(exp[0].all(), self.experience01.state.all())
        self.assertEqual(exp[1], self.experience01.action)
        self.assertEqual(exp[2], reward_gt)
        self.assertEqual(exp[3], self.experience02.done)
        self.assertEqual(exp[4].all(), self.experience02.new_state.all())


class TestRLDataset(TestCase):

    def setUp(self) -> None:
        mock_states = np.random.rand(32, 4, 84, 84)
        mock_action = np.random.rand(32)
        mock_rewards = np.random.rand(32)
        mock_dones = np.random.rand(32)
        mock_next_states = np.random.rand(32, 4, 84, 84)
        self.sample = mock_states, mock_action, mock_rewards, mock_dones, mock_next_states

        self.buffer = Mock()
        self.buffer.sample = Mock(return_value=self.sample)
        self.dataset = RLDataset(buffer=self.buffer, sample_size=32)
        self.dl = DataLoader(self.dataset, batch_size=32)

    def test_rl_dataset_batch(self):
        """test that the dataset gives the correct batch"""

        for i_batch, sample_batched in enumerate(self.dl):
            self.assertIsInstance(sample_batched, list)
            self.assertEqual(sample_batched[0].shape, torch.Size([32, 4, 84, 84]))
            self.assertEqual(sample_batched[1].shape, torch.Size([32]))
            self.assertEqual(sample_batched[2].shape, torch.Size([32]))
            self.assertEqual(sample_batched[3].shape, torch.Size([32]))
            self.assertEqual(sample_batched[4].shape, torch.Size([32, 4, 84, 84]))
