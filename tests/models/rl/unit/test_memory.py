from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import torch

from pl_bolts.models.rl.common.memory import ReplayBuffer, Experience, PERBuffer, Buffer


class TestBuffer(TestCase):

    def train_batch(self):
        """Returns an iterator used for testing"""
        return iter([i for i in range(100)])

    def setUp(self) -> None:
        self.state = np.random.rand(4, 84, 84)
        self.next_state = np.random.rand(4, 84, 84)
        self.action = np.ones([1])
        self.reward = np.ones([1])
        self.done = np.zeros([1])
        self.experience = Experience(self.state, self.action, self.reward, self.done, self.next_state)
        self.source = Mock()
        self.source.step = Mock(return_value=(self.experience, torch.tensor(0), False))
        self.batch_size = 8
        self.buffer = Buffer(8)

        for _ in range(self.batch_size):
            self.buffer.append(self.experience)

    def test_sample_batch(self):
        """check that a sinlge sample is returned"""
        sample = self.buffer.sample()
        self.assertEqual(len(sample), 5)
        self.assertEqual(sample[0].shape, (self.batch_size, 4, 84, 84))
        self.assertEqual(sample[1].shape, (self.batch_size, 1))
        self.assertEqual(sample[2].shape, (self.batch_size, 1))
        self.assertEqual(sample[3].shape, (self.batch_size, 1))
        self.assertEqual(sample[4].shape, (self.batch_size, 4, 84, 84))


class TestReplayBuffer(TestCase):

    def setUp(self) -> None:
        self.state = np.random.rand(32, 32)
        self.next_state = np.random.rand(32, 32)
        self.action = np.ones([1])
        self.reward = np.ones([1])
        self.done = np.zeros([1])
        self.experience = Experience(self.state, self.action, self.reward, self.done, self.next_state)

        self.source = Mock()
        self.source.step = Mock(return_value=(self.experience, torch.tensor(0), False))
        self.warm_start = 10
        self.buffer = ReplayBuffer(20)
        for _ in range(self.warm_start):
            self.buffer.append(self.experience)

    def test_replay_buffer_append(self):
        """Test that you can append to the replay buffer"""

        self.assertEqual(len(self.buffer), self.warm_start)

        self.buffer.append(self.experience)

        self.assertEqual(len(self.buffer), self.warm_start + 1)

    def test_replay_buffer_populate(self):
        """Tests that the buffer is populated correctly with warm_start"""
        self.assertEqual(len(self.buffer.buffer), self.warm_start)

    def test_replay_buffer_update(self):
        """Tests that buffer append works correctly"""
        batch_size = 3
        self.assertEqual(len(self.buffer.buffer), self.warm_start)
        for i in range(batch_size):
            self.buffer.append(self.experience)
        self.assertEqual(len(self.buffer.buffer), self.warm_start + batch_size)

    def test_replay_buffer_sample(self):
        """Test that you can sample from the buffer and the outputs are the correct shape"""
        batch_size = 3

        for i in range(10):
            self.buffer.append(self.experience)

        batch = self.buffer.sample(batch_size)

        self.assertEqual(len(batch), 5)

        # states
        states = batch[0]
        self.assertEqual(states.shape, (batch_size, 32, 32))
        # action
        actions = batch[1]
        self.assertEqual(actions.shape, (batch_size, 1))
        # reward
        rewards = batch[2]
        self.assertEqual(rewards.shape, (batch_size, 1))
        # dones
        dones = batch[3]
        self.assertEqual(dones.shape, (batch_size, 1))
        # next states
        next_states = batch[4]
        self.assertEqual(next_states.shape, (batch_size, 32, 32))


class TestPrioReplayBuffer(TestCase):

    def setUp(self) -> None:
        self.buffer = PERBuffer(10)

        self.state = np.random.rand(32, 32)
        self.next_state = np.random.rand(32, 32)
        self.action = np.ones([1])
        self.reward = np.ones([1])
        self.done = np.zeros([1])
        self.experience = Experience(self.state, self.action, self.reward, self.done, self.next_state)

    def test_replay_buffer_append(self):
        """Test that you can append to the replay buffer and the latest experience has max priority"""

        self.assertEqual(len(self.buffer), 0)

        self.buffer.append(self.experience)

        self.assertEqual(len(self.buffer), 1)
        self.assertEqual(self.buffer.priorities[0], 1.0)

    def test_replay_buffer_sample(self):
        """Test that you can sample from the buffer and the outputs are the correct shape"""
        batch_size = 3

        for i in range(10):
            self.buffer.append(self.experience)

        batch, indices, weights = self.buffer.sample(batch_size)

        self.assertEqual(len(batch), 5)
        self.assertEqual(len(indices), batch_size)
        self.assertEqual(len(weights), batch_size)

        # states
        states = batch[0]
        self.assertEqual(states.shape, (batch_size, 32, 32))
        # action
        actions = batch[1]
        self.assertEqual(actions.shape, (batch_size, 1))
        # reward
        rewards = batch[2]
        self.assertEqual(rewards.shape, (batch_size, 1))
        # dones
        dones = batch[3]
        self.assertEqual(dones.shape, (batch_size, 1))
        # next states
        next_states = batch[4]
        self.assertEqual(next_states.shape, (batch_size, 32, 32))
