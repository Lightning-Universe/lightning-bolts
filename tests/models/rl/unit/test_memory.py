from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import torch

from pl_bolts.models.rl.common.memory import Buffer, Experience, MultiStepBuffer, PERBuffer, ReplayBuffer


class TestBuffer(TestCase):
    def train_batch(self):
        """Returns an iterator used for testing."""
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
        """check that a sinlge sample is returned."""
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
        """Test that you can append to the replay buffer."""

        self.assertEqual(len(self.buffer), self.warm_start)

        self.buffer.append(self.experience)

        self.assertEqual(len(self.buffer), self.warm_start + 1)

    def test_replay_buffer_populate(self):
        """Tests that the buffer is populated correctly with warm_start."""
        self.assertEqual(len(self.buffer.buffer), self.warm_start)

    def test_replay_buffer_update(self):
        """Tests that buffer append works correctly."""
        batch_size = 3
        self.assertEqual(len(self.buffer.buffer), self.warm_start)
        for i in range(batch_size):
            self.buffer.append(self.experience)
        self.assertEqual(len(self.buffer.buffer), self.warm_start + batch_size)

    def test_replay_buffer_sample(self):
        """Test that you can sample from the buffer and the outputs are the correct shape."""
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
        """Test that you can append to the replay buffer and the latest experience has max priority."""

        self.assertEqual(len(self.buffer), 0)

        self.buffer.append(self.experience)

        self.assertEqual(len(self.buffer), 1)
        self.assertEqual(self.buffer.priorities[0], 1.0)

    def test_replay_buffer_sample(self):
        """Test that you can sample from the buffer and the outputs are the correct shape."""
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


class TestMultiStepReplayBuffer(TestCase):
    def setUp(self) -> None:
        self.gamma = 0.9
        self.buffer = MultiStepBuffer(capacity=10, n_steps=2, gamma=self.gamma)

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

    def test_append_single_experience_less_than_n(self):
        """If a single experience is added and n > 1 nothing should be added to the buffer as it is waiting
        experiences to equal n."""
        self.assertEqual(len(self.buffer), 0)

        self.buffer.append(self.experience01)

        self.assertEqual(len(self.buffer), 0)

    def test_append_single_experience(self):
        """If a single experience is added and n > 1 nothing should be added to the buffer as it is waiting
        experiences to equal n."""
        self.assertEqual(len(self.buffer), 0)

        self.buffer.append(self.experience01)

        self.assertEqual(len(self.buffer.exp_history_queue), 0)
        self.assertEqual(len(self.buffer.history), 1)

    def test_append_single_experience2(self):
        """If a single experience is added and the number of experiences collected >= n, the multi step experience
        should be added to the full buffer."""
        self.assertEqual(len(self.buffer), 0)

        self.buffer.append(self.experience01)
        self.buffer.append(self.experience02)

        self.assertEqual(len(self.buffer.buffer), 1)
        self.assertEqual(len(self.buffer.history), 2)

    def test_sample_single_experience(self):
        """if there is only a single experience added, sample should return nothing."""
        self.buffer.append(self.experience01)

        with self.assertRaises(Exception) as context:
            _ = self.buffer.sample(batch_size=1)

        self.assertIsInstance(context.exception, Exception)

    def test_sample_multi_experience(self):
        """if there is only a single experience added, sample should return nothing."""
        self.buffer.append(self.experience01)
        self.buffer.append(self.experience02)

        batch = self.buffer.sample(batch_size=1)

        next_state = batch[4]
        self.assertEqual(next_state.all(), self.next_state_02.all())

    def test_get_transition_info_2_step(self):
        """Test that the accumulated experience is correct and."""
        self.buffer.append(self.experience01)
        self.buffer.append(self.experience02)

        reward = self.buffer.buffer[0].reward
        next_state = self.buffer.buffer[0].new_state
        done = self.buffer.buffer[0].done

        reward_gt = self.experience01.reward + (self.gamma * self.experience02.reward) * (1 - done)

        self.assertEqual(reward, reward_gt)
        self.assertEqual(next_state.all(), self.next_state_02.all())
        self.assertEqual(self.experience02.done, done)

    def test_get_transition_info_3_step(self):
        """Test that the accumulated experience is correct with multi step."""
        self.buffer = MultiStepBuffer(capacity=10, n_steps=3, gamma=self.gamma)

        self.buffer.append(self.experience01)
        self.buffer.append(self.experience02)
        self.buffer.append(self.experience02)

        reward = self.buffer.buffer[0].reward
        next_state = self.buffer.buffer[0].new_state
        done = self.buffer.buffer[0].done

        reward_01 = self.experience02.reward + self.gamma * self.experience03.reward * (1 - done)
        reward_gt = self.experience01.reward + self.gamma * reward_01 * (1 - done)

        self.assertEqual(reward, reward_gt)
        self.assertEqual(next_state.all(), self.next_state_02.all())
        self.assertEqual(self.experience03.done, done)

    def test_sample_3_step(self):
        """Test that final output of the 3 step sample is correct."""
        self.buffer = MultiStepBuffer(capacity=10, n_steps=3, gamma=self.gamma)

        self.buffer.append(self.experience01)
        self.buffer.append(self.experience02)
        self.buffer.append(self.experience02)

        reward_gt = 1.71

        batch = self.buffer.sample(1)

        self.assertEqual(batch[0].all(), self.experience01.state.all())
        self.assertEqual(batch[1], self.experience01.action)
        self.assertEqual(batch[2], reward_gt)
        self.assertEqual(batch[3], self.experience02.done)
        self.assertEqual(batch[4].all(), self.experience02.new_state.all())
