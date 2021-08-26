from unittest import TestCase

import gym
from torch import Tensor

from pl_bolts.models.rl.common.gym_wrappers import ToTensor


class TestToTensor(TestCase):
    def setUp(self) -> None:
        self.env = ToTensor(gym.make("CartPole-v0"))

    def test_wrapper(self):
        state = self.env.reset()
        self.assertIsInstance(state, Tensor)

        new_state, _, _, _ = self.env.step(1)
        self.assertIsInstance(new_state, Tensor)
