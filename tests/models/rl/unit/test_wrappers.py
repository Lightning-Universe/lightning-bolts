from unittest import TestCase

import gym
from pl_bolts.models.rl.common.gym_wrappers import ToTensor
from torch import Tensor


class TestToTensor(TestCase):
    def setUp(self) -> None:
        self.env = ToTensor(gym.make("CartPole-v0"))

    def test_wrapper(self):
        state = self.env.reset()
        assert isinstance(state, Tensor)

        new_state, _, _, _ = self.env.step(1)
        assert isinstance(new_state, Tensor)
