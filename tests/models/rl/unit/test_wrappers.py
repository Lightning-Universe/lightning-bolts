from unittest import TestCase

import gym
import torch

from pl_bolts.models.rl.common.gym_wrappers import ToTensor


class TestToTensor(TestCase):

    def setUp(self) -> None:
        self.env = ToTensor(gym.make("CartPole-v0"))

    def test_wrapper(self):
        state = self.env.reset()
        self.assertIsInstance(state, torch.Tensor)

        new_state, _, _, _ = self.env.step(1)
        self.assertIsInstance(new_state, torch.Tensor)


class TestAtari(TestCase):
    def setUp(self) -> None:
        from pl_bolts.models.rl.common.gym_wrappers import make_atari_env
        self.env = ToTensor(make_atari_env("ZaxxonNoFrameskip-v0"))

    def test_wrapper(self):
        state = self.env.reset()
        self.assertIsInstance(state, torch.Tensor)

        new_state, _, _, _ = self.env.step(1)
        self.assertIsInstance(new_state, torch.Tensor)


class TestGetName(TestCase):
    def setUp(self) -> None:
        from pl_bolts.models.rl.common.gym_wrappers import get_game_type
        self.test_func = get_game_type

    def test_get_game_type_robotics(self) -> None:
        game_type = self.test_func("FetchSlideDense-v1")
        assert game_type == 'robotics'

    def test_get_game_type_procgen(self) -> None:
        game_type = self.test_func("procgen-starpilot-v0")
        assert game_type == 'procgen'