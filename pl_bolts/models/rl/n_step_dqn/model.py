"""
N Step Deep Q-network
"""

import argparse

import torch

from pl_bolts.models.rl.common import wrappers
from pl_bolts.models.rl.common.agents import ValueAgent
from pl_bolts.models.rl.common.experience import NStepExperienceSource
from pl_bolts.models.rl.common.memory import ReplayBuffer
from pl_bolts.models.rl.dqn.model import DQN


class NStepDQN(DQN):
    """ NStep DQN Model """

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__(hparams)
        self.hparams = hparams

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.env = wrappers.make_env(self.hparams.env)
        self.env.seed(123)

        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        self.net = None
        self.target_net = None
        self.build_networks()

        self.agent = ValueAgent(
            self.net,
            self.n_actions,
            eps_start=hparams.eps_start,
            eps_end=hparams.eps_end,
            eps_frames=hparams.eps_last_frame,
        )
        self.source = NStepExperienceSource(
            self.env, self.agent, device, n_steps=self.hparams.n_steps
        )
        self.buffer = ReplayBuffer(self.hparams.replay_size)

        self.total_reward = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.episode_steps = 0
        self.total_episode_steps = 0
        self.reward_list = []
        for _ in range(100):
            self.reward_list.append(-21)
        self.avg_reward = 0
