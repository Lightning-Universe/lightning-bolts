import argparse
from collections import OrderedDict
from typing import Tuple, List

import gym
import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.functional import log_softmax
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pl_bolts.datamodules import ExperienceSourceDataset
from pl_bolts.datamodules.experience_source import DiscountedExperienceSource
from pl_bolts.models.rl.common import cli
from pl_bolts.models.rl.common.agents import PolicyAgent
from pl_bolts.models.rl.common.networks import MLP


class Reinforce(pl.LightningModule):

    def __init__(
        self,
        env: str,
        gamma: float = 0.99,
        lr: float = 0.01,
        batch_size: int = 8,
        n_steps: int = 10,
        avg_reward_len: int = 100,
        num_envs: int = 1,
        entropy_beta: float = 0.01,
        epoch_len: int = 1000,
        num_batch_episodes: int = 4,
        **kwargs
    ) -> None:
        """
        PyTorch Lightning implementation of `REINFORCE
        <https://papers.nips.cc/paper/
        1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf>`_
        Paper authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour
        Model implemented by:

            - `Donal Byrne <https://github.com/djbyrne>`

        Example:
            >>> from pl_bolts.models.rl.reinforce_model import Reinforce
            ...
            >>> model = Reinforce("PongNoFrameskip-v4")

        Train::

            trainer = Trainer()
            trainer.fit(model)

        Args:
            env: gym environment tag
            gamma: discount factor
            lr: learning rate
            batch_size: size of minibatch pulled from the DataLoader
            batch_episodes: how many episodes to rollout for each batch of training
            avg_reward_len: how many episodes to take into account when calculating the avg reward

        Note:
            This example is based on:
            https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter11/02_cartpole_reinforce.py

        Note:
            Currently only supports CPU and single GPU training with `distributed_backend=dp`
        """
        super().__init__()

        # Hyperparameters
        self.lr = lr
        self.batch_size = batch_size * num_envs
        self.batches_per_epoch = self.batch_size * epoch_len
        self.entropy_beta = entropy_beta
        self.gamma = gamma
        self.n_steps = n_steps
        self.num_batch_episodes = num_batch_episodes

        self.save_hyperparameters()

        # Model components
        self.env = [gym.make(env) for _ in range(num_envs)]
        self.net = MLP(self.env[0].observation_space.shape, self.env[0].action_space.n)
        self.agent = PolicyAgent(self.net)
        self.exp_source = DiscountedExperienceSource(
            self.env, self.agent, gamma=gamma, n_steps=self.n_steps
        )

        # Tracking metrics
        self.total_steps = 0
        self.total_rewards = [0]
        self.done_episodes = 0
        self.avg_rewards = 0
        self.reward_sum = 0.0
        self.batch_episodes = 0
        self.avg_reward_len = avg_reward_len

        self.batch_states = []
        self.batch_actions = []
        self.batch_qvals = []
        self.cur_rewards = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def calc_qvals(self, rewards: List[float]) -> List[float]:
        """Calculate the discounted rewards of all rewards in list

        Args:
            rewards: list of rewards from latest batch

        Returns:
            list of discounted rewards
        """
        assert isinstance(rewards[0], float)

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * self.gamma) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def train_batch(
        self,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Contains the logic for generating a new batch of data to be passed to the DataLoader

        Yield:
            yields a tuple of Lists containing tensors for states, actions and rewards of the batch.
        """
        for exp in self.exp_source.runner(self.device):

            self.batch_states.append(exp.state)
            self.batch_actions.append(exp.action)
            self.cur_rewards.append(exp.reward)

            # Check if episode is completed and update trackers
            if exp.done:
                self.batch_qvals.extend(self.calc_qvals(self.cur_rewards))
                self.cur_rewards.clear()
                self.batch_episodes += 1

            # Check if episodes have finished and use total reward
            new_rewards = self.exp_source.pop_total_rewards()
            if new_rewards:
                for reward in new_rewards:
                    self.done_episodes += 1
                    self.total_rewards.append(reward)
                    self.avg_rewards = float(
                        np.mean(self.total_rewards[-self.avg_reward_len:])
                    )

            self.total_steps += 1

            if self.batch_episodes >= self.num_batch_episodes:
                for state, action, qval in zip(
                    self.batch_states, self.batch_actions, self.batch_qvals
                ):
                    yield state, action, qval

                self.batch_episodes = 0

            # Simulates epochs
            if self.total_steps % self.batches_per_epoch == 0:
                break

    def loss(self, states, actions, scaled_rewards) -> torch.Tensor:
        logits = self.net(states)

        # policy loss
        log_prob = log_softmax(logits, dim=1)
        log_prob_actions = scaled_rewards * log_prob[range(self.batch_size), actions]
        loss = -log_prob_actions.mean()

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved

        Args:
            batch: current mini batch of replay data
            _: batch number, not used

        Returns:
            Training loss and log metrics
        """
        states, actions, scaled_rewards = batch

        loss = self.loss(states, actions, scaled_rewards)

        log = {
            "episodes": self.done_episodes,
            "reward": self.total_rewards[-1],
            "avg_reward": self.avg_rewards,
        }

        return OrderedDict(
            {
                "loss": loss,
                "avg_reward": self.avg_rewards,
                "log": log,
                "progress_bar": log,
            }
        )

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceSourceDataset(self.train_batch)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0][0][0].device.index if self.on_gpu else "cpu"

    @staticmethod
    def add_model_specific_args(arg_parser) -> argparse.ArgumentParser:
        """
        Adds arguments for DQN model

        Note: these params are fine tuned for Pong env

        Args:
            arg_parser: the current argument parser to add to

        Returns:
            arg_parser with model specific cargs added
        """

        arg_parser.add_argument(
            "--entropy_beta", type=float, default=0.01, help="entropy value",
        )

        return arg_parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = cli.add_base_args(parser)
    parser = Reinforce.add_model_specific_args(parser)
    args = parser.parse_args()

    model = Reinforce(**args.__dict__)

    # save checkpoints based on avg_reward
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="avg_reward", mode="max", period=1, verbose=True
    )

    seed_everything(123)
    trainer = pl.Trainer.from_argparse_args(
        args, deterministic=True, checkpoint_callback=checkpoint_callback
    )
    trainer.fit(model)
