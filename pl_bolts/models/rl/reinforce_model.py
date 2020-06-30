"""

"""
import argparse
from collections import OrderedDict
from copy import deepcopy
from itertools import chain
from pprint import pprint
from typing import Tuple, List

import gym
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import Tensor
from torch.nn.functional import log_softmax
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pl_bolts.models.rl import DQN
from pl_bolts.models.rl.common import cli
from pl_bolts.models.rl.common.agents import PolicyAgent
from pl_bolts.models.rl.common.experience import EpisodicExperienceStream
from pl_bolts.models.rl.common.memory import Experience
from pl_bolts.models.rl.common.networks import MLP
from pl_bolts.models.rl.common.wrappers import ToTensor


class Reinforce(pl.LightningModule):
    """ Basic REINFORCE Policy Model """

    def __init__(self, env: str, gamma: float = 0.99, lr: float = 1e-4, batch_size: int = 32,
                 batch_episodes: int = 4, **kwargs) -> None:
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

        .. note::
            This example is based on:
             https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition\
             /blob/master/Chapter11/02_cartpole_reinforce.py

        .. note:: Currently only supports CPU and single GPU training with `distributed_backend=dp`

        """
        super().__init__()

        # self.env = wrappers.make_env(self.hparams.env)    # use for Atari
        self.env = ToTensor(gym.make(env))  # use for Box2D/Control
        self.env.seed(123)

        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        self.net = None
        self.build_networks()

        self.agent = PolicyAgent(self.net)

        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.batch_episodes = batch_episodes

        self.total_reward = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.episode_steps = 0
        self.total_episode_steps = 0

        self.reward_list = []
        for _ in range(100):
            self.reward_list.append(0)
        self.avg_reward = 0

    def build_networks(self) -> None:
        """Initializes the DQN train and target networks"""
        self.net = MLP(self.obs_shape, self.n_actions)

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

    def calc_qvals(self, rewards: List[List]) -> List[List]:
        """
        Takes in the rewards for each batched episode and returns list of qvals for each batched episode

        Args:
            rewards: list of rewards for each episodes in the batch

        Returns:
            List of qvals for each episodes
        """
        res = []
        sum_r = 0.0
        for reward in reversed(rewards):
            sum_r *= self.gamma
            sum_r += reward
            res.append(deepcopy(sum_r))
        return list(reversed(res))

    def process_batch(
        self, batch: List[List[Experience]]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Takes in a batch of episodes and retrieves the q vals, the states and the actions for the batch

        Args:
            batch: list of episodes, each containing a list of Experiences

        Returns:
            q_vals, states and actions used for calculating the loss
        """
        # get outputs for each episode
        batch_rewards, batch_states, batch_actions = [], [], []
        for episode in batch:
            ep_rewards, ep_states, ep_actions = [], [], []

            # log the outputs for each step
            for step in episode:
                ep_rewards.append(step[2].float())
                ep_states.append(step[0])
                ep_actions.append(step[1])

            # add episode outputs to the batch
            batch_rewards.append(ep_rewards)
            batch_states.append(ep_states)
            batch_actions.append(ep_actions)

        # get qvals
        batch_qvals = []
        for reward in batch_rewards:
            batch_qvals.append(self.calc_qvals(reward))

        # flatten the batched outputs
        batch_actions, batch_qvals, batch_rewards, batch_states = self.flatten_batch(
            batch_actions, batch_qvals, batch_rewards, batch_states
        )

        return batch_qvals, batch_states, batch_actions, batch_rewards

    @staticmethod
    def flatten_batch(
        batch_actions: List[List[Tensor]],
        batch_qvals: List[List[Tensor]],
        batch_rewards: List[List[Tensor]],
        batch_states: List[List[Tuple[Tensor, Tensor]]],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Takes in the outputs of the processed batch and flattens the several episodes into a single tensor for each
        batched output

        Args:
            batch_actions: actions taken in each batch episodes
            batch_qvals: Q vals for each batch episode
            batch_rewards: reward for each batch episode
            batch_states: states for each batch episodes

        Returns:
            The input batched results flattend into a single tensor
        """
        # flatten all episode steps into a single list
        batch_qvals = list(chain.from_iterable(batch_qvals))
        batch_states = list(chain.from_iterable(batch_states))
        batch_actions = list(chain.from_iterable(batch_actions))
        batch_rewards = list(chain.from_iterable(batch_rewards))

        # stack steps into single tensor and remove extra dimension
        batch_qvals = torch.stack(batch_qvals).squeeze()
        batch_states = torch.stack(batch_states).squeeze()
        batch_actions = torch.stack(batch_actions).squeeze()
        batch_rewards = torch.stack(batch_rewards).squeeze()

        return batch_actions, batch_qvals, batch_rewards, batch_states

    def loss(
        self,
        batch_qvals: List[Tensor],
        batch_states: List[Tensor],
        batch_actions: List[Tensor],
    ) -> torch.Tensor:
        """
        Calculates the mse loss using a batch of states, actions and Q values from several episodes. These have all
        been flattend into a single tensor.

        Args:
            batch_qvals: current mini batch of q values
            batch_actions: current batch of actions
            batch_states: current batch of states

        Returns:
            loss
        """
        logits = self.net(batch_states)
        log_prob = log_softmax(logits, dim=1)
        log_prob_actions = (
            batch_qvals * log_prob[range(len(batch_states)), batch_actions]
        )
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
        device = self.get_device(batch)

        batch_qvals, batch_states, batch_actions, batch_rewards = self.process_batch(
            batch
        )

        # get avg reward over the batched episodes
        self.episode_reward = sum(batch_rewards) / len(batch)
        self.reward_list.append(self.episode_reward)
        self.avg_reward = sum(self.reward_list) / len(self.reward_list)

        # calculates training loss
        loss = self.loss(batch_qvals, batch_states, batch_actions)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        self.episode_count += self.batch_episodes

        log = {
            "episode_reward": torch.tensor(self.episode_reward).to(device),
            "train_loss": loss,
            "avg_reward": self.avg_reward,
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "episode_reward": torch.tensor(self.episode_reward).to(device),
            "episodes": torch.tensor(self.episode_count),
            "avg_reward": self.avg_reward,
        }

        self.episode_reward = 0

        return OrderedDict(
            {
                "loss": loss,
                "reward": self.avg_reward,
                "log": log,
                "progress_bar": status,
            }
        )

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = EpisodicExperienceStream(
            self.env, self.agent, self.device, episodes=self.batch_episodes
        )
        dataloader = DataLoader(dataset=dataset)
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
            "--batch_episodes",
            type=int,
            default=4,
            help="how many episodes to run per batch",
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

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)
