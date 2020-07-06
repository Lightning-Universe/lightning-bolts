"""
Vanilla Policy Gradient

"""
import argparse
from collections import OrderedDict
from copy import deepcopy
from typing import Tuple, List

import gym
import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import Tensor
from torch.nn.functional import log_softmax, softmax
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pl_bolts.datamodules.experience_source import ExperienceSourceDataset, NStepExperienceSource
from pl_bolts.models.rl.common import cli
from pl_bolts.models.rl.common.agents import PolicyAgent
from pl_bolts.models.rl.common.networks import MLP
from pl_bolts.models.rl.common.wrappers import ToTensor


class PolicyGradient(pl.LightningModule):
    """ Vanilla Policy Gradient Model """

    def __init__(self, env: str, gamma: float = 0.99, lr: float = 1e-4, batch_size: int = 32,
                 entropy_beta: float = 0.01, batch_episodes: int = 4, *args, **kwargs) -> None:
        """
        PyTorch Lightning implementation of `Vanilla Policy Gradient
        <https://papers.nips.cc/paper/
        1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf>`_

        Paper authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour

        Model implemented by:

            - `Donal Byrne <https://github.com/djbyrne>`

        Example:

            >>> from pl_bolts.models.rl.vanilla_policy_gradient_model import PolicyGradient
            ...
            >>> model = PolicyGradient("PongNoFrameskip-v4")

        Train::

            trainer = Trainer()
            trainer.fit(model)

        Args:
            env: gym environment tag
            gamma: discount factor
            lr: learning rate
            batch_size: size of minibatch pulled from the DataLoader
            batch_episodes: how many episodes to rollout for each batch of training
            entropy_beta: dictates the level of entropy per batch

        .. note::
            This example is based on:
             https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition\
             /blob/master/Chapter11/04_cartpole_pg.py

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
        self.source = NStepExperienceSource(env=self.env, agent=self.agent, n_steps=10)

        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.batch_episodes = batch_episodes
        self.entropy_beta = entropy_beta
        self.baseline = 0

        # Metrics

        self.reward_sum = 0
        self.env_steps = 0
        self.total_steps = 0
        self.total_reward = 0
        self.episode_count = 0

        self.reward_list = []
        for _ in range(100):
            self.reward_list.append(torch.tensor(0))
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

    def calc_qvals(self, rewards: List[Tensor]) -> List[Tensor]:
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
        res = list(reversed(res))
        # Subtract the mean (baseline) from the q_vals to reduce the high variance
        sum_q = 0
        for rew in res:
            sum_q += rew
        mean_q = sum_q / len(res)
        return [q - mean_q for q in res]

    def loss(
        self,
        batch_scales: List[Tensor],
        batch_states: List[Tensor],
        batch_actions: List[Tensor],
    ) -> torch.Tensor:
        """
        Calculates the mse loss using a batch of states, actions and Q values from several episodes. These have all
        been flattend into a single tensor.

        Args:
            batch_scales: current mini batch of rewards minus the baseline
            batch_actions: current batch of actions
            batch_states: current batch of states

        Returns:
            loss
        """
        logits = self.net(batch_states)

        log_prob, policy_loss = self.calc_policy_loss(
            batch_actions, batch_scales, batch_states, logits
        )

        entropy_loss_v = self.calc_entropy_loss(log_prob, logits)

        loss = policy_loss + entropy_loss_v

        return loss

    def calc_entropy_loss(self, log_prob: Tensor, logits: Tensor) -> Tensor:
        """
        Calculates the entropy to be added to the loss function
        Args:
            log_prob: log probabilities for each action
            logits: the raw outputs of the network

        Returns:
            entropy penalty for each state
        """
        prob_v = softmax(logits, dim=1)
        entropy_v = -(prob_v * log_prob).sum(dim=1).mean()
        entropy_loss_v = -self.entropy_beta * entropy_v
        return entropy_loss_v

    @staticmethod
    def calc_policy_loss(
        batch_actions: Tensor, batch_qvals: Tensor, batch_states: Tensor, logits: Tensor
    ) -> Tuple[List, Tensor]:
        """
        Calculate the policy loss give the batch outputs and logits
        Args:
            batch_actions: actions from batched episodes
            batch_qvals: Q values from batched episodes
            batch_states: states from batched episodes
            logits: raw output of the network given the batch_states

        Returns:
            policy loss
        """
        log_prob = log_softmax(logits, dim=1)
        log_prob_actions = (
            batch_qvals * log_prob[range(len(batch_states)), batch_actions]
        )
        policy_loss = -log_prob_actions.mean()
        return log_prob, policy_loss

    def train_batch(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Contains the logic for generating a new batch of data to be passed to the DataLoader

        Returns:
            yields a tuple of Lists containing tensors for states, actions and rewards of the batch.
        """
        states = []
        actions = []
        scales = []

        for _ in range(self.batch_size):

            # take a step in the env
            exp, reward, done = self.source.step(self.device)
            self.env_steps += 1
            self.total_steps += 1

            # update the baseline
            self.reward_sum += exp.reward
            self.baseline = self.reward_sum / self.total_steps

            # gather the experience data
            states.append(exp.new_state)
            actions.append(exp.action)
            scales.append(exp.reward - self.baseline)

            self.total_reward += reward

            if done:
                # tracking metrics
                self.episode_count += 1
                self.reward_list.append(self.total_reward)
                self.avg_reward = sum(self.reward_list[-100:]) / 100

                self.logger.experiment.add_scalar("reward", self.total_reward, self.total_steps)

                # reset metrics
                self.total_reward = 0
                self.env_steps = 0

        yield from zip(states, actions, scales)

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
        states, actions, scales = batch

        # calculates training loss
        loss = self.loss(scales, states, actions)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        log = {
            "train_loss": loss,
            "avg_reward": self.avg_reward,
            "episode_count": self.episode_count,
            "baseline": self.baseline
        }

        return OrderedDict(
            {
                "loss": loss,
                "log": log,
                "progress_bar": log
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
            parent
        """
        arg_parser.add_argument(
            "--batch_episodes",
            type=int,
            default=4,
            help="how episodes to run per batch",
        )
        arg_parser.add_argument(
            "--entropy_beta", type=int, default=0.01, help="entropy beta"
        )
        return arg_parser


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = cli.add_base_args(parser)
    parser = PolicyGradient.add_model_specific_args(parser)
    args = parser.parse_args()

    model = PolicyGradient(**args.__dict__)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)
