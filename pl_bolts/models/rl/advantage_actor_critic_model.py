"""
Advantage Actor Critic (A2C)
"""
import argparse
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pl_bolts.datamodules import ExperienceSourceDataset
from pl_bolts.models.rl.common.agents import ActorCriticAgent
from pl_bolts.models.rl.common.networks import ActorCriticMLP
from pl_bolts.utils import _GYM_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _GYM_AVAILABLE:
    import gym
else:  # pragma: no cover
    warn_missing_pkg('gym')


class AdvantageActorCritic(pl.LightningModule):
    """
    PyTorch Lightning implementation of `Advantage Actor Critic
    <https://arxiv.org/abs/1602.01783v2>`

    Paper Authors: Volodymyr Mnih, Adrià Puigdomènech Badia, et al.

    Model implemented by:

        - `Jason Wang <https://github.com/blahBlahhhJ>`

    Example:
        >>> from pl_bolts.models.rl.advantage_actor_critic_model import AdvantageActorCritic
        ...
        >>> model = AdvantageActorCritic("CartPole-v0")

    Train::

        trainer = Trainer()
        trainer.fit(model)
    """
    def __init__(
        self,
        env: str,
        gamma: float = 0.99,
        lr: float = 0.001,
        batch_size: int = 32,
        n_steps: int = 10,
        avg_reward_len: int = 100,
        entropy_beta: float = 0.01,
        critic_beta: float = 0.5,
        epoch_len: int = 1000,
        **kwargs
    ) -> None:
        """
        Args:
            env: gym environment tag
            gamma: discount factor
            lr: learning rate
            batch_size: size of minibatch pulled from the DataLoader
            batch_episodes: how many episodes to rollout for each batch of training
            avg_reward_len: how many episodes to take into account when calculating the avg reward
            entropy_beta: dictates the level of entropy per batch
            critic_beta: dictates the level of critic loss per batch
            epoch_len: how many batches before pseudo epoch
        """
        super().__init__()

        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('This Module requires gym environment which is not installed yet.')

        # Hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.batches_per_epoch = self.batch_size * epoch_len
        self.entropy_beta = entropy_beta
        self.critic_beta = critic_beta
        self.gamma = gamma
        self.n_steps = n_steps

        self.save_hyperparameters()

        # Model components
        self.env = gym.make(env)
        self.net = ActorCriticMLP(self.env.observation_space.shape, self.env.action_space.n)
        self.agent = ActorCriticAgent(self.net)

        # Tracking metrics
        self.total_rewards = [0]
        self.episode_reward = 0
        self.done_episodes = 0
        self.avg_rewards = 0
        self.avg_reward_len = avg_reward_len
        self.eps = np.finfo(np.float32).eps.item()
        self.batch_states = []
        self.batch_actions = []
        self.batch_rewards = []
        self.batch_masks = []

        self.state = self.env.reset()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Passes in a state x through the network and gets the log prob of each action
        and the value for the state as an output

        Args:
            x: environment state

        Returns:
            action log probabilities, values
        """
        if not isinstance(x, list):
            x = [x]

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)

        logprobs, values = self.net(x)
        return logprobs, values

    def train_batch(self, ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Contains the logic for generating a new batch of data to be passed to the DataLoader

        Returns:
            yields a tuple of Lists containing tensors for
            states, actions, and returns of the batch.

        Note:
            states: a list of numpy array
            actions: a list of list of int
            returns: a torch tensor
        """
        while True:
            for _ in range(self.batch_size):
                action = self.agent(self.state, self.device)[0]

                next_state, reward, done, _ = self.env.step(action)

                self.batch_rewards.append(reward)
                self.batch_actions.append(action)
                self.batch_states.append(self.state)
                self.batch_masks.append(done)
                self.state = next_state
                self.episode_reward += reward

                if done:
                    self.done_episodes += 1
                    self.state = self.env.reset()
                    self.total_rewards.append(self.episode_reward)
                    self.episode_reward = 0
                    self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len:]))

            _, last_value = self.forward(self.state)

            returns = self.compute_returns(self.batch_rewards, self.batch_masks, last_value)
            for idx in range(self.batch_size):
                yield self.batch_states[idx], self.batch_actions[idx], returns[idx]

            self.batch_states = []
            self.batch_actions = []
            self.batch_rewards = []
            self.batch_masks = []

    def compute_returns(self, rewards, dones, last_value):
        """
        Calculate the discounted rewards of the batched rewards

        Args:
            rewards: list of batched rewards
            dones: list of done masks
            last_value: the predicted value for the last state (for bootstrap)

        Returns:
            list of discounted rewards
        """
        g = last_value
        returns = []

        for r, d in zip(rewards[::-1], dones[::-1]):
            g = r + self.gamma * g * (1 - d)
            returns.append(g)

        # reverse list and stop the gradients
        returns = torch.tensor(returns[::-1])

        return returns

    def loss(self, states, actions, returns):
        """
        Calculates the loss for A2C which is a weighted sum of
        actor loss (MSE), critic loss (PG), and entropy (for exploration)

        Args:
            states: tensor of shape (batch_size, state dimension)
            actions: tensor of shape (batch_size, )
            returns: tensor of shape (batch_size, )
        """

        logprobs, values = self.net(states)

        # calculates (normalized) advantage
        with torch.no_grad():
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + self.eps)

        # entropy loss
        entropy = -logprobs.exp() * logprobs
        entropy = self.entropy_beta * entropy.sum(1).mean()

        # actor loss
        logprobs = logprobs[range(self.batch_size), actions]
        actor_loss = -(logprobs * advs).mean()

        # critic loss
        critic_loss = self.critic_beta * torch.square(returns - values).mean()

        # total loss (weighted sum)
        total_loss = actor_loss + critic_loss - entropy
        return total_loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> OrderedDict:
        """
        Perform one actor-critic update using a batch of data

        Args:
            batch: a batch of (states, actions, returns)
        """
        states, actions, returns = batch
        loss = self.loss(states, actions, returns)

        log = {
            "episodes": self.done_episodes,
            "reward": self.total_rewards[-1],
            "avg_reward": self.avg_rewards,
        }
        return OrderedDict({
            "loss": loss,
            "avg_reward": self.avg_rewards,
            "log": log,
            "progress_bar": log,
        })

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
        Adds arguments for A2C model

        Args:
            arg_parser: the current argument parser to add to

        Returns:
            arg_parser with model specific cargs added
        """

        arg_parser.add_argument("--entropy_beta", type=float, default=0.01, help="entropy coefficient")
        arg_parser.add_argument("--critic_beta", type=float, default=0.5, help="critic loss coefficient")
        arg_parser.add_argument("--batches_per_epoch", type=int, default=10000, help="number of batches in an epoch")
        arg_parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
        arg_parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
        arg_parser.add_argument("--env", type=str, required=True, help="gym environment tag")
        arg_parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        arg_parser.add_argument("--seed", type=int, default=123, help="seed for training run")

        arg_parser.add_argument(
            "--avg_reward_len",
            type=int,
            default=100,
            help="how many episodes to include in avg reward",
        )

        return arg_parser


def cli_main():
    parser = argparse.ArgumentParser(add_help=False)

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = AdvantageActorCritic.add_model_specific_args(parser)
    args = parser.parse_args()

    model = AdvantageActorCritic(**args.__dict__)

    # save checkpoints based on avg_reward
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="avg_reward", mode="max", period=1, verbose=True)

    seed_everything(123)
    trainer = pl.Trainer.from_argparse_args(args, deterministic=True, checkpoint_callback=checkpoint_callback)
    trainer.fit(model)


if __name__ == '__main__':
    cli_main()
