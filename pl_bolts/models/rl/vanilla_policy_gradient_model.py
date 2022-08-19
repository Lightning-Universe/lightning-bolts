import argparse
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor, optim
from torch.nn.functional import log_softmax, softmax
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pl_bolts.datamodules import ExperienceSourceDataset
from pl_bolts.models.rl.common.agents import PolicyAgent
from pl_bolts.models.rl.common.networks import MLP
from pl_bolts.utils import _GYM_AVAILABLE
from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg

if _GYM_AVAILABLE:
    import gym
else:  # pragma: no cover
    warn_missing_pkg("gym")


@under_review()
class VanillaPolicyGradient(LightningModule):
    r"""PyTorch Lightning implementation of `Vanilla Policy Gradient`_.

    Paper authors: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour

    Model implemented by:

        - `Donal Byrne <https://github.com/djbyrne>`

    Example:
        >>> from pl_bolts.models.rl.vanilla_policy_gradient_model import VanillaPolicyGradient
        ...
        >>> model = VanillaPolicyGradient("CartPole-v0")

    Train::

        trainer = Trainer()
        trainer.fit(model)

    Note:
        This example is based on:
        https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter11/04_cartpole_pg.py

    Note:
        Currently only supports CPU and single GPU training with `accelerator=dp`

    .. _`Vanilla Policy Gradient`:
        https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf
    """

    def __init__(
        self,
        env: str,
        gamma: float = 0.99,
        lr: float = 0.01,
        batch_size: int = 8,
        n_steps: int = 10,
        avg_reward_len: int = 100,
        entropy_beta: float = 0.01,
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
            entropy_beta: dictates the level of entropy per batch
            avg_reward_len: how many episodes to take into account when calculating the avg reward
            epoch_len: how many batches before pseudo epoch
        """
        super().__init__()

        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("This Module requires gym environment which is not installed yet.")

        # Hyperparameters
        self.lr = lr
        self.batch_size = batch_size
        self.batches_per_epoch = self.batch_size * epoch_len
        self.entropy_beta = entropy_beta
        self.gamma = gamma
        self.n_steps = n_steps

        self.save_hyperparameters()

        # Model components
        self.env = gym.make(env)
        self.net = MLP(self.env.observation_space.shape, self.env.action_space.n)
        self.agent = PolicyAgent(self.net)

        # Tracking metrics
        self.total_rewards = []
        self.episode_rewards = []
        self.done_episodes = 0
        self.avg_rewards = 0
        self.avg_reward_len = avg_reward_len
        self.eps = np.finfo(np.float32).eps.item()
        self.batch_states = []
        self.batch_actions = []

        self.state = self.env.reset()

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def train_batch(
        self,
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.

        Returns:
            yields a tuple of Lists containing tensors for states, actions and rewards of the batch.
        """

        while True:

            action = self.agent(self.state, self.device)

            next_state, reward, done, _ = self.env.step(action[0])

            self.episode_rewards.append(reward)
            self.batch_actions.append(action)
            self.batch_states.append(self.state)
            self.state = next_state

            if done:
                self.done_episodes += 1
                self.state = self.env.reset()
                self.total_rewards.append(sum(self.episode_rewards))
                self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))

                returns = self.compute_returns(self.episode_rewards)

                for idx in range(len(self.batch_actions)):
                    yield self.batch_states[idx], self.batch_actions[idx], returns[idx]

                self.batch_states = []
                self.batch_actions = []
                self.episode_rewards = []

    def compute_returns(self, rewards):
        """Calculate the discounted rewards of the batched rewards.

        Args:
            rewards: list of batched rewards

        Returns:
            list of discounted rewards
        """
        reward = 0
        returns = []

        for r in rewards[::-1]:
            reward = r + self.gamma * reward
            returns.insert(0, reward)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        return returns

    def loss(self, states, actions, scaled_rewards) -> Tensor:
        """Calculates the loss for VPG.

        Args:
            states: batched states
            actions: batch actions
            scaled_rewards: batche Q values

        Returns:
            loss for the current batch
        """

        logits = self.net(states)

        # policy loss
        log_prob = log_softmax(logits, dim=1)
        log_prob_actions = scaled_rewards * log_prob[range(self.batch_size), actions[0]]
        policy_loss = -log_prob_actions.mean()

        # entropy loss
        prob = softmax(logits, dim=1)
        entropy = -(prob * log_prob).sum(dim=1).mean()
        entropy_loss = -self.entropy_beta * entropy

        # total loss
        loss = policy_loss + entropy_loss

        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], _) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

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
        """Initialize Adam optimizer."""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = ExperienceSourceDataset(self.train_batch)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0][0][0].device.index if self.on_gpu else "cpu"

    @staticmethod
    def add_model_specific_args(arg_parser) -> argparse.ArgumentParser:
        """Adds arguments for DQN model.

        Note:
            These params are fine tuned for Pong env.

        Args:
            arg_parser: the current argument parser to add to

        Returns:
            arg_parser with model specific cargs added
        """

        arg_parser.add_argument("--entropy_beta", type=float, default=0.01, help="entropy value")
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


@under_review()
def cli_main():
    parser = argparse.ArgumentParser(add_help=False)

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = VanillaPolicyGradient.add_model_specific_args(parser)
    args = parser.parse_args()

    model = VanillaPolicyGradient(**args.__dict__)

    # save checkpoints based on avg_reward
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="avg_reward", mode="max", verbose=True)

    seed_everything(123)
    trainer = Trainer.from_argparse_args(args, deterministic=True, callbacks=checkpoint_callback)
    trainer.fit(model)


if __name__ == "__main__":
    cli_main()
