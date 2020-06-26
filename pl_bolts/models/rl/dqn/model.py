"""
Deep Reinforcement Learning: Deep Q-network (DQN)
This example is based on https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-
Second-Edition/blob/master/Chapter06/02_dqn_pong.py
"""

import argparse
from collections import OrderedDict
from typing import Tuple, List, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pl_bolts.models.rl.common import wrappers
from pl_bolts.models.rl.common.agents import ValueAgent
from pl_bolts.models.rl.common.experience import ExperienceSource, RLDataset
from pl_bolts.models.rl.common.memory import ReplayBuffer
from pl_bolts.models.rl.common.networks import CNN


class DQN(pl.LightningModule):
    """ Basic DQN Model """

    def __init__(
            self,
            env: str,
            gpus: int = 0,
            eps_start: float = 1.0,
            eps_end: float = 0.02,
            eps_last_frame: int = 150000,
            sync_rate: int = 1000,
            gamma: float = 0.99,
            learning_rate: float = 1e-4,
            batch_size: int = 32,
            replay_size: int = 100000,
            warm_start_size: int = 10000,
            num_samples: int = 500,
            **kwargs,
    ):
        """
        PyTorch Lightning implementation of `DQN <https://arxiv.org/abs/1312.5602>`_

        Paper authors: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves,
        Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller.

        Model implemented by:

            - `Donal Byrne <https://github.com/djbyrne>`

        Example:

            >>> from pl_bolts.models.rl.dqn.model import DQN
            ...
            >>> model = DQN("PongNoFrameskip-v4")

        Train::

            trainer = Trainer()
            trainer.fit(model)

        Args:
            env: gym environment tag
            gpus: number of gpus being used
            eps_start: starting value of epsilon for the epsilon-greedy exploration
            eps_end: final value of epsilon for the epsilon-greedy exploration
            eps_last_frame: the final frame in for the decrease of epsilon. At this frame espilon = eps_end
            sync_rate: the number of iterations between syncing up the target network with the train network
            gamma: discount factor
            learning_rate: learning rate
            batch_size: size of minibatch pulled from the DataLoader
            replay_size: total capacity of the replay buffer
            warm_start_size: how many random steps through the environment to be carried out at the start of
            training to fill the buffer with a starting point
            sample_len: the number of samples to pull from the dataset iterator and feed to the DataLoader

        ..note: Currently only supports cpu and single gpu training with `distributed_backend=dp`

        """
        super().__init__()

        # Environment
        self.env = wrappers.make_env(env)
        self.env.seed(123)

        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        # Model Attributes
        self.buffer = None
        self.source = None
        self.dataset = None

        self.net = None
        self.target_net = None
        self.build_networks()

        self.agent = ValueAgent(
            self.net,
            self.n_actions,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_frames=eps_last_frame,
        )

        # Hyperparameters
        self.sync_rate = sync_rate
        self.gamma = gamma
        self.lr = learning_rate
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.warm_start_size = warm_start_size
        self.sample_len = num_samples

        self.save_hyperparameters()

        # Metrics
        self.total_reward = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.episode_steps = 0
        self.total_episode_steps = 0
        self.reward_list = []
        for _ in range(100):
            self.reward_list.append(-21)
        self.avg_reward = -21

    def populate(self, warm_start: int) -> None:
        """Populates the buffer with initial experience"""
        if warm_start > 0:
            for _ in range(warm_start):
                self.source.agent.epsilon = 1.0
                exp, _, _ = self.source.step()
                self.buffer.append(exp)

    def build_networks(self) -> None:
        """Initializes the DQN train and target networks"""
        self.net = CNN(self.obs_shape, self.n_actions)
        self.target_net = CNN(self.obs_shape, self.n_actions)

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

    def loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = (
            self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        )

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

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
        self.agent.update_epsilon(self.global_step)

        # step through environment with agent and add to buffer
        exp, reward, done = self.source.step()
        self.buffer.append(exp)

        self.episode_reward += reward
        self.episode_steps += 1

        # calculates training loss
        loss = self.loss(batch)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.reward_list.append(self.total_reward)
            self.avg_reward = sum(self.reward_list[-100:]) / 100
            self.episode_count += 1
            self.episode_reward = 0
            self.total_episode_steps = self.episode_steps
            self.episode_steps = 0

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": self.total_reward,
            "avg_reward": self.avg_reward,
            "train_loss": loss,
            "episode_steps": self.total_episode_steps,
        }
        status = {
            "steps": self.global_step,
            "avg_reward": self.avg_reward,
            "total_reward": self.total_reward,
            "episodes": self.episode_count,
            "episode_steps": self.episode_steps,
            "epsilon": self.agent.epsilon,
        }

        return OrderedDict(
            {
                "loss": loss,
                "avg_reward": self.avg_reward,
                "log": log,
                "progress_bar": status,
            }
        )

    def test_step(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Evaluate the agent for 10 episodes"""
        self.agent.epsilon = 0.0
        test_reward = self.source.run_episode()

        return {"test_reward": test_reward}

    def test_epoch_end(self, outputs) -> Dict[str, torch.Tensor]:
        """Log the avg of the test results"""
        rewards = [x["test_reward"] for x in outputs]
        avg_reward = sum(rewards) / len(rewards)
        tensorboard_logs = {"avg_test_reward": avg_reward}
        return {"avg_test_reward": avg_reward, "log": tensorboard_logs}

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def prepare_data(self) -> None:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        device = torch.device(self.trainer.root_gpu) if self.trainer.gpus >= 1 else self.device
        self.source = ExperienceSource(self.env, self.agent, device)
        self.buffer = ReplayBuffer(self.replay_size)
        self.populate(self.warm_start_size)

        self.dataset = RLDataset(self.buffer, self.sample_len)

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Get test loader"""
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size)

    @staticmethod
    def add_model_specific_args(arg_parser) -> argparse.ArgumentParser:
        """
        Adds arguments for DQN model

        Note: these params are fine tuned for Pong env

        Args:
            arg_parser: parent parser
        """
        arg_parser.add_argument(
            "--sync_rate",
            type=int,
            default=1000,
            help="how many frames do we update the target network",
        )
        arg_parser.add_argument(
            "--replay_size",
            type=int,
            default=100000,
            help="capacity of the replay buffer",
        )
        arg_parser.add_argument(
            "--warm_start_size",
            type=int,
            default=10000,
            help="how many samples do we use to fill our buffer at the start of training",
        )
        arg_parser.add_argument(
            "--eps_last_frame",
            type=int,
            default=150000,
            help="what frame should epsilon stop decaying",
        )
        arg_parser.add_argument(
            "--eps_start", type=float, default=1.0, help="starting value of epsilon"
        )
        arg_parser.add_argument(
            "--eps_end", type=float, default=0.02, help="final value of epsilon"
        )
        arg_parser.add_argument(
            "--warm_start_steps",
            type=int,
            default=10000,
            help="max episode reward in the environment",
        )

        return arg_parser
