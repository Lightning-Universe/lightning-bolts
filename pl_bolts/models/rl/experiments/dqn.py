import argparse
from collections import OrderedDict
from typing import Tuple, List, Dict

import gym
import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning import seed_everything
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import numpy as np
from pl_bolts.datamodules import ExperienceSourceDataset
from pl_bolts.losses.rl import dqn_loss
from pl_bolts.models.rl.common import wrappers, cli
from pl_bolts.models.rl.common.agents import ValueAgent
from pl_bolts.models.rl.common.experience import ExperienceSource, RLDataset
from pl_bolts.models.rl.common.memory import ReplayBuffer
from pl_bolts.models.rl.common.networks import CNN
from pl_bolts.models.rl.ptan.actions import EpsilonGreedyActionSelector
from pl_bolts.models.rl.ptan.agents import TargetNet, DQNAgent
from pl_bolts.models.rl.ptan.experience import ExperienceSourceFirstLast, ExperienceReplayBuffer


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = 1.0
        self.epsilon_final = 0.02
        self.epsilon_frames = 10**5
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)


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
            avg_reward_len: int = 100,
            **kwargs,
    ):
        super().__init__()

        self.envs = []
        for _ in range(3):
            # env = gym.make(env)
            env_n = wrappers.make_env(env)
            env_n.seed(123)
            self.envs.append(env_n)

        self.obs_shape = env_n.observation_space.shape
        self.n_actions = env_n.action_space.n

        self.batch_size = batch_size * 3
        self.lr = learning_rate

        self.net = None
        self.target_net = None
        self.build_networks()
        self.selector = EpsilonGreedyActionSelector(epsilon=1.0)

        self.epsilon_tracker = EpsilonTracker(self.selector)
        self.agent = DQNAgent(self.net, self.selector, device='cuda:0')

        self.exp_source = ExperienceSourceFirstLast(
            self.envs, self.agent, gamma=0.99)
        self.buffer = ExperienceReplayBuffer(
            self.exp_source, buffer_size=100000)

        self.buffer.populate(10000)

        self.total_reward = 0

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

    def build_networks(self) -> None:
        """Initializes the DQN train and target networks"""
        self.net = CNN(self.obs_shape, self.n_actions)
        self.target_net = TargetNet(self.net)

    def train_batch(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Contains the logic for generating a new batch of data to be passed to the DataLoader

        Returns:
            yields a tuple of Lists containing tensors for states, actions and rewards of the batch.
        """

        self.buffer.populate(1)
        sample = self.buffer.sample(self.batch_size)

        return iter(sample)

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


        # calculates training loss
        loss = calc_loss_dqn(batch, self.net, self.target_net, gamma=0.99)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        rewards = self.exp_source.pop_total_rewards()

        if rewards:
            self.episode_steps += 1
            self.total_reward = rewards

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": self.total_reward,
            # "avg_reward": self.avg_reward,
            "train_loss": loss,
            # "episode_steps": self.total_episode_steps,
        }
        status = {
            "steps": self.global_step,
            # "avg_reward": self.avg_reward,
            "total_reward": self.total_reward,
            # "episodes": self.episode_count,
            # "episode_steps": self.episode_steps,
            # "epsilon": self.agent.epsilon,
        }

        return OrderedDict(
            {
                "loss": loss,
                # "avg_reward": self.avg_reward,
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
        dataset = ExperienceSourceDataset(self.train_batch)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader()

    @staticmethod
    def add_model_specific_args(arg_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = cli.add_base_args(parser)
    parser = DQN.add_model_specific_args(parser)
    args = parser.parse_args()

    model = DQN(**args.__dict__)

    seed_everything(123)
    trainer = pl.Trainer.from_argparse_args(args, deterministic=True)
    trainer.fit(model)



