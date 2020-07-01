"""
Noisy DQN
"""
import argparse
from collections import OrderedDict
from typing import Tuple

import torch
import pytorch_lightning as pl

from pl_bolts.losses.rl import dqn_loss
from pl_bolts.models.rl.common import cli
from pl_bolts.models.rl.common.networks import NoisyCNN
from pl_bolts.models.rl.dqn_model import DQN


class NoisyDQN(DQN):
    """
    PyTorch Lightning implementation of `Noisy DQN <https://arxiv.org/abs/1706.10295>`_

    Paper authors: Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alex Graves,
    Vlad Mnih, Remi Munos, Demis Hassabis, Olivier Pietquin, Charles Blundell, Shane Legg

    Model implemented by:

        - `Donal Byrne <https://github.com/djbyrne>`

    Example:

        >>> from pl_bolts.models.rl.n_step_dqn_model import NStepDQN
        ...
        >>> model = NStepDQN("PongNoFrameskip-v4")

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
        lr: learning rate
        batch_size: size of minibatch pulled from the DataLoader
        replay_size: total capacity of the replay buffer
        warm_start_size: how many random steps through the environment to be carried out at the start of
        training to fill the buffer with a starting point
        sample_len: the number of samples to pull from the dataset iterator and feed to the DataLoader

    .. note:: Currently only supports CPU and single GPU training with `distributed_backend=dp`

    """

    def build_networks(self) -> None:
        """Initializes the Noisy DQN train and target networks"""
        self.net = NoisyCNN(self.obs_shape, self.n_actions)
        self.target_net = NoisyCNN(self.obs_shape, self.n_actions)

    def on_train_start(self) -> None:
        """Set the agents epsilon to 0 as the exploration comes from the network"""
        self.agent.epsilon = 0.0

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
        # step through environment with agent and add to buffer
        exp, reward, done = self.source.step()
        self.buffer.append(exp)

        self.episode_reward += reward
        self.episode_steps += 1

        # calculates training loss
        loss = dqn_loss(batch, self.net, self.target_net)

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
            "total_reward": torch.tensor(self.total_reward).to(self.device),
            "avg_reward": torch.tensor(self.avg_reward),
            "train_loss": loss,
            "episode_steps": torch.tensor(self.total_episode_steps),
        }
        status = {
            "steps": torch.tensor(self.global_step).to(self.device),
            "avg_reward": torch.tensor(self.avg_reward),
            "total_reward": torch.tensor(self.total_reward).to(self.device),
            "episodes": self.episode_count,
            "episode_steps": self.episode_steps,
            "epsilon": self.agent.epsilon,
        }

        return OrderedDict(
            {
                "loss": loss,
                "avg_reward": torch.tensor(self.avg_reward),
                "log": log,
                "progress_bar": status,
            }
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = cli.add_base_args(parser)
    parser = NoisyDQN.add_model_specific_args(parser)
    args = parser.parse_args()

    model = NoisyDQN(**args.__dict__)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)
