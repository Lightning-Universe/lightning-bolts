"""
Prioritized Experience Replay DQN
"""
import argparse
from collections import OrderedDict
from typing import Tuple, List

import torch
import pytorch_lightning as pl

from pl_bolts.losses.rl import per_dqn_loss
from pl_bolts.models.rl.common import cli
from pl_bolts.models.rl.common.experience import ExperienceSource, PrioRLDataset
from pl_bolts.models.rl.common.memory import PERBuffer
from pl_bolts.models.rl.dqn_model import DQN


class PERDQN(DQN):
    """
    PyTorch Lightning implementation of `DQN With Prioritized Experience Replay <https://arxiv.org/abs/1511.05952>`_

    Paper authors: Tom Schaul, John Quan, Ioannis Antonoglou, David Silver

    Model implemented by:

        - `Donal Byrne <https://github.com/djbyrne>`

    Example:

            >>> from pl_bolts.models.rl.per_dqn_model import PERDQN
            ...
            >>> model = PERDQN("PongNoFrameskip-v4")

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
            num_samples: the number of samples to pull from the dataset iterator and feed to the DataLoader

        .. note::
            This example is based on:
             https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition\
             /blob/master/Chapter08/05_dqn_prio_replay.py

        .. note:: Currently only supports CPU and single GPU training with `distributed_backend=dp`

        """
    def train_batch(self) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """
        Contains the logic for generating a new batch of data to be passed to the DataLoader

        Returns:
            yields a Experience tuple.
        """
        self.agent.update_epsilon(self.total_steps)

        # take a step in the env
        exp, reward, done = self.source.step(self.device)
        self.episode_steps += 1
        self.total_steps += 1

        self.reward_sum += exp.reward
        self.episode_reward += reward

        # gather the experience data
        self.buffer.append(exp)

        if done:
            # tracking metrics
            self.episode_count += 1
            self.reward_list.append(self.total_reward)
            self.avg_reward = sum(self.reward_list[-self.avg_reward_len:]) / self.avg_reward_len
            self.total_reward = self.episode_reward
            self.total_episode_steps = self.episode_steps

            self.logger.experiment.add_scalar("reward", self.total_reward, self.total_steps)

            # reset metrics
            self.episode_reward = 0
            self.episode_steps = 0

        samples, indices, weights = self.buffer.sample(self.sample_size)

        states, actions, rewards, dones, new_states = samples

        for idx, _ in enumerate(dones):
            yield (
                      states[idx],
                      actions[idx],
                      rewards[idx],
                      dones[idx],
                      new_states[idx],
                  ), indices[idx], weights[idx]

    def training_step(self, batch, _) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved

        Args:
            batch: current mini batch of replay data
            _: batch number, not used

        Returns:
            Training loss and log metrics
        """
        samples, indices, weights = batch

        indices = indices.cpu().numpy()

        # calculates training loss
        loss, batch_weights = per_dqn_loss(samples, weights, self.net, self.target_net)

        # update priorities in buffer
        self.buffer.update_priorities(indices, batch_weights)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

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

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def prepare_data(self) -> None:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        self.source = ExperienceSource(self.env, self.agent)
        self.buffer = PERBuffer(self.replay_size)
        self.populate(self.warm_start_size)

        self.dataset = PrioRLDataset(self.buffer, self.batch_size)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = cli.add_base_args(parser)
    parser = PERDQN.add_model_specific_args(parser)
    args = parser.parse_args()

    model = PERDQN(**args.__dict__)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)
