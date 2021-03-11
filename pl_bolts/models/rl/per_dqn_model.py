"""
Prioritized Experience Replay DQN
"""
import argparse
from collections import OrderedDict
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from pl_bolts.datamodules import ExperienceSourceDataset
from pl_bolts.losses.rl import per_dqn_loss
from pl_bolts.models.rl.common.memory import Experience, PERBuffer
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

    .. note::
        This example is based on:
         https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter08/05_dqn_prio_replay.py

    .. note:: Currently only supports CPU and single GPU training with `distributed_backend=dp`

        """

    def train_batch(self, ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Contains the logic for generating a new batch of data to be passed to the DataLoader

        Returns:
            yields a Experience tuple containing the state, action, reward, done and next_state.
        """
        episode_reward = 0
        episode_steps = 0

        while True:
            self.total_steps += 1
            action = self.agent(self.state, self.device)

            next_state, r, is_done, _ = self.env.step(action[0])

            episode_reward += r
            episode_steps += 1

            exp = Experience(
                state=self.state,
                action=action[0],
                reward=r,
                done=is_done,
                new_state=next_state,
            )

            self.agent.update_epsilon(self.global_step)
            self.buffer.append(exp)
            self.state = next_state

            if is_done:
                self.done_episodes += 1
                self.total_rewards.append(episode_reward)
                self.total_episode_steps.append(episode_steps)
                self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len:]))
                self.state = self.env.reset()
                episode_steps = 0
                episode_reward = 0

            samples, indices, weights = self.buffer.sample(self.batch_size)

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
        loss, batch_weights = per_dqn_loss(samples, weights, self.net, self.target_net, self.gamma)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        # update priorities in buffer
        self.buffer.update_priorities(indices, batch_weights)

        # update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict({
            "total_reward": self.total_rewards[-1],
            "avg_reward": self.avg_rewards,
            "train_loss": loss,
            # "episodes": self.total_episode_steps,
        })

        return OrderedDict({
            "loss": loss,
            "avg_reward": self.avg_rewards,
        })

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        self.buffer = PERBuffer(self.replay_size)
        self.populate(self.warm_start_size)

        self.dataset = ExperienceSourceDataset(self.train_batch)
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size)


def cli_main():
    parser = argparse.ArgumentParser(add_help=False)

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = PERDQN.add_model_specific_args(parser)
    args = parser.parse_args()

    model = PERDQN(**args.__dict__)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == "__main__":
    cli_main()
