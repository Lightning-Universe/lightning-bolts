"""
N Step Deep Q-network
This example is based on: https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/
blob/master/Chapter08/05_dqn_prio_replay.py
"""

from collections import OrderedDict
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader

from pl_bolts.models.reinforcement.common.experience import ExperienceSource, PrioRLDataset
from pl_bolts.models.reinforcement.common.memory import PERBuffer
from pl_bolts.models.reinforcement.dqn.model import DQN


class PERDQN(DQN):
    """ PER DQN Model """

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
    ):
        """
        PyTorch Lightning implementation of `DQN With Prioritized Experience Replay <https://arxiv.org/abs/1511.05952>`_

        Paper authors: Tom Schaul, John Quan, Ioannis Antonoglou, David Silver

        Model implemented by:

            - `Donal Byrne <https://github.com/djbyrne>`

        Example:

            >>> from pl_bolts.models.reinforcement.per_dqn.model import PERDQN
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
        """
        super().__init__(env, gpus, eps_start, eps_end, eps_last_frame, sync_rate, gamma, learning_rate, batch_size,
                         replay_size, warm_start_size, num_samples)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.source = ExperienceSource(self.env, self.agent, device)
        self.buffer = PERBuffer(self.replay_size)

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

        self.agent.update_epsilon(self.global_step)

        # step through environment with agent and add to buffer
        exp, reward, done = self.source.step()
        self.buffer.append(exp)

        self.episode_reward += reward
        self.episode_steps += 1

        # calculates training loss
        loss, batch_weights = self.loss(samples, weights)

        # update priorities in buffer
        self.buffer.update_priorities(indices, batch_weights)

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

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def loss(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_weights: List
    ) -> Tuple[torch.Tensor, List]:
        """
        Calculates the mse loss with the priority weights of the batch from the PER buffer

        Args:
            batch: current mini batch of replay data
            batch_weights: how each of these samples are weighted in terms of priority

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        batch_weights = torch.tensor(batch_weights)

        actions_v = actions.unsqueeze(-1)
        state_action_vals = self.net(states).gather(1, actions_v)
        state_action_vals = state_action_vals.squeeze(-1)
        with torch.no_grad():
            next_s_vals = self.target_net(next_states).max(1)[0]
            next_s_vals[dones] = 0.0
            exp_sa_vals = next_s_vals.detach() * self.gamma + rewards
        loss = (state_action_vals - exp_sa_vals) ** 2
        losses_v = batch_weights * loss
        return losses_v.mean(), (losses_v + 1e-5).data.cpu().numpy()

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        self.buffer = PERBuffer(self.replay_size)
        self.populate(self.warm_start_size)

        dataset = PrioRLDataset(self.buffer, self.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size,)
        return dataloader
