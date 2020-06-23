"""
N Step Deep Q-network
This example is based on: https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/
blob/master/Chapter08/05_dqn_prio_replay.py
"""

from collections import OrderedDict
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader

from pl_bolts.models.rl.common.agents import ValueAgent
from pl_bolts.models.rl.common.experience import ExperienceSource, PrioRLDataset
from pl_bolts.models.rl.common.memory import PERBuffer
from pl_bolts.models.rl.dqn.model import DQN


class PERDQN(DQN):
    """ PER DQN Model """

    def __init__(self, hparams):
        super().__init__(hparams)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.agent = ValueAgent(
            self.net,
            self.n_actions,
            eps_start=hparams.eps_start,
            eps_end=hparams.eps_end,
            eps_frames=hparams.eps_last_frame,
        )
        self.source = ExperienceSource(self.env, self.agent, device)
        self.buffer = PERBuffer(self.hparams.replay_size)

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
        # self.buffer.update_beta(self.global_step)

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
        if self.global_step % self.hparams.sync_rate == 0:
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
            exp_sa_vals = next_s_vals.detach() * self.hparams.gamma + rewards
        loss = (state_action_vals - exp_sa_vals) ** 2
        losses_v = batch_weights * loss
        return losses_v.mean(), (losses_v + 1e-5).data.cpu().numpy()

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        self.buffer = PERBuffer(self.hparams.replay_size)
        self.populate(self.hparams.warm_start_size)

        dataset = PrioRLDataset(self.buffer, self.hparams.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size,)
        return dataloader
