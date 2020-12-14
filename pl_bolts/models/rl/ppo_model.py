import argparse
from collections import OrderedDict
from typing import List, Tuple

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.datamodules import ExperienceSourceDataset
from pl_bolts.datamodules.experience_source import Experience
from pl_bolts.models.rl.common.agents import PolicyAgentContinous, PolicyAgentCategorical
from pl_bolts.models.rl.common.networks import MLP
from pl_bolts.utils.warnings import warn_missing_pkg

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import numpy as np

try:
    import gym
except ModuleNotFoundError:
    warn_missing_pkg('gym')  # pragma: no-cover
    _GYM_AVAILABLE = False
else:
    _GYM_AVAILABLE = True


class PPO(pl.LightningModule):
    """
    PyTorch Lightning implementation of `PPO
    <https://arxiv.org/abs/1707.06347>`_
    Paper authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov

    Model implemented by:

    Example:
        >>> from pl_bolts.models.rl.ppo_model import PPO
        >>> model = PPO("CartPole-v0")
    Train:
        trainer = Trainer()
        trainer.fit(model)
    Note:
        This example is based on:
        https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py
        https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py

    """
    def __init__(
        self,
        env: str,
        gamma: float = 0.99,
        lam: float = 0.95,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        max_episode_len: float = 200,
        batch_size: int = 512,
        steps_per_epoch: int = 2048,
        clip_ratio: float = 0.2,
        **kwargs
    ) -> None:

        """
        Args:
            env: gym environment tag
            gamma: discount factor
            lam: advantage discount factor (lambda in the paper)
            lr_actor: learning rate of actor network
            lr_critic: learning rate of critic network
            max_episode_len: maximum number interactions (actions) in an episode
            batch_size:  batch_size when training network- can simulate number of policy updates performed per epoch
            steps_per_epoch: how many action-state pairs to rollout for trajectory collection per epoch
            clip_ratio: hyperparameter for clipping in the policy objective
        """
        super().__init__()

        if not _GYM_AVAILABLE:
            raise ModuleNotFoundError('This Module requires gym environment which is not installed yet.')

        # Hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.max_episode_len = max_episode_len
        self.clip_ratio = clip_ratio
        self.save_hyperparameters()

        self.env = gym.make(env)
        # value network
        self.critic = MLP(self.env.observation_space.shape, 1)
        # policy network (agent)
        if type(self.env.action_space) == gym.spaces.box.Box:
            act_dim = self.env.action_space.shape[0]
            self.net = MLP(self.env.observation_space.shape, act_dim)
            self.agent = PolicyAgentContinous(self.net, act_dim)
        elif type(self.env.action_space) == gym.spaces.discrete.Discrete:
            self.net = MLP(self.env.observation_space.shape, self.env.action_space.n)
            self.agent = PolicyAgentCategorical(self.net)
        else:
            raise NotImplementedError('Env action space should be of type Box (continous) or Discrete (categorical). Got type: ', 
                                      type(self.env.action_space))

        self.batch_states = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []

        self.ep_rewards = []
        self.ep_values = []

        self.done_episodes = 0
        self.epoch_rewards = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_reward = 0

        self.state = torch.FloatTensor(self.env.reset())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Passes in a state x through the network and returns the policy and a sampled action
        Args:
            x: environment state
        Returns:
            Tuple of policy and action
        """
        pi, action = self.agent(x)
        value = self.critic(x)
        return pi, action, value

    def discount_rewards(self, rewards: List[float], discount: float) -> List[float]:
        """Calculate the discounted rewards of all rewards in list
        Args:
            rewards: list of rewards/advantages
        Returns:
            list of discounted rewards/advantages
        """
        assert isinstance(rewards[0], float)

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * discount) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def calc_advantage(self, rewards: List[float], values: List[float], last_value: float) -> List[float]:
        """Calculate the advantage given rewards, state values, and the last value of episode
        Args:
            rewards: list of episode rewards
            values: list of state values from critic
            last_value: value of last state of episode
        Returns:
            list of advantages
        """
        rews = rewards + [last_value]
        vals = values + [last_value]
        # GAE
        delta = [rews[i] + self.gamma * vals[i+1] - vals[i] for i in range(len(rews)-1)]
        adv = self.discount_rewards(delta, self.gamma * self.lam)

        return adv

    def train_batch(
            self,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Contains the logic for generating trajectory data to train policy and value network
        Yield:
           Tuple of Lists containing tensors for states, actions, log probs, qvals and advantage
        """

        for step in range(self.steps_per_epoch):
            with torch.no_grad():
                pi, action = self.agent(self.state)
                log_prob = self.agent.get_log_prob(pi, action)
                value = self.critic(self.state)

            next_state, reward, done, _ = self.env.step(action.numpy())

            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_logp.append(log_prob)

            self.ep_rewards.append(reward)
            self.ep_values.append(value.item())

            self.state = torch.FloatTensor(next_state)

            epoch_end = step == (self.steps_per_epoch-1)
            terminal = len(self.ep_rewards) == self.max_episode_len

            if epoch_end or done or terminal:
                # if trajectory ends abtruptly, boostrap value of next state
                if (terminal or epoch_end) and not done:
                    with torch.no_grad():
                        last_value = self.critic(self.state).item()
                else:
                    last_value = 0

                # discounted cumulative reward
                self.batch_qvals += self.discount_rewards(self.ep_rewards+[last_value], self.gamma)[:-1]
                # advantage
                self.batch_adv += self.calc_advantage(self.ep_rewards, self.ep_values, last_value)
                # logs
                self.done_episodes += 1
                self.epoch_rewards+=np.sum(self.ep_rewards)
                # reset params
                self.ep_rewards = []
                self.ep_values = []
                self.state = torch.FloatTensor(self.env.reset())

            if epoch_end:
                train_data = zip(self.batch_states, self.batch_actions, self.batch_logp, self.batch_qvals, self.batch_adv)

                for state, action, logp_old, qval, adv in train_data:
                    yield state, action, logp_old, qval, adv

                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_adv.clear()
                self.batch_logp.clear()
                self.batch_qvals.clear()

                self.avg_ep_reward = self.epoch_rewards/self.done_episodes
                self.avg_reward = self.epoch_rewards/self.steps_per_epoch
                self.avg_ep_len = self.steps_per_epoch/self.done_episodes

                self.epoch_rewards = 0
                self.done_episodes = 0

    def actor_loss(self, state, action, logp_old, qval, adv) -> torch.Tensor:
        pi, _ = self.agent(state)
        logp = self.agent.get_log_prob(pi, action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_actor

    def critic_loss(self, state, action, logp_old, qval, adv) -> torch.Tensor:
        value = self.critic(state)
        loss_critic = (qval - value).pow(2).mean()
        return loss_critic

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx, optimizer_idx):
        """
        Carries out a single update to actor and critic network from a batch of replay buffer.

        Args:
            batch: batch of replay buffer/trajectory data
            batch_idx: not used
            optimizer_idx: idx that controls optimizing actor or critic network
        Returns:
            loss
        """
        state, action, old_logp, qval, adv = batch
        self.log("avg_ep_len", self.avg_ep_len, prog_bar=True, on_step=False, on_epoch=True)
        self.log("avg_ep_reward", self.avg_ep_reward, prog_bar=True, on_step=False, on_epoch=True)
        self.log("avg_reward", self.avg_reward, prog_bar=True, on_step=False, on_epoch=True)

        if optimizer_idx == 0:
            loss_actor = self.actor_loss(state, action, old_logp, qval, adv)
            self.log('loss_actor', loss_actor, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            return loss_actor

        elif optimizer_idx == 1:
            loss_critic = self.critic_loss(state, action, old_logp, qval, adv)
            self.log('loss_critic', loss_critic, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            return loss_critic

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer_actor = optim.Adam(self.agent.net.parameters(), lr=self.lr_actor)
        optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        return [optimizer_actor, optimizer_critic]

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
        Adds arguments for PPO model
        Args:
            arg_parser: the current argument parser to add to
        Returns:
            arg_parser with model specific cargs added
        """
        arg_parser.add_argument("--steps_per_epoch", type=int, default=2048, help="number of samples in an epoch")
        arg_parser.add_argument("--batch_size", type=int, default=512, help="size of the batches")

        arg_parser.add_argument("--env", type=str, required=True, help="gym environment tag")
        arg_parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")

        return arg_parser


def cli_main():
    parser = argparse.ArgumentParser(add_help=False)

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = PPO.add_model_specific_args(parser)
    args = parser.parse_args()

    model = PPO(**args.__dict__)

    # save checkpoints based on avg_reward
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="avg_ep_reward", mode="max",
        period=1, verbose=True
    )

    seed_everything(123)
    trainer = pl.Trainer.from_argparse_args(
        args, deterministic=True, checkpoint_callback=checkpoint_callback
    )
    trainer.fit(model)


if __name__ == '__main__':
    cli_main()
