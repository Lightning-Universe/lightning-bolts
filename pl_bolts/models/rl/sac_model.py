"""Soft Actor Critic."""
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pl_bolts.datamodules.experience_source import Experience, ExperienceSourceDataset
from pl_bolts.models.rl.common.agents import SoftActorCriticAgent
from pl_bolts.models.rl.common.memory import MultiStepBuffer
from pl_bolts.models.rl.common.networks import MLP, ContinuousMLP
from pl_bolts.utils import _GYM_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _GYM_AVAILABLE:
    import gym
else:  # pragma: no cover
    warn_missing_pkg("gym")
    Env = object


class SAC(LightningModule):
    def __init__(
        self,
        env: str,
        eps_start: float = 1.0,
        eps_end: float = 0.02,
        eps_last_frame: int = 150000,
        sync_rate: int = 1,
        gamma: float = 0.99,
        policy_learning_rate: float = 3e-4,
        q_learning_rate: float = 3e-4,
        target_alpha: float = 5e-3,
        batch_size: int = 128,
        replay_size: int = 1000000,
        warm_start_size: int = 10000,
        avg_reward_len: int = 100,
        min_episode_reward: int = -21,
        seed: int = 123,
        batches_per_epoch: int = 10000,
        n_steps: int = 1,
        **kwargs,
    ):
        super().__init__()

        # Environment
        self.env = gym.make(env)
        self.test_env = gym.make(env)

        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.shape[0]

        # Model Attributes
        self.buffer = None
        self.dataset = None

        self.policy = None
        self.q1 = None
        self.q2 = None
        self.target_q1 = None
        self.target_q2 = None
        self.build_networks()

        self.agent = SoftActorCriticAgent(self.policy)

        # Hyperparameters
        self.save_hyperparameters()

        # Metrics
        self.total_episode_steps = [0]
        self.total_rewards = [0]
        self.done_episodes = 0
        self.total_steps = 0

        # Average Rewards
        self.avg_reward_len = avg_reward_len

        for _ in range(avg_reward_len):
            self.total_rewards.append(torch.tensor(min_episode_reward, device=self.device))

        self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))

        self.state = self.env.reset()

        self.automatic_optimization = False

    def run_n_episodes(self, env, n_epsiodes: int = 1) -> List[int]:
        """Carries out N episodes of the environment with the current agent without exploration.

        Args:
            env: environment to use, either train environment or test environment
            n_epsiodes: number of episodes to run
        """
        total_rewards = []

        for _ in range(n_epsiodes):
            episode_state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.agent.get_action(episode_state, self.device)
                next_state, reward, done, _ = env.step(action[0])
                episode_state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

        return total_rewards

    def populate(self, warm_start: int) -> None:
        """Populates the buffer with initial experience."""
        if warm_start > 0:
            self.state = self.env.reset()

            for _ in range(warm_start):
                action = self.agent(self.state, self.device)
                next_state, reward, done, _ = self.env.step(action[0])
                exp = Experience(state=self.state, action=action[0], reward=reward, done=done, new_state=next_state)
                self.buffer.append(exp)
                self.state = next_state

                if done:
                    self.state = self.env.reset()

    def build_networks(self) -> None:
        """Initializes the SAC policy and q networks (with targets)"""
        action_bias = torch.from_numpy((self.env.action_space.high + self.env.action_space.low) / 2)
        action_scale = torch.from_numpy((self.env.action_space.high - self.env.action_space.low) / 2)
        self.policy = ContinuousMLP(self.obs_shape, self.n_actions, action_bias=action_bias, action_scale=action_scale)

        concat_shape = [self.obs_shape[0] + self.n_actions]
        self.q1 = MLP(concat_shape, 1)
        self.q2 = MLP(concat_shape, 1)
        self.target_q1 = MLP(concat_shape, 1)
        self.target_q2 = MLP(concat_shape, 1)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

    def soft_update_target(self, q_net, target_net):
        """Update the weights in target network using a weighted sum.

        w_target := (1-a) * w_target + a * w_q

        Args:
            q_net: the critic (q) network
            target_net: the target (q) network
        """
        for q_param, target_param in zip(q_net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                (1.0 - self.hparams.target_alpha) * target_param.data + self.hparams.target_alpha * q_param
            )

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.policy(x).sample()
        return output

    def train_batch(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.

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

            exp = Experience(state=self.state, action=action[0], reward=r, done=is_done, new_state=next_state)

            self.buffer.append(exp)
            self.state = next_state

            if is_done:
                self.done_episodes += 1
                self.total_rewards.append(episode_reward)
                self.total_episode_steps.append(episode_steps)
                self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))
                self.state = self.env.reset()
                episode_steps = 0
                episode_reward = 0

            states, actions, rewards, dones, new_states = self.buffer.sample(self.hparams.batch_size)

            for idx, _ in enumerate(dones):
                yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]

            # Simulates epochs
            if self.total_steps % self.hparams.batches_per_epoch == 0:
                break

    def loss(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Calculates the loss for SAC which contains a total of 3 losses.

        Args:
            batch: a batch of states, actions, rewards, dones, and next states
        """
        states, actions, rewards, dones, next_states = batch
        rewards = rewards.unsqueeze(-1)
        dones = dones.float().unsqueeze(-1)

        # actor
        dist = self.policy(states)
        new_actions, new_logprobs = dist.rsample_and_log_prob()
        new_logprobs = new_logprobs.unsqueeze(-1)

        new_states_actions = torch.cat((states, new_actions), 1)
        new_q1_values = self.q1(new_states_actions)
        new_q2_values = self.q2(new_states_actions)
        new_qmin_values = torch.min(new_q1_values, new_q2_values)

        policy_loss = (new_logprobs - new_qmin_values).mean()

        # critic
        states_actions = torch.cat((states, actions), 1)
        q1_values = self.q1(states_actions)
        q2_values = self.q2(states_actions)

        with torch.no_grad():
            next_dist = self.policy(next_states)
            new_next_actions, new_next_logprobs = next_dist.rsample_and_log_prob()
            new_next_logprobs = new_next_logprobs.unsqueeze(-1)

            new_next_states_actions = torch.cat((next_states, new_next_actions), 1)
            next_q1_values = self.target_q1(new_next_states_actions)
            next_q2_values = self.target_q2(new_next_states_actions)
            next_qmin_values = torch.min(next_q1_values, next_q2_values) - new_next_logprobs
            target_values = rewards + (1.0 - dones) * self.hparams.gamma * next_qmin_values

        q1_loss = F.mse_loss(q1_values, target_values)
        q2_loss = F.mse_loss(q2_values, target_values)

        return policy_loss, q1_loss, q2_loss

    def training_step(self, batch: Tuple[Tensor, Tensor], _):
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            _: batch number, not used
        """
        policy_optim, q1_optim, q2_optim = self.optimizers()
        policy_loss, q1_loss, q2_loss = self.loss(batch)

        policy_optim.zero_grad()
        self.manual_backward(policy_loss)
        policy_optim.step()

        q1_optim.zero_grad()
        self.manual_backward(q1_loss)
        q1_optim.step()

        q2_optim.zero_grad()
        self.manual_backward(q2_loss)
        q2_optim.step()

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.soft_update_target(self.q1, self.target_q1)
            self.soft_update_target(self.q2, self.target_q2)

        self.log_dict(
            {
                "total_reward": self.total_rewards[-1],
                "avg_reward": self.avg_rewards,
                "policy_loss": policy_loss,
                "q1_loss": q1_loss,
                "q2_loss": q2_loss,
                "episodes": self.done_episodes,
                "episode_steps": self.total_episode_steps[-1],
            }
        )

    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Evaluate the agent for 10 episodes."""
        test_reward = self.run_n_episodes(self.test_env, 1)
        avg_reward = sum(test_reward) / len(test_reward)
        return {"test_reward": avg_reward}

    def test_epoch_end(self, outputs) -> Dict[str, Tensor]:
        """Log the avg of the test results."""
        rewards = [x["test_reward"] for x in outputs]
        avg_reward = sum(rewards) / len(rewards)
        self.log("avg_test_reward", avg_reward)
        return {"avg_test_reward": avg_reward}

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        self.buffer = MultiStepBuffer(self.hparams.replay_size, self.hparams.n_steps)
        self.populate(self.hparams.warm_start_size)

        self.dataset = ExperienceSourceDataset(self.train_batch)
        return DataLoader(dataset=self.dataset, batch_size=self.hparams.batch_size)

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()

    def test_dataloader(self) -> DataLoader:
        """Get test loader."""
        return self._dataloader()

    def configure_optimizers(self) -> Tuple[Optimizer]:
        """Initialize Adam optimizer."""
        policy_optim = optim.Adam(self.policy.parameters(), self.hparams.policy_learning_rate)
        q1_optim = optim.Adam(self.q1.parameters(), self.hparams.q_learning_rate)
        q2_optim = optim.Adam(self.q2.parameters(), self.hparams.q_learning_rate)
        return policy_optim, q1_optim, q2_optim

    @staticmethod
    def add_model_specific_args(
        arg_parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Adds arguments for DQN model.

        Note:
            These params are fine tuned for Pong env.

        Args:
            arg_parser: parent parser
        """
        arg_parser.add_argument(
            "--sync_rate",
            type=int,
            default=1,
            help="how many frames do we update the target network",
        )
        arg_parser.add_argument(
            "--replay_size",
            type=int,
            default=1000000,
            help="capacity of the replay buffer",
        )
        arg_parser.add_argument(
            "--warm_start_size",
            type=int,
            default=10000,
            help="how many samples do we use to fill our buffer at the start of training",
        )
        arg_parser.add_argument("--batches_per_epoch", type=int, default=10000, help="number of batches in an epoch")
        arg_parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
        arg_parser.add_argument("--policy_lr", type=float, default=3e-4, help="policy learning rate")
        arg_parser.add_argument("--q_lr", type=float, default=3e-4, help="q learning rate")
        arg_parser.add_argument("--env", type=str, required=True, help="gym environment tag")
        arg_parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")

        arg_parser.add_argument(
            "--avg_reward_len",
            type=int,
            default=100,
            help="how many episodes to include in avg reward",
        )
        arg_parser.add_argument(
            "--n_steps",
            type=int,
            default=1,
            help="how many frames do we update the target network",
        )

        return arg_parser


def cli_main():
    parser = argparse.ArgumentParser(add_help=False)

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = SAC.add_model_specific_args(parser)
    args = parser.parse_args()

    model = SAC(**args.__dict__)

    # save checkpoints based on avg_reward
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="avg_reward", mode="max", verbose=True)

    seed_everything(123)
    trainer = Trainer.from_argparse_args(args, deterministic=True, callbacks=checkpoint_callback)

    trainer.fit(model)


if __name__ == "__main__":
    cli_main()
