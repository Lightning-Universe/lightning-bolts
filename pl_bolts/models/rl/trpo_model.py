import argparse
from collections import defaultdict, deque
from statistics import mean
from typing import Tuple, Any, Union, Callable

import gym
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import Tensor
from torch.distributions import Normal, Categorical
from torch.distributions.utils import clamp_probs
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pl_bolts.datamodules import ExperienceSourceDataset
from pl_bolts.models.rl.common.conjugate_gradient import conjugate_gradient
from pl_bolts.models.rl.common.kl_divergence import (
    kl_divergence_between_discrete_distributions,
    kl_divergence_between_continuous_distributions,
)
from pl_bolts.models.rl.common.networks import MLP, ActorCategorical, ActorContinous
from pl_bolts.models.rl.common.rewards import discount_rewards, calc_advantage
from pl_bolts.models.rl.common.running_statistics import ZFilter
from pl_bolts.utils import _GYM_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _GYM_AVAILABLE:
    pass
else:  # pragma: no cover
    warn_missing_pkg("gym")


class TRPO(LightningModule):
    """PyTorch Lightning implementation of `Trusted Region Policy Optimization.

    <https://arxiv.org/pdf/1502.05477.pdf>`_

    Paper authors: John Schulman, Sergey Levine, Philipp Moritz, Michael Jordan, Pieter Abbeel
    Model implemented by:
        `Kamil Pluciński <https://github.com/plutasnyy>`_
        `Iwo Naglik <https://github.com/NaIwo>`_

    Example:
        >>> from pl_bolts.models.rl.trpo_model import TRPO
        >>> model = TRPO("CartPole-v0")

    Train:
        >>> trainer = Trainer()
        >>> trainer.fit(model)

    Note:
        This example is based on the implementation from:
        Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO
        Paper authors: Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras,
        Firdaus Janoos, Larry Rudolph, Aleksander Madry
        <https://arxiv.org/abs/2005.12729>_

    """

    def __init__(
            self,
            env: str,
            gamma: float = 0.99,
            lam: float = 0.95,
            lr_critic: float = 1e-4,
            max_episode_len: int = 200,
            batch_size: int = 2048,
            kl_div_threshold: float = 0.07,
            cg_iters: int = 10,
            normalize_states: bool = True,
            **kwargs: Any,
    ):
        """
        Args:
            env: gym environment tag
            gamma: discount factor
            lam: advantage discount factor
            lr_critic: learning rate of critic network
            max_episode_len: maximum number interactions (actions) in an episode
            batch_size: number of steps (actions) performed during one 'epoch' - these batch is then used for several
                epochs of learning critic and one TRPO step
            kl_div_threshold: maximum allowed change in KL divergence during policy updates
            cg_iters: number of conjugate gradient iterations
            normalize_states: flag whether to normalize states using Z filter
        """

        super().__init__()
        self.gamma = gamma
        self.lam = lam
        self.lr_critic = lr_critic
        self.kl_div_threshold = kl_div_threshold
        self.batch_size = batch_size
        self.cg_iters = cg_iters
        self.normalize_states = normalize_states
        self.max_episode_len = max_episode_len

        self.save_hyperparameters()

        self.env = gym.make(env)
        self.state_dim = self.env.observation_space.shape
        self.critic = self.get_critic_net()

        self.actor = None
        self.kl_div = None
        self.get_distribution = None
        self.action_dim = None
        if isinstance(self.env.action_space, gym.spaces.box.Box):
            self._initialize_continuous_env_strategy()
        elif isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self._initialize_discrete_env_strategy()
        else:
            raise NotImplementedError(
                "Env action space should be of type Box (continous) or Discrete (categorical). Got type: "
                "{type(self.env.action_space)}"
            )

        self.state_filter = ZFilter(shape=self.state_dim)
        self.last_100_returns = deque(maxlen=100)
        self.batch_log_dict = {}

    def get_critic_net(self) -> Module:
        """
        Override this method if you would like to use a different network than the MLP (e.g. CNN) as a critic.
        """
        return MLP(self.state_dim, 1)

    def get_actor_net(self) -> Module:
        """
        Override this method if you would like to use a different network than the MLP (e.g. CNN) as an actor.
        """
        return MLP(self.state_dim, self.action_dim)

    def _initialize_continuous_env_strategy(self) -> None:
        self.action_dim = self.env.action_space.shape[0]
        self.actor = ActorContinous(actor_net=self.get_actor_net(), act_dim=self.action_dim)
        self.kl_div = kl_divergence_between_continuous_distributions
        self.get_distribution = self._distribution_from_continuous_input

    def _initialize_discrete_env_strategy(self) -> None:
        self.action_dim = self.env.action_space.n
        self.actor = ActorCategorical(actor_net=self.get_actor_net())
        self.kl_div = kl_divergence_between_discrete_distributions
        self.get_distribution = self._distribution_from_categorical_input

    def train_dataloader(self) -> DataLoader:
        dataset = ExperienceSourceDataset(self.generate_trajectory_samples)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader

    @torch.no_grad()
    def generate_trajectory_samples(self) -> Tuple[Tensor, Tensor, float, float, float]:
        """
        Generates one batch, when episode length exceeds the size of batch_size the last value is bootstrapped

        Yields:
            Tuple of:
                State: Tensor
                Action: Tensor
                Action log probability: float
                G - discounted return to the end of episode: float,
                Advantage: float
        """

        state = torch.FloatTensor(self.env.reset())
        state = self.state_filter(state)

        batch_data = defaultdict(list)

        episode_rewards = []
        episode_values = []
        episode_step = 0

        for epoch_step in range(self.batch_size):
            state = state.to(device=self.device)

            pi, action, value = self(state)
            action_log_prob = self.actor.get_log_prob(pi, action).item()

            next_state, reward, done, _ = self.env.step(action.cpu().numpy())
            if self.normalize_states:
                next_state = self.state_filter(next_state)

            episode_step += 1

            batch_data["states"].append(state)
            batch_data["actions"].append(action)
            batch_data["action_log_probs"].append(action_log_prob)

            episode_rewards.append(reward)
            episode_values.append(value.item())

            state = torch.FloatTensor(next_state)

            terminal = episode_step == self.max_episode_len
            epoch_end = epoch_step == (self.batch_size - 1)

            if done or terminal or epoch_end:

                cumulated_reward = sum(episode_rewards)
                batch_data["returns"].append(cumulated_reward)

                if not epoch_end:
                    batch_data["episode_lengths"].append(episode_step)
                    self.last_100_returns.append(cumulated_reward)

                # if trajectory ends abruptly, boostrap value of next state
                last_value = 0 if done else self.critic(state.to(self.device)).item()

                discounted_returns = discount_rewards(episode_rewards + [last_value], discount=self.gamma)
                batch_data["discounted_returns"].extend(discounted_returns[:-1])

                advantages = calc_advantage(episode_rewards, episode_values, last_value, gamma=self.gamma, lam=self.lam)
                batch_data["advantages"].extend(advantages)

                state = torch.FloatTensor(self.env.reset())

                episode_rewards = []
                episode_values = []
                episode_step = 0

        self.batch_log_dict = {
            "epoch_mean_return": mean(batch_data["returns"]),
            "epoch_mean_episode_length": mean(batch_data["episode_lengths"]),
            "last_100_returns": mean(self.last_100_returns),
        }

        fields_to_yield = ["states", "actions", "action_log_probs", "discounted_returns", "advantages"]
        lists_yo_yield = [batch_data[key] for key in fields_to_yield]
        for state, action, action_log_prob, discounted_return, advantage in zip(*lists_yo_yield):
            yield state, action, action_log_prob, discounted_return, advantage

    def forward(self, state: Tensor) -> Tuple[Union[Categorical, Normal], Tensor, Tensor]:
        """
        Args:
            state: Tensor representing the state

        Returns:
            pi: Predicted distribution, instance of Categorical or Normal
            action: Tensor representing chosen action
            value: Tensor with value estimated by critic
        """
        pi, action = self.actor(state.float())
        value = self.critic(state.float())
        return pi, action, value

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int) -> None:
        """
        Performs 10 epochs of updates for critic network and single TRPO step for actor network

        Args:
            batch: batch of trajectory data
            batch_idx: not used

        Returns:
            None - loss is redundant as all updates are performed inside the function
        """
        old_states, old_actions, old_log_action_probability, old_discounted_return, old_advantages = batch

        # Start the optimization of performing 10 epochs of critic network training on trajectories in batch data.
        # The trajectories are randomly split into smaller batches of size 32 in each epoch.
        value_net_optimizer = self.optimizers()
        for _ in range(10):
            state_indices = np.arange(old_discounted_return.nelement())
            np.random.shuffle(state_indices)
            splits = np.array_split(state_indices, 32)

            for indexes_of_states_in_batch in splits:
                value_net_optimizer.zero_grad()
                current_values = self.critic(old_states[indexes_of_states_in_batch].float()).squeeze(-1)
                value_loss = (current_values - old_discounted_return[indexes_of_states_in_batch]).pow(2).mean()
                self.log("loss_critic", value_loss.item())
                value_loss.backward()
                value_net_optimizer.step()

        normalized_advantages = (old_advantages - old_advantages.mean()) / (old_advantages.std() + 1e-8)

        # Save the state of the actor to undo changes made during the backtracking search
        # if a better policy is not found.
        initial_actor_parameters = self._flatten(self.actor.parameters()).clone()

        # Perform a forward phase with the actor to determine the direction of the gradients in which
        # we will look to improve policy
        pi, _ = self.actor(old_states.float())  # TODO check description
        log_action_probabilities = self.actor.get_log_prob(pi, old_actions)

        surrogate_reward = self.surrogate_reward(
            old_log_action_probability, log_action_probabilities, normalized_advantages
        )
        actor_gradient = torch.autograd.grad(surrogate_reward, self.actor.parameters(), retain_graph=True)
        actor_surrogate_gradient = self._flatten(actor_gradient)

        # Calculate the KL divergence to determine the direction of the gradient improvement
        distribution = self.get_distribution(pi)
        kl = self.kl_div(distribution, distribution)
        g = self._flatten(torch.autograd.grad(kl, self.actor.parameters(), create_graph=True))

        def fisher_product(x):
            contig_flat = lambda q: torch.cat([y.contiguous().view(-1) for y in q])
            hv = torch.autograd.grad(g @ x, self.actor.parameters(), retain_graph=True)
            return contig_flat(hv).detach() + x * 0.1

        step_direction = conjugate_gradient(fisher_product, actor_surrogate_gradient, max_iterations=self.cg_iters)

        # Determine the maximum step of change
        max_length = torch.sqrt(
            2 * self.kl_div_threshold / (step_direction @ fisher_product(step_direction))
        )
        max_step = max_length * step_direction

        with torch.no_grad():

            def backtrack_fn(params_update: Tensor) -> float:
                """
                Tries to update actor in some direction and verifies whether the new policy meets KL constraint
                Args:
                    params_update: Direction in which the actor weights should be updated

                Returns:
                    Difference in surrogate reward if new policy satisfies KL constraint, -inf otherwise

                """
                self.update_actor_weights(initial_actor_parameters + params_update.data)
                new_pi, _ = self.actor(old_states.float())
                new_action_log_probs = self.actor.get_log_prob(new_pi, old_actions)

                new_surrogate_reward = self.surrogate_reward(
                    old_log_action_probability, new_action_log_probs, normalized_advantages
                )

                new_distribution = self.get_distribution(new_pi)
                kl = self.kl_div(distribution, new_distribution)

                if kl > self.kl_div_threshold or new_surrogate_reward <= surrogate_reward:
                    return -float("inf")

                return new_surrogate_reward - surrogate_reward

            expected_improve = actor_surrogate_gradient @ max_step
            final_step = self._backtracking_line_search(backtrack_fn, max_step, expected_improve, num_tries=10)
            self.update_actor_weights(initial_actor_parameters + final_step)

        self.log_dict(self.batch_log_dict, prog_bar=True, logger=True, on_step=True)

    @staticmethod
    def surrogate_reward(old_log_probabilities: Tensor, new_log_probabilities: Tensor, advantages: Tensor) -> float:
        """
        Computes surrogate reward R = E[r_t * A_t]
        """
        loss = torch.exp(new_log_probabilities - old_log_probabilities) * advantages
        return loss.mean()

    def update_actor_weights(self, new_weights: Tensor) -> None:
        """
        Assigns parameters from flatten vector of weights to actor network

        Args:
            new_weights: Flat tensor of weights

        Returns:
            None
        """
        pointer = 0
        for param in self.actor.parameters():
            numel = param.numel()
            param.data = new_weights[pointer: pointer + numel].view(param.shape).data
            pointer += numel

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic, eps=1e-5)

    @staticmethod
    def _flatten(g: Union[Tensor, Tuple[Tensor]]) -> Tensor:
        return torch.cat([t.view(-1) for t in g])

    @staticmethod
    def _distribution_from_categorical_input(pi: Categorical) -> Tensor:
        assert isinstance(pi, Categorical)
        return clamp_probs(pi.probs)

    @staticmethod
    def _distribution_from_continuous_input(pi: Normal) -> Tuple[Tensor, Tensor]:
        assert isinstance(pi, Normal)
        mean, std = pi.mean, pi.stddev[0]
        return mean, std

    @staticmethod
    def _backtracking_line_search(
            f: Callable, x: Any, expected_improve_rate: float, num_tries: int = 10, accept_ratio: float = 0.1
    ) -> Tensor:
        """
        Performs Backtracking Line Search algorithm. In each iteration of the algorithm, the step is halved,
        and then it is verified whether the new policy meets the KL constraint.

        Args:
            f: Objective function
            x: Step direction
            expected_improve_rate: Expected improve rate
            num_tries: Number of iterations
            accept_ratio: How much of the expected improvement rate we have to improve by

        Returns:
            Improvement step satisfying KL constraint, or tensor of zeros if the constraint is not satisfied
        """
        for i in range(num_tries):
            scaling = 2 ** (-i)
            scaled = x * scaling
            improve = f(scaled)
            expected_improve = expected_improve_rate * scaling
            if improve / expected_improve > accept_ratio and improve > 0:
                return scaled
        return torch.zeros_like(x)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument("--env", type=str, default="CartPole-v0", help="Gym environment tag")
        parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
        parser.add_argument("--lam", type=float, default=0.95, help="Advantage discount factor")
        parser.add_argument("--lr-critic", type=float, default=1e-4, help="Learning rate of critic network")
        parser.add_argument("--max-episode-len", type=int, default=200, help="Capacity of the replay buffer")
        parser.add_argument(
            "--batch-size", type=int, default=2048, help="Number of agent actions performed during one epoch"
        )
        parser.add_argument(
            "--kl-div-threshold",
            type=float,
            default=0.05,
            help="Maximum allowed change in KL divergence during policy updates",
        )
        parser.add_argument("--cg-iters", type=int, default=10, help="Number of conjugate gradient iterations")

        normalize_states_parser = parser.add_mutually_exclusive_group(required=False)
        normalize_states_parser.add_argument('--normalize-states', dest='normalize_states', action='store_true',
                                             help="Normalize states using Z filter")
        normalize_states_parser.add_argument('--no-normalize-states', dest='normalize_states', action='store_false',
                                             help="Do not normalize states using Z filter")
        parser.set_defaults(normalize_states=True)
        return parser


def cli_main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = Trainer.add_argparse_args(parent_parser)

    parser = TRPO.add_model_specific_args(parent_parser)
    args = parser.parse_args()
    model = TRPO(**vars(args))

    seed_everything(0)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == "__main__":
    cli_main()
