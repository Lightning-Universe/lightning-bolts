"""
Datamodules for RL models that rely on experiences generated during training
Based on implementations found here: https://github.com/Shmuma/ptan/blob/master/ptan/experience.py
"""
from abc import ABC
from collections import deque, namedtuple
from typing import Callable, Iterable, List, Tuple

import torch
from torch.utils.data import IterableDataset

from pl_bolts.utils import _GYM_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _GYM_AVAILABLE:
    from gym import Env
else:  # pragma: no cover
    warn_missing_pkg("gym")
    Env = object

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "done", "new_state"])


class ExperienceSourceDataset(IterableDataset):
    """
    Basic experience source dataset. Takes a generate_batch function that returns an iterator.
    The logic for the experience source and how the batch is generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable) -> None:
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterable:
        iterator = self.generate_batch()
        return iterator


# Experience Sources
class BaseExperienceSource(ABC):
    """
    Simplest form of the experience source
    """

    def __init__(self, env, agent) -> None:
        """
        Args:
            env: Environment that is being used
            agent: Agent being used to make decisions
        """
        self.env = env
        self.agent = agent

    def runner(self) -> Experience:
        """Iterable method that yields steps from the experience source"""
        raise NotImplementedError("ExperienceSource has no stepper method implemented")


class ExperienceSource(BaseExperienceSource):
    """
    Experience source class handling single and multiple environment steps
    """

    def __init__(self, env, agent, n_steps: int = 1) -> None:
        """
        Args:
            env: Environment that is being used
            agent: Agent being used to make decisions
            n_steps: Number of steps to return from each environment at once
        """
        super().__init__(env, agent)

        self.pool = env if isinstance(env, (list, tuple)) else [env]
        self.exp_history_queue = deque()

        self.n_steps = n_steps
        self.total_steps = []
        self.states = []
        self.histories = []
        self.cur_rewards = []
        self.cur_steps = []
        self.iter_idx = 0

        self._total_rewards = []

        self.init_environments()

    def runner(self, device: torch.device) -> Tuple[Experience]:
        """Experience Source iterator yielding Tuple of experiences for n_steps. These come from the pool
        of environments provided by the user.

        Args:
            device: current device to be used for executing experience steps

        Returns:
            Tuple of Experiences
        """
        while True:
            # get actions for all envs
            actions = self.env_actions(device)

            # step through each env
            for env_idx, (env, action) in enumerate(zip(self.pool, actions)):

                exp = self.env_step(env_idx, env, action)
                history = self.histories[env_idx]
                history.append(exp)
                self.states[env_idx] = exp.new_state

                self.update_history_queue(env_idx, exp, history)

                # Yield all accumulated history tuples to model
                while self.exp_history_queue:
                    yield self.exp_history_queue.popleft()

            self.iter_idx += 1

    def update_history_queue(self, env_idx, exp, history) -> None:
        """
        Updates the experience history queue with the lastest experiences. In the event of an experience step is in
        the done state, the history will be incrementally appended to the queue, removing the tail of the history
        each time.

        Args:
            env_idx: index of the environment
            exp: the current experience
            history: history of experience steps for this environment
        """
        # If there is a full history of step, append history to queue
        if len(history) == self.n_steps:
            self.exp_history_queue.append(tuple(history))

        if exp.done:
            if 0 < len(history) < self.n_steps:
                self.exp_history_queue.append(tuple(history))

            # generate tail of history, incrementally append history to queue
            while len(history) > 2:
                history.popleft()
                self.exp_history_queue.append(tuple(history))

            # when there are only 2 experiences left in the history,
            # append to the queue then update the env stats and reset the environment
            if len(history) > 1:
                self.update_env_stats(env_idx)

                history.popleft()
                self.exp_history_queue.append(tuple(history))

            # Clear that last tail in the history once all others have been added to the queue
            history.clear()

    def init_environments(self) -> None:
        """
        For each environment in the pool setups lists for tracking history of size n, state, current reward and
        current step
        """
        for env in self.pool:
            self.states.append(env.reset())
            self.histories.append(deque(maxlen=self.n_steps))
            self.cur_rewards.append(0.0)
            self.cur_steps.append(0)

    def env_actions(self, device) -> List[List[int]]:
        """
        For each environment in the pool, get the correct action
        Returns:
            List of actions for each env, with size (num_envs, action_size)
        """
        actions = []
        states_actions = self.agent(self.states, device)

        assert len(self.states) == len(states_actions)

        for idx, action in enumerate(states_actions):
            actions.append(action if isinstance(action, list) else [action])

        return actions

    def env_step(self, env_idx: int, env: Env, action: List[int]) -> Experience:
        """
        Carries out a step through the given environment using the given action

        Args:
            env_idx: index of the current environment
            env: env at index env_idx
            action: action for this environment step

        Returns:
            Experience tuple
        """
        next_state, r, is_done, _ = env.step(action[0])

        self.cur_rewards[env_idx] += r
        self.cur_steps[env_idx] += 1

        exp = Experience(state=self.states[env_idx], action=action[0], reward=r, done=is_done, new_state=next_state)

        return exp

    def update_env_stats(self, env_idx: int) -> None:
        """
        To be called at the end of the history tail generation during the termination state. Updates the stats
        tracked for all environments

        Args:
            env_idx: index of the environment used to update stats
        """
        self._total_rewards.append(self.cur_rewards[env_idx])
        self.total_steps.append(self.cur_steps[env_idx])
        self.cur_rewards[env_idx] = 0
        self.cur_steps[env_idx] = 0
        self.states[env_idx] = self.pool[env_idx].reset()

    def pop_total_rewards(self) -> List[float]:
        """
        Returns the list of the current total rewards collected
        Returns:
            list of total rewards for all completed episodes for each environment since last pop
        """
        rewards = self._total_rewards

        if rewards:
            self._total_rewards = []
            self.total_steps = []

        return rewards

    def pop_rewards_steps(self):
        """
        Returns the list of the current total rewards and steps collected
        Returns:
            list of total rewards and steps for all completed episodes for each environment since last pop
        """
        res = list(zip(self._total_rewards, self.total_steps))
        if res:
            self._total_rewards, self.total_steps = [], []
        return res


class DiscountedExperienceSource(ExperienceSource):
    """Outputs experiences with a discounted reward over N steps"""

    def __init__(self, env: Env, agent, n_steps: int = 1, gamma: float = 0.99) -> None:
        super().__init__(env, agent, (n_steps + 1))
        self.gamma = gamma
        self.steps = n_steps

    def runner(self, device: torch.device) -> Experience:
        """
        Iterates through experience tuple and calculate discounted experience

        Args:
            device: current device to be used for executing experience steps

        Yields:
            Discounted Experience
        """
        for experiences in super().runner(device):
            last_exp_state, tail_experiences = self.split_head_tail_exp(experiences)

            total_reward = self.discount_rewards(tail_experiences)

            yield Experience(
                state=experiences[0].state,
                action=experiences[0].action,
                reward=total_reward,
                done=experiences[0].done,
                new_state=last_exp_state
            )

    def split_head_tail_exp(self, experiences: Tuple[Experience]) -> Tuple[List, Tuple[Experience]]:
        """
        Takes in a tuple of experiences and returns the last state and tail experiences based on
        if the last state is the end of an episode

        Args:
            experiences: Tuple of N Experience

        Returns:
            last state (Array or None) and remaining Experience
        """
        if experiences[-1].done and len(experiences) <= self.steps:
            last_exp_state = experiences[-1].new_state
            tail_experiences = experiences
        else:
            last_exp_state = experiences[-1].state
            tail_experiences = experiences[:-1]
        return last_exp_state, tail_experiences

    def discount_rewards(self, experiences: Tuple[Experience]) -> float:
        """
        Calculates the discounted reward over N experiences

        Args:
            experiences: Tuple of Experience

        Returns:
            total discounted reward
        """
        total_reward = 0.0
        for exp in reversed(experiences):
            total_reward = (self.gamma * total_reward) + exp.reward  # type: ignore[attr-defined]
        return total_reward
