"""
Datamodules for RL models that rely on experiences generated during training

Based on implementations found here: https://github.com/Shmuma/ptan/blob/master/ptan/experience.py
"""
from abc import ABC, abstractmethod
from collections import deque, namedtuple
from typing import Iterable, Callable, List, Deque, Tuple

import torch
from gym import Env
from torch.utils.data import IterableDataset

# Datasets

Experience = namedtuple(
    "Experience", field_names=["state", "action", "reward", "done", "new_state"]
)


class ExperienceSourceDataset(IterableDataset):
    """
    Basic experience source dataset. Takes a generate_batch function that returns an iterator.
    The logic for the experience source and how the batch is generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable):
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterable:
        iterator = self.generate_batch()
        return iterator


# Experience Sources
class BaseExperienceSource(ABC):
    """
    Simplest form of the experience source

    Args:
        env: Environment that is being used
        agent: Agent being used to make decisions
    """

    def __init__(self, env, agent) -> None:
        self.env = env
        self.agent = agent

    @abstractmethod
    def __iter__(self) -> Experience:
        raise NotImplementedError("ExperienceSource has no __iter__ method implemented")


class ExperienceSource(BaseExperienceSource):
    """
    Experience source class handling single and multiple environment steps

    Args:
        env: Environment that is being used
        agent: Agent being used to make decisions
        n_steps: Number of steps to return from each environment at once
    """

    def __init__(self, env, agent, n_steps: int = 1) -> None:
        super().__init__(env, agent)

        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]

        self.n_steps = n_steps
        self.total_rewards = []
        self.total_steps = []
        self.states = []
        self.histories = []
        self.cur_rewards = []
        self.cur_steps = []
        self.iter_idx = 0

        self.init_envs()

    def init_envs(self) -> None:
        """
        For each environment in the pool setups lists for tracking history of size n, state, current reward and
        current step
        """
        for env in self.pool:
            self.states.append(env.reset())
            self.histories.append(deque(maxlen=self.n_steps))
            self.cur_rewards.append(0.0)
            self.cur_steps.append(0)

    def env_actions(self) -> List[List[int]]:
        """
        For each environment in the pool, get the correct action

        Returns:
            List of actions for each env, with size (num_envs, action_size)
        """
        actions = [None] * len(self.states)
        states_actions = self.agent(self.states)

        for idx, action in enumerate(states_actions):
            actions[idx] = action if isinstance(action, list) else [action]

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

        self.cur_rewards[env_idx] += 1
        self.cur_steps[env_idx] += 1

        exp = Experience(state=self.states[env_idx], action=action[0], reward=r, done=is_done, new_state=next_state)

        return exp

    def __iter__(self) -> Tuple[Experience]:
        """Experience Source iterator yielding Tuple of experiences for n_steps. These come from the pool
        of environments provided by the user.

        Returns:
            Tuple of Experiences
        """
        while True:

            # get actions for all envs
            actions = self.env_actions()

            # step through each env
            for env_idx, (env, action) in enumerate(zip(self.pool, actions)):

                exp = self.env_step(env_idx, env, action)
                history = self.histories[env_idx]
                history.append(exp)

                self.states[env_idx] = exp.new_state

                if len(history) == self.n_steps:
                    yield self.history

