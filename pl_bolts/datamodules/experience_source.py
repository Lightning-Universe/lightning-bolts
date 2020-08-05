"""
Datamodules for RL models that rely on experiences generated during training

Based on implementations found here: https://github.com/Shmuma/ptan/blob/master/ptan/experience.py
"""
from collections import deque, namedtuple
from typing import Iterable, Callable, Tuple, List

import numpy as np
import torch
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


class ExperienceSource(object):
    """
    Basic single step experience source

    Args:
        env: Environment that is being used
        agent: Agent being used to make decisions
    """

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.state = self.env.reset()

    def _reset(self) -> None:
        """resets the env and state"""
        self.state = self.env.reset()

    def step(self, device: torch.device) -> Tuple[Experience, float, bool]:
        """Takes a single step through the environment"""
        action = self.agent(self.state, device)
        new_state, reward, done, _ = self.env.step(action)
        experience = Experience(
            state=self.state,
            action=action,
            reward=reward,
            new_state=new_state,
            done=done,
        )
        self.state = new_state

        if done:
            self.state = self.env.reset()

        return experience, reward, done

    def run_episode(self, device: torch.device) -> float:
        """Carries out a single episode and returns the total reward. This is used for testing"""
        done = False
        total_reward = 0

        while not done:
            _, reward, done = self.step(device)
            total_reward += reward

        return total_reward


class NStepExperienceSource(ExperienceSource):
    """Expands upon the basic ExperienceSource by collecting experience across N steps"""

    def __init__(self, env, agent, n_steps: int = 1, gamma: float = 0.99):
        super().__init__(env, agent)
        self.gamma = gamma
        self.n_steps = n_steps
        self.n_step_buffer = deque(maxlen=n_steps)

    def step(self, device: torch.device) -> Tuple[Experience, float, bool]:
        """
        Takes an n-step in the environment

        Returns:
            Experience
        """
        exp = self.n_step(device)

        while len(self.n_step_buffer) < self.n_steps:
            self.n_step(device)

        reward, next_state, done = self.get_transition_info()
        first_experience = self.n_step_buffer[0]
        multi_step_experience = Experience(
            first_experience.state, first_experience.action, reward, done, next_state
        )

        return multi_step_experience, exp.reward, exp.done

    def n_step(self, device: torch.device) -> Experience:
        """
        Takes a  single step in the environment and appends it to the n-step buffer

        Returns:
            Experience
        """
        exp, _, _ = super().step(device)
        self.n_step_buffer.append(exp)
        return exp

    def get_transition_info(self) -> Tuple[np.float, np.array, np.int]:
        """
        get the accumulated transition info for the n_step_buffer
        Args:
            gamma: discount factor

        Returns:
            multi step reward, final observation and done
        """
        last_experience = self.n_step_buffer[-1]
        final_state = last_experience.new_state
        done = last_experience.done
        reward = last_experience.reward

        # calculate reward
        # in reverse order, go through all the experiences up till the first experience
        for experience in reversed(list(self.n_step_buffer)[:-1]):
            reward_t = experience.reward
            new_state_t = experience.new_state
            done_t = experience.done

            reward = reward_t + self.gamma * reward * (1 - done_t)
            final_state, done = (new_state_t, done_t) if done_t else (final_state, done)

        return reward, final_state, done


class EpisodicExperienceStream(ExperienceSource, IterableDataset):
    """
    Basic experience stream that iteratively yield the current experience of the agent in the env

    Args:
        env: Environmen that is being used
        agent: Agent being used to make decisions
    """

    def __init__(self, env, agent, device: torch.device, episodes: int = 1):
        super().__init__(env, agent)
        self.episodes = episodes
        self.device = device

    def __getitem__(self, item):
        return item

    def __iter__(self) -> List[Experience]:
        """
        Plays a step through the environment until the episode is complete

        Returns:
            Batch of all transitions for the entire episode
        """
        episode_steps, batch = [], []

        while len(batch) < self.episodes:
            exp = self.step(self.device)
            episode_steps.append(exp)

            if exp.done:
                batch.append(episode_steps)
                episode_steps = []

        yield batch

    def step(self, device: torch.device) -> Experience:
        """Carries out a single step in the environment"""
        action = self.agent(self.state, device)
        new_state, reward, done, _ = self.env.step(action)
        experience = Experience(
            state=self.state,
            action=action,
            reward=reward,
            new_state=new_state,
            done=done,
        )
        self.state = new_state

        if done:
            self.state = self.env.reset()

        return experience
