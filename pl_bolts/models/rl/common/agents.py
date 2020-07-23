"""
Agent module containing classes for Agent logic

Based on the implementations found here: https://github.com/Shmuma/ptan/blob/master/ptan/agent.py
"""
from abc import ABC
from random import randint

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import List


class Agent(ABC):
    """Basic agent that always returns 0"""

    def __init__(self, net: nn.Module):
        self.net = net

    def __call__(self, state: torch.Tensor, device: str, *args, **kwargs) -> List[int]:
        """
        Using the given network, decide what action to carry

        Args:
            state: current state of the environment
            device: device used for current batch
        Returns:
            action
        """
        return [0]


class ValueAgent(Agent):
    """Value based agent that returns an action based on the Q values from the network"""

    def __init__(
            self,
            net: nn.Module,
            action_space: int,
            eps_start: float = 1.0,
            eps_end: float = 0.2,
            eps_frames: float = 1000,
    ):
        super().__init__(net)
        self.action_space = action_space
        self.eps_start = eps_start
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_frames = eps_frames

    @torch.no_grad()
    def __call__(self, state: torch.Tensor, device: str) -> List[int]:
        """
        Takes in the current state and returns the action based on the agents policy

        Args:
            state: current state of the environment
            device: the device used for the current batch

        Returns:
            action defined by policy
        """

        if np.random.random() < self.epsilon:
            action = self.get_random_action()
        else:
            action = self.get_action(state, device)

        return [action]

    def get_random_action(self) -> int:
        """returns a random action"""
        action = randint(0, self.action_space - 1)

        return action

    def get_action(self, state: torch.Tensor, device: torch.device):
        """
            Returns the best action based on the Q values of the network
            Args:
                state: current state of the environment
                device: the device used for the current batch
            Returns:
                action defined by Q values
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=device)

        q_values = self.net(state)
        _, actions = torch.max(q_values, dim=1)
        return actions.detach().item()

    def update_epsilon(self, step: int) -> None:
        """
        Updates the epsilon value based on the current step

        Args:
            step: current global step
        """
        self.epsilon = max(self.eps_end, self.eps_start - (step + 1) / self.eps_frames)


class PolicyAgent(Agent):
    """Policy based agent that returns an action based on the networks policy"""

    @torch.no_grad()
    def __call__(self, states: torch.Tensor, device: str) -> List[int]:
        """
        Takes in the current state and returns the action based on the agents policy

        Args:
            states: current state of the environment
            device: the device used for the current batch

        Returns:
            action defined by policy
        """
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(device)

        if device.type != "cpu":
            states = states.cuda(device)

        # get the logits and pass through softmax for probability distribution
        probabilities = F.softmax(self.net(states))
        prob_np = probabilities.data.cpu().numpy()

        # take the numpy values and randomly select action based on prob distribution
        actions = [np.random.choice(len(prob), p=prob) for prob in prob_np]

        return actions
