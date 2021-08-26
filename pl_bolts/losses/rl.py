"""Loss functions for the RL models."""

from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor, nn


def dqn_loss(batch: Tuple[Tensor, Tensor], net: nn.Module, target_net: nn.Module, gamma: float = 0.99) -> Tensor:
    """Calculates the mse loss using a mini batch from the replay buffer.

    Args:
        batch: current mini batch of replay data
        net: main training network
        target_net: target network of the main training network
        gamma: discount factor

    Returns:
        loss
    """
    states, actions, rewards, dones, next_states = batch

    actions = actions.long().squeeze(-1)

    state_action_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_state_values = target_net(next_states).max(1)[0]
        next_state_values[dones] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards

    return nn.MSELoss()(state_action_values, expected_state_action_values)


def double_dqn_loss(
    batch: Tuple[Tensor, Tensor],
    net: nn.Module,
    target_net: nn.Module,
    gamma: float = 0.99,
) -> Tensor:
    """Calculates the mse loss using a mini batch from the replay buffer. This uses an improvement to the original
    DQN loss by using the double dqn. This is shown by using the actions of the train network to pick the value
    from the target network. This code is heavily commented in order to explain the process clearly.

    Args:
        batch: current mini batch of replay data
        net: main training network
        target_net: target network of the main training network
        gamma: discount factor

    Returns:
        loss
    """
    states, actions, rewards, dones, next_states = batch  # batch of experiences, batch_size = 16

    actions = actions.long().squeeze(-1)

    state_action_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # dont want to mess with gradients when using the target network
    with torch.no_grad():
        next_outputs = net(next_states)  # [16, 2], [batch, action_space]

        next_state_acts = next_outputs.max(1)[1].unsqueeze(-1)  # take action at the index with the highest value
        next_tgt_out = target_net(next_states)

        # Take the value of the action chosen by the train network
        next_state_values = next_tgt_out.gather(1, next_state_acts).squeeze(-1)
        next_state_values[dones] = 0.0  # any steps flagged as done get a 0 value
        next_state_values = next_state_values.detach()  # remove values from the graph, no grads needed

    # calc expected discounted return of next_state_values
    expected_state_action_values = next_state_values * gamma + rewards

    # Standard MSE loss between the state action values of the current state and the
    # expected state action values of the next state
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def per_dqn_loss(
    batch: Tuple[Tensor, Tensor],
    batch_weights: List,
    net: nn.Module,
    target_net: nn.Module,
    gamma: float = 0.99,
) -> Tuple[Tensor, np.ndarray]:
    """Calculates the mse loss with the priority weights of the batch from the PER buffer.

    Args:
        batch: current mini batch of replay data
        batch_weights: how each of these samples are weighted in terms of priority
        net: main training network
        target_net: target network of the main training network
        gamma: discount factor

    Returns:
        loss and batch_weights
    """
    states, actions, rewards, dones, next_states = batch

    actions = actions.long()

    batch_weights = torch.tensor(batch_weights)

    actions_v = actions.unsqueeze(-1)
    outputs = net(states)
    state_action_vals = outputs.gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)

    with torch.no_grad():
        next_s_vals = target_net(next_states).max(1)[0]
        next_s_vals[dones] = 0.0
        exp_sa_vals = next_s_vals.detach() * gamma + rewards
    loss = (state_action_vals - exp_sa_vals) ** 2
    losses_v = batch_weights * loss
    return losses_v.mean(), (losses_v + 1e-5).data.cpu().numpy()
