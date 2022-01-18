from typing import List, Union

import numpy as np


def discount_rewards(rewards: Union[np.array, List[float]], discount: float = 0.99) -> List[float]:
    """Calculate the discounted rewards of all rewards in list.

    Args:
        rewards: list of rewards/advantages
        discount: discount factor

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


def calc_advantage(
    rewards: List[float], values: List[float], last_value: float, gamma: float = 0.99, lam: float = 0.95
) -> List[float]:
    """Calculate the advantage given rewards, state values, and the last value of episode.

    Args:
        rewards: list of episode rewards
        values: list of state values from critic
        last_value: value of last state of episode
        gamma: gamma parameter
        lam: lambda parameter

    Returns:
        list of advantages
    """
    rews = rewards + [last_value]
    vals = values + [last_value]
    # GAE
    delta = [rews[i] + gamma * vals[i + 1] - vals[i] for i in range(len(rews) - 1)]
    adv = discount_rewards(delta, gamma * lam)

    return adv
