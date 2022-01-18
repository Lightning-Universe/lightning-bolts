import numpy as np

from pl_bolts.models.rl.common.rewards import discount_rewards, calc_advantage


def test_discount_rewards():
    """Test calculation of discounted rewards."""
    rewards = np.ones(4)
    gt_qvals = [3.9403989999999998, 2.9701, 1.99, 1.0]

    qvals = discount_rewards(rewards, discount=0.99)

    assert gt_qvals == qvals


def test_advantages():
    """Test calculation of advantages."""
    rewards = [1, 1, 2, 1]
    values = [1, 0, 0.5, 0]
    expected_advantages = [4.38835898419875, 4.6659850975, 3.371595, 1.99]
    # Explanation  deltas = rewards + 0.99 * values_(t+1) - values_t =
    # [ 1 + (0.99 * 0 - 1) = 0,
    #   1 + (0.99 * 0.5 - 0) = 1.495,
    #   2 + (0.99 * 0 - 0.5) = 1.5,
    #   1 + (0.99 * 1 (last_value) - 0 = 1.99]
    # Now calculate discounted rewards based on deltas and 0.99 * 0.95 = 0.9405
    # A_t = 1.99, A_(t-1) = 1.99 * 0.9405 + 1.5 = 3.371595, A_(t-2) = 3.371595 * 0.9405 + 1.495 = 4.6659850975 etc..

    advantages = calc_advantage(rewards, values, last_value=1, gamma=0.99, lam=0.95)

    assert advantages == expected_advantages
