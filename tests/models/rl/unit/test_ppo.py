import numpy as np
import torch
from pytorch_lightning import Trainer
from torch import Tensor

from pl_bolts.models.rl.ppo_model import PPO


def test_discount_rewards():
    """Test calculation of discounted rewards."""
    model = PPO(env="CartPole-v0", batch_size=16, gamma=0.99)

    rewards = np.ones(4)
    gt_qvals = [3.9403989999999998, 2.9701, 1.99, 1.0]

    qvals = model.discount_rewards(rewards, discount=0.99)

    assert gt_qvals == qvals


def test_critic_loss():
    """Test the critic loss function."""

    model = PPO(env="CartPole-v0", batch_size=16, gamma=0.99)
    obs_dim = model.env.observation_space.shape[0]

    batch_states = torch.rand(8, obs_dim)
    batch_qvals = torch.rand(8)

    loss = model.critic_loss(batch_states, batch_qvals)

    assert isinstance(loss, Tensor)


def test_actor_loss_categorical():
    """Test the actor loss function on categorical action-space environment."""

    model = PPO(env="CartPole-v0", batch_size=16, gamma=0.99)
    obs_dim = model.env.observation_space.shape[0]

    batch_states = torch.rand(8, obs_dim)
    batch_actions = torch.rand(8).long()
    batch_logp_old = torch.rand(8)
    batch_adv = torch.rand(8)

    loss = model.actor_loss(batch_states, batch_actions, batch_logp_old, batch_adv)

    assert isinstance(loss, Tensor)


def test_actor_loss_continuous():
    """Test the actor loss function on continuous action-space environment."""

    model = PPO(env="MountainCarContinuous-v0", batch_size=16, gamma=0.99)
    obs_dim = model.env.observation_space.shape[0]
    action_dim = model.env.action_space.shape[0]

    batch_states = torch.rand(8, obs_dim)
    batch_actions = torch.rand(8, action_dim)
    batch_logp_old = torch.rand(8)
    batch_adv = torch.rand(8)

    loss = model.actor_loss(batch_states, batch_actions, batch_logp_old, batch_adv)

    assert isinstance(loss, Tensor)


def test_generate_trajectory_samples():
    model = PPO("CartPole-v0", batch_size=16)

    obs_dim = model.env.observation_space.shape[0]

    sample_gen = model.generate_trajectory_samples()
    state, action, logp_old, qval, adv = next(sample_gen)

    assert state.shape[0] == obs_dim
    assert action
    assert logp_old
    assert qval
    assert adv


def test_training_categorical():
    model = PPO("CartPole-v0", batch_size=16)
    trainer = Trainer(max_epochs=1)
    trainer.fit(model)


def test_training_continous():
    model = PPO("MountainCarContinuous-v0", batch_size=16)
    trainer = Trainer(max_epochs=1)
    trainer.fit(model)
