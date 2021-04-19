import argparse

import gym
import torch

from pl_bolts.models.rl.advantage_actor_critic_model import AdvantageActorCritic
from pl_bolts.models.rl.common.gym_wrappers import ToTensor
from pl_bolts.models.rl.common.networks import ActorCriticMLP


def test_a2c_loss():
    """Test the reinforce loss function"""

    env = ToTensor(gym.make("CartPole-v0"))
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    net = ActorCriticMLP(obs_shape, n_actions)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = AdvantageActorCritic.add_model_specific_args(parent_parser)
    args_list = [
        "--env",
        "CartPole-v0",
        "--batch_size",
        "32",
    ]
    hparams = parent_parser.parse_args(args_list)
    model = AdvantageActorCritic(**vars(hparams))

    batch_states = torch.rand(32, 4)
    batch_actions = torch.rand(32).long()
    batch_qvals = torch.rand(32)

    loss = model.loss(batch_states, batch_actions, batch_qvals)

    assert isinstance(loss, torch.Tensor)


def test_a2c_train_batch():
    """Tests that a single batch generates correctly"""
    env = ToTensor(gym.make("CartPole-v0"))
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    net = ActorCriticMLP(obs_shape, n_actions)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = AdvantageActorCritic.add_model_specific_args(parent_parser)
    args_list = [
        "--env",
        "CartPole-v0",
        "--batch_size",
        "32",
    ]
    hparams = parent_parser.parse_args(args_list)
    model = AdvantageActorCritic(**vars(hparams))

    model.n_steps = 4
    model.hparams.batch_size = 1
    xp_dataloader = model.train_dataloader()

    batch = next(iter(xp_dataloader))

    assert len(batch) == 3
    assert len(batch[0]) == model.hparams.batch_size
    assert isinstance(batch, list)
    assert isinstance(batch[0], torch.Tensor)
    assert isinstance(batch[1], torch.Tensor)
    assert isinstance(batch[2], torch.Tensor)
