import argparse

import torch
from torch import Tensor

from pl_bolts.models.rl.sac_model import SAC


def test_sac_loss():
    """Test the reinforce loss function."""
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = SAC.add_model_specific_args(parent_parser)
    args_list = [
        "--env",
        "Pendulum-v0",
        "--batch_size",
        "32",
    ]
    hparams = parent_parser.parse_args(args_list)
    model = SAC(**vars(hparams))

    batch_states = torch.rand(32, 3)
    batch_actions = torch.rand(32, 1)
    batch_rewards = torch.rand(32)
    batch_dones = torch.ones(32)
    batch_next_states = torch.rand(32, 3)
    batch = (batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states)

    policy_loss, q1_loss, q2_loss = model.loss(batch)

    assert isinstance(policy_loss, Tensor)
    assert isinstance(q1_loss, Tensor)
    assert isinstance(q2_loss, Tensor)


def test_sac_train_batch():
    """Tests that a single batch generates correctly."""
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = SAC.add_model_specific_args(parent_parser)
    args_list = [
        "--env",
        "Pendulum-v0",
        "--batch_size",
        "32",
    ]
    hparams = parent_parser.parse_args(args_list)
    model = SAC(**vars(hparams))

    xp_dataloader = model.train_dataloader()

    batch = next(iter(xp_dataloader))

    assert len(batch) == 5
    assert len(batch[0]) == model.hparams.batch_size
    assert isinstance(batch, list)
    assert all(isinstance(batch[i], Tensor) for i in range(5))
