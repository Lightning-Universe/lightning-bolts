import argparse

import pytest
import torch.cuda
from pl_bolts.models.rl.advantage_actor_critic_model import AdvantageActorCritic
from pl_bolts.models.rl.sac_model import SAC
from pl_bolts.utils import _GYM_GREATER_EQUAL_0_20
from pytorch_lightning import Trainer


def test_a2c_cli():
    """Smoke test that the A2C model runs."""

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = AdvantageActorCritic.add_model_specific_args(parent_parser)
    args_list = ["--env", "CartPole-v0"]
    hparams = parent_parser.parse_args(args_list)

    trainer = Trainer(
        gpus=int(torch.cuda.is_available()),
        max_steps=100,
        max_epochs=100,  # Set this as the same as max steps to ensure that it doesn't stop early
        val_check_interval=1,  # This just needs 'some' value, does not effect training right now
        fast_dev_run=True,
    )
    model = AdvantageActorCritic(hparams.env)
    trainer.fit(model)


@pytest.mark.skipif(_GYM_GREATER_EQUAL_0_20, reason="gym.error.DeprecatedEnv: Env Pendulum-v0 not found")
def test_sac_cli():
    """Smoke test that the SAC model runs."""

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = Trainer.add_argparse_args(parent_parser)
    parent_parser = SAC.add_model_specific_args(parent_parser)
    args_list = [
        "--warm_start_size",
        "100",
        "--gpus",
        "0",  # fixme: RuntimeError on GPU: Expected all tensors to be on the same device
        "--env",
        "Pendulum-v0",
        "--batch_size",
        "10",
    ]
    hparams = parent_parser.parse_args(args_list)

    trainer = Trainer(
        gpus=hparams.gpus,
        max_steps=100,
        max_epochs=100,  # Set this as the same as max steps to ensure that it doesn't stop early
        val_check_interval=1,  # This just needs 'some' value, does not effect training right now
        fast_dev_run=True,
    )
    model = SAC(**hparams.__dict__)
    trainer.fit(model)
