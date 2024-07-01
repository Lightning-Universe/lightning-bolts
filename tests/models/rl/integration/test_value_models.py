import argparse
from unittest import TestCase

import pytest
import torch
from lightning.pytorch import Trainer
from pl_bolts.models.rl.double_dqn_model import DoubleDQN
from pl_bolts.models.rl.dqn_model import DQN
from pl_bolts.models.rl.dueling_dqn_model import DuelingDQN
from pl_bolts.models.rl.noisy_dqn_model import NoisyDQN
from pl_bolts.models.rl.per_dqn_model import PERDQN
from pl_bolts.utils import _IS_WINDOWS


class TestValueModels(TestCase):
    def setUp(self) -> None:
        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser = Trainer.add_argparse_args(parent_parser)
        parent_parser = DQN.add_model_specific_args(parent_parser)
        args_list = [
            "--warm_start_size",
            "100",
            "--gpus",
            str(int(torch.cuda.is_available())),
            "--env",
            "PongNoFrameskip-v4",
        ]
        self.hparams = parent_parser.parse_args(args_list)

        self.trainer = Trainer(
            gpus=self.hparams.gpus,
            max_steps=100,
            max_epochs=100,  # Set this as the same as max steps to ensure that it doesn't stop early
            val_check_interval=1,  # This just needs 'some' value, does not effect training right now
            fast_dev_run=True,
        )

    @pytest.mark.skipif(_IS_WINDOWS, reason="strange TimeOut or MemoryError")  # todo
    def test_dqn(self):
        """Smoke test that the DQN model runs."""
        model = DQN(self.hparams.env, num_envs=5)
        self.trainer.fit(model)

    def test_double_dqn(self):
        """Smoke test that the Double DQN model runs."""
        model = DoubleDQN(self.hparams.env)
        self.trainer.fit(model)

    @pytest.mark.skipif(_IS_WINDOWS, reason="strange TimeOut or MemoryError")  # todo
    def test_dueling_dqn(self):
        """Smoke test that the Dueling DQN model runs."""
        model = DuelingDQN(self.hparams.env)
        self.trainer.fit(model)

    @pytest.mark.skipif(_IS_WINDOWS, reason="strange TimeOut or MemoryError")  # todo
    def test_noisy_dqn(self):
        """Smoke test that the Noisy DQN model runs."""
        model = NoisyDQN(self.hparams.env)
        self.trainer.fit(model)

    @pytest.mark.skip(reason="CI is killing this test")
    def test_per_dqn(self):
        """Smoke test that the PER DQN model runs."""
        model = PERDQN(self.hparams.env)
        self.trainer.fit(model)

    # def test_n_step_dqn(self):
    #     """Smoke test that the N Step DQN model runs"""
    #     model = DQN(self.hparams.env, n_steps=self.hparams.n_steps)
    #     result = self.trainer.fit(model)
