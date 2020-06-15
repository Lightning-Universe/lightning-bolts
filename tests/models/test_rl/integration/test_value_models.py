import argparse
from unittest import TestCase
import pytorch_lightning as pl

from pl_bolts.models.rl.common import cli
from pl_bolts.models.rl.double_dqn.model import DoubleDQNLightning
from pl_bolts.models.rl.dqn.model import DQNLightning
from pl_bolts.models.rl.dueling_dqn.model import DuelingDQNLightning
from pl_bolts.models.rl.n_step_dqn.model import NStepDQNLightning
from pl_bolts.models.rl.noisy_dqn.model import NoisyDQNLightning
from pl_bolts.models.rl.per_dqn.model import PERDQNLightning


class TestValueModels(TestCase):

    def setUp(self) -> None:
        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser = cli.add_base_args(parent=parent_parser)
        parent_parser = DQNLightning.add_model_specific_args(parent_parser)
        args_list = [
            "--algo", "dqn",
            "--warm_start_steps", "100",
            "--episode_length", "100",
            "--gpus", "0"
        ]
        self.hparams = parent_parser.parse_args(args_list)

        self.trainer = pl.Trainer(
            gpus=self.hparams.gpus,
            max_steps=100,
            max_epochs=100,  # Set this as the same as max steps to ensure that it doesn't stop early
            val_check_interval=1000  # This just needs 'some' value, does not effect training right now
        )

    def test_dqn(self):
        """Smoke test that the DQN model runs"""
        model = DQNLightning(self.hparams)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)

    def test_double_dqn(self):
        """Smoke test that the Double DQN model runs"""
        model = DoubleDQNLightning(self.hparams)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)

    def test_dueling_dqn(self):
        """Smoke test that the Dueling DQN model runs"""
        model = DuelingDQNLightning(self.hparams)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)

    def test_noisy_dqn(self):
        """Smoke test that the Noisy DQN model runs"""
        model = NoisyDQNLightning(self.hparams)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)

    def test_per_dqn(self):
        """Smoke test that the PER DQN model runs"""
        model = PERDQNLightning(self.hparams)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)

    def test_n_step_dqn(self):
        """Smoke test that the N Step DQN model runs"""
        model = NStepDQNLightning(self.hparams)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)
