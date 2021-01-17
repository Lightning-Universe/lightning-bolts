import argparse
from unittest import TestCase

import pytorch_lightning as pl

from pl_bolts.models.rl.reinforce_model import Reinforce
from pl_bolts.models.rl.vanilla_policy_gradient_model import VanillaPolicyGradient


class TestPolicyModels(TestCase):

    def setUp(self) -> None:
        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser = VanillaPolicyGradient.add_model_specific_args(parent_parser)
        args_list = [
            "--env",
            "CartPole-v0",
        ]
        self.hparams = parent_parser.parse_args(args_list)

        self.trainer = pl.Trainer(
            gpus=0,
            max_steps=100,
            max_epochs=100,  # Set this as the same as max steps to ensure that it doesn't stop early
            val_check_interval=1,  # This just needs 'some' value, does not effect training right now
            fast_dev_run=True
        )

    def test_reinforce(self):
        """Smoke test that the reinforce model runs"""

        model = Reinforce(self.hparams.env)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)

    def test_policy_gradient(self):
        """Smoke test that the policy gradient model runs"""
        model = VanillaPolicyGradient(self.hparams.env)
        result = self.trainer.fit(model)

        self.assertEqual(result, 1)
