import argparse

from pytorch_lightning import Trainer

from pl_bolts.models.rl.sac_model import SAC


def test_sac():
    """Smoke test that the SAC model runs"""

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = Trainer.add_argparse_args(parent_parser)
    parent_parser = SAC.add_model_specific_args(parent_parser)
    args_list = [
        "--warm_start_size",
        "100",
        "--gpus",
        "0",
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
        fast_dev_run=True
    )
    model = SAC(**hparams.__dict__)
    result = trainer.fit(model)

    assert result == 1
