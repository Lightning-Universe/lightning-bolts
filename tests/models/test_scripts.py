import importlib
from unittest import mock

import pytest
import torch
from pl_bolts.utils import _IS_WINDOWS, _JSONARGPARSE_GREATER_THAN_4_16_0

from tests import _MARK_REQUIRE_GPU, DATASETS_PATH

_DEFAULT_ARGS = f" --data_dir {DATASETS_PATH} --max_epochs 1 --max_steps 2 --batch_size 4 "
_DEFAULT_LIGHTNING_CLI_ARGS = (
    f" fit --data.data_dir {DATASETS_PATH} --data.batch_size 4 --trainer.max_epochs 1 --trainer.max_steps 2"
)


@pytest.mark.parametrize(
    ("script_path", "cli_args"),
    [
        ("models.gans.basic.basic_gan_module", _DEFAULT_ARGS + "--dataset mnist"),
        ("models.gans.basic.basic_gan_module", _DEFAULT_ARGS + "--dataset cifar10"),
        ("models.gans.dcgan.dcgan_module", _DEFAULT_ARGS + "--dataset mnist"),
        ("models.gans.srgan.srgan_module", _DEFAULT_ARGS + "--dataset mnist --scale_factor 4"),
        ("models.gans.srgan.srresnet_module", _DEFAULT_ARGS + "--dataset mnist --scale_factor 4"),
        ("models.mnist_module", _DEFAULT_ARGS),
        ("models.autoencoders.basic_ae.basic_ae_module", _DEFAULT_ARGS),
        ("models.autoencoders.basic_vae.basic_vae_module", _DEFAULT_ARGS),
        ("models.regression.linear_regression", "--max_epochs 1 --max_steps 2"),
        ("models.regression.logistic_regression", "--max_epochs 1 --max_steps 2"),
        pytest.param(
            "models.vision.image_gpt.igpt_module",
            _DEFAULT_ARGS + "--gpus 1",
            marks=pytest.mark.skipif(**_MARK_REQUIRE_GPU),
        ),
        pytest.param(
            "models.detection.retinanet.retinanet_module",
            _DEFAULT_LIGHTNING_CLI_ARGS + f" --trainer.gpus {int(torch.cuda.is_available())}",
            marks=[
                pytest.mark.skipif(**_MARK_REQUIRE_GPU),
                pytest.mark.skipif(not _JSONARGPARSE_GREATER_THAN_4_16_0, reason="Failing on CI, need to be fixed"),
            ],
        ),
    ],
)
@pytest.mark.skipif(_IS_WINDOWS, reason="strange TimeOut or MemoryError")  # todo
def test_cli_run_(script_path, cli_args):
    py_module = importlib.import_module(f"pl_bolts.{script_path}")

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        py_module.cli_main()
