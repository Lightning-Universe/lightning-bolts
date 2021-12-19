from unittest import mock

import pytest

from tests import _MARK_REQUIRE_GPU, DATASETS_PATH

_DEFAULT_ARGS = f" --data_dir {DATASETS_PATH} --max_epochs 1 --max_steps 2 --batch_size 4"
_DEFAULT_LIGHTNING_CLI_ARGS = (
    f" fit --data.data_dir {DATASETS_PATH} --data.batch_size 4 --trainer.max_epochs 1 --trainer.max_steps 2"
)


@pytest.mark.parametrize("dataset_name", ["mnist", "cifar10"])
@pytest.mark.parametrize(
    "cli_args",
    [
        " --dataset %(dataset_name)s" + _DEFAULT_ARGS,
    ],
)
def test_cli_run_basic_gan(cli_args, dataset_name):
    from pl_bolts.models.gans.basic.basic_gan_module import cli_main

    cli_args = cli_args % {"dataset_name": dataset_name}
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()


@pytest.mark.parametrize("cli_args", ["--dataset mnist" + _DEFAULT_ARGS])
def test_cli_run_dcgan(cli_args):
    from pl_bolts.models.gans.dcgan.dcgan_module import cli_main

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()


@pytest.mark.parametrize("cli_args", ["--dataset mnist --scale_factor 4" + _DEFAULT_ARGS])
def test_cli_run_srgan(cli_args):
    from pl_bolts.models.gans.srgan.srgan_module import cli_main

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()


@pytest.mark.parametrize("cli_args", ["--dataset mnist --scale_factor 4" + _DEFAULT_ARGS])
def test_cli_run_srresnet(cli_args):
    from pl_bolts.models.gans.srgan.srresnet_module import cli_main

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()


@pytest.mark.parametrize("cli_args", [_DEFAULT_ARGS])
def test_cli_run_mnist(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.mnist_module import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize("cli_args", [_DEFAULT_ARGS])
def test_cli_run_basic_ae(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.autoencoders.basic_ae.basic_ae_module import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize("cli_args", [_DEFAULT_ARGS])
def test_cli_run_basic_vae(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.autoencoders.basic_vae.basic_vae_module import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize("cli_args", ["--max_epochs 1 --max_steps 2"])
def test_cli_run_lin_regression(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.regression.linear_regression import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize("cli_args", ["--max_epochs 1 --max_steps 2"])
def test_cli_run_log_regression(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.regression.logistic_regression import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize("cli_args", [_DEFAULT_ARGS + " --gpus 1"])
@pytest.mark.skipif(**_MARK_REQUIRE_GPU)
def test_cli_run_vision_image_gpt(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.vision.image_gpt.igpt_module import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize("cli_args", [_DEFAULT_LIGHTNING_CLI_ARGS + " --trainer.gpus 1"])
@pytest.mark.skipif(**_MARK_REQUIRE_GPU)
def test_cli_run_retinanet(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.detection.retinanet.retinanet_module import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()
