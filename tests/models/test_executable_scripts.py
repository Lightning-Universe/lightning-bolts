from unittest import mock

import pytest


@pytest.mark.parametrize('cli_args', [
    '--dataset mnist --max_epochs 1 --batch_size 2 --limit_train_batches 2 --limit_val_batches 2',
    '--dataset cifar10 --max_epochs 1 --batch_size 2 --limit_train_batches 2 --limit_val_batches 2',
])
def test_cli_run_basic_gan(cli_args):
    from pl_bolts.models.gans.basic.basic_gan_module import cli_main

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()


@pytest.mark.parametrize('cli_args', [
    '--max_epochs 1 --limit_train_batches 2 --limit_val_batches 2 --batch_size 2 --encoder resnet18',
])
def test_cli_run_cpc(cli_args):
    from pl_bolts.models.self_supervised.cpc.cpc_module import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 2'])
def test_cli_run_mnist(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.mnist_module import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize('cli_args', [
    '--dataset cifar10 --max_epochs 1 --batch_size 2 --fast_dev_run',
])
def test_cli_run_basic_ae(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.autoencoders.basic_ae.basic_ae_module import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize('cli_args', [
    '--dataset cifar10 --max_epochs 1 --batch_size 2 --fast_dev_run',
])
def test_cli_run_basic_vae(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.autoencoders.basic_vae.basic_vae_module import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 2'])
def test_cli_run_lin_regression(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.regression.linear_regression import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 2'])
def test_cli_run_log_regression(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.regression.logistic_regression import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 2'])
def test_cli_run_vision_image_gpt(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.vision.image_gpt.igpt_module import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()
