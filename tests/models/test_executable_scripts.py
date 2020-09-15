from unittest import mock

import pytest


@pytest.mark.parametrize('cli_args', [
    '--dataset mnist --max_epochs 1 --batch_size 2 --limit_train_batches 2 --limit_val_batches 2',
    '--dataset cifar10 --max_epochs 1 --batch_size 2 --limit_train_batches 2 --limit_val_batches 2',
])
def test_cli_basic_gan(cli_args):
    from pl_bolts.models.gans.basic.basic_gan_module import cli_main

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()


@pytest.mark.parametrize('cli_args', [
    '--dataset cifar10 --max_epochs 1 --batch_size 2 --fast_dev_run',
])
def test_cli_basic_vae(cli_args):
    from pl_bolts.models.autoencoders.basic_vae.basic_vae_module import cli_main

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()


@pytest.mark.parametrize('cli_args', [
    '--dataset cifar10 --max_epochs 1 --batch_size 2 --fast_dev_run',
])
def test_cli_basic_ae(cli_args):
    from pl_bolts.models.autoencoders.basic_ae.basic_ae_module import cli_main

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()


@pytest.mark.parametrize('cli_args', [
    '--max_epochs 1 --limit_train_batches 2 --limit_val_batches 2 --batch_size 2 --encoder resnet18',
])
def test_cli_cpc(cli_args):
    from pl_bolts.models.self_supervised.cpc.cpc_module import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()