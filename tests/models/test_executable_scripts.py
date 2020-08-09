from unittest import mock
import pytest


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --limit_train_batches 3 --limit_val_batches 3'])
def test_cli_basic_gan(cli_args):
    from pl_bolts.models.gans.basic.basic_gan_module import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --limit_train_batches 3 --limit_val_batches 3'])
def test_cli_basic_vae(cli_args):
    from pl_bolts.models.autoencoders.basic_vae.basic_vae_module import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --limit_train_batches 3 --limit_val_batches 3'])
def test_cli_cpc(cli_args):
    from pl_bolts.models.self_supervised.cpc.cpc_module import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()
