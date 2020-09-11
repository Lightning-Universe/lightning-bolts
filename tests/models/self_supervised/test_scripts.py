from unittest import mock

import pytest


@pytest.mark.parametrize('cli_args', ["--max_epochs 1 --max_steps 3 --fast_dev_run --batch_size 2"])
def test_cli_run_self_supervised_amdim(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.amdim.amdim_module import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3 --fast_dev_run --batch_size 2 --encoder resnet18'])
def test_cli_run_self_supervised_cpc(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.cpc.cpc_module import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3 --fast_dev_run --batch_size 2'])
def test_cli_run_self_supervised_moco(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.moco.moco2_module import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3 --fast_dev_run --batch_size 2 --online_ft'])
def test_cli_run_self_supervised_simclr(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.simclr.simclr_module import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3 --fast_dev_run --batch_size 2 --online_ft'])
def test_cli_run_self_supervised_byol(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.byol.byol_module import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()
