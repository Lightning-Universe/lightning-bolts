from unittest import mock

import pytest


@pytest.mark.skip(reason='seems to freeze CLI run...')  # TODO
@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
def test_cli_run_self_supervised_amdim(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.amdim.amdim_module import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()


@pytest.mark.skip(reason='seems to freeze CLI run...')  # TODO
@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
def test_cli_run_self_supervised_cpc(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.cpc.cpc_module import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()


@pytest.mark.skip(reason='seems to freeze CLI run...')  # TODO
@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
def test_cli_run_self_supervised_moco(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.moco.moco2_module import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()


@pytest.mark.skip(reason='seems to freeze CLI run...')  # TODO
@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
def test_cli_run_self_supervised_simclr(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.simclr.simclr_module import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()
