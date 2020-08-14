from unittest import mock

import pytest


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 3'])
def test_mnist(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.mnist_module import run_cli

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        run_cli()
