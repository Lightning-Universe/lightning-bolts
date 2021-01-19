from unittest import mock

import pytest
import torch

from tests import DATASETS_PATH


@pytest.mark.parametrize(
    'cli_args', [
        f"--data_dir {DATASETS_PATH}"
        " --max_epochs 1"
        " --max_steps 3"
        " --fast_dev_run 1"
        " --batch_size 2"
        " --num_workers 0"
    ]
)
def test_cli_run_self_supervised_amdim(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.amdim.amdim_module import cli_main

    cli_args = cli_args.strip().split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


# TODO: this test is hanging (runs for more then 10min) so we need to use GPU or optimize it...
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.parametrize(
    'cli_args', [
        f' --data_dir {DATASETS_PATH} --max_epochs 1'
        ' --max_steps 3'
        ' --fast_dev_run 1'
        ' --batch_size 2'
        ' --encoder resnet18'
        ' --num_workers 0'
    ]
)
def test_cli_run_self_supervised_cpc(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.cpc.cpc_module import cli_main

    cli_args = cli_args.strip().split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    'cli_args', [
        f' --data_dir {DATASETS_PATH}'
        ' --max_epochs 1'
        ' --max_steps 3'
        ' --fast_dev_run 1'
        ' --batch_size 2'
        ' --num_workers 0'
    ]
)
def test_cli_run_self_supervised_moco(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.moco.moco2_module import cli_main

    cli_args = cli_args.strip().split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    'cli_args', [
        f' --data_dir {DATASETS_PATH}'
        ' --max_epochs 1'
        ' --max_steps 3'
        ' --fast_dev_run 1'
        ' --batch_size 2'
        ' --num_workers 0'
        ' --online_ft'
        ' --gpus 0'
        ' --fp32'
    ]
)
def test_cli_run_self_supervised_simclr(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.simclr.simclr_module import cli_main

    cli_args = cli_args.strip().split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    'cli_args', [
        f' --data_dir {DATASETS_PATH}'
        ' --max_epochs 1'
        ' --max_steps 3'
        ' --fast_dev_run 1'
        ' --batch_size 2'
        " --num_workers 0"
        ' --online_ft'
    ]
)
def test_cli_run_self_supervised_byol(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.byol.byol_module import cli_main

    cli_args = cli_args.strip().split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    'cli_args', [
        ' --dataset cifar10'
        f' --data_dir {DATASETS_PATH}'
        ' --max_epochs 1'
        ' --max_steps 3'
        ' --fast_dev_run 1'
        ' --batch_size 2'
        ' --arch resnet18'
        ' --hidden_mlp 512'
        ' --sinkhorn_iterations 1'
        ' --nmb_prototypes 2'
        ' --num_workers 0'
        ' --queue_length 0'
        ' --gpus 0'
    ]
)
def test_cli_run_self_supervised_swav(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.swav.swav_module import cli_main

    cli_args = cli_args.strip().split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    'cli_args', [
        ' --dataset cifar10'
        f' --data_dir {DATASETS_PATH}'
        ' --max_epochs 1'
        ' --max_steps 3'
        ' --fast_dev_run 1'
        ' --batch_size 2'
        ' --num_workers 0'
        ' --online_ft'
        ' --gpus 0'
        ' --fp32'
    ]
)
def test_cli_run_self_supervised_simsiam(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.simsiam.simsiam_module import cli_main

    cli_args = cli_args.strip().split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()
