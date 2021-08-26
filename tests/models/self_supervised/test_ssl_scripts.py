from unittest import mock

import pytest

from tests import _MARK_REQUIRE_GPU, DATASETS_PATH

_DEFAULT_ARGS = f"--data_dir {DATASETS_PATH} --max_epochs 1 --max_steps 2 --batch_size 8 --num_workers 0"


# todo: failing for GPU as some is on CPU other on GPU
@pytest.mark.skip(reason="FIXME: failing for GPU as some is on CPU other on GPU")
@pytest.mark.parametrize(
    "cli_args",
    [
        _DEFAULT_ARGS + " --gpus 1",
    ],
)
@pytest.mark.skipif(**_MARK_REQUIRE_GPU)
def test_cli_run_ssl_amdim(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.amdim.amdim_module import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


# todo: failing test for wrong dimensions
@pytest.mark.skip(reason="FIXME: failing test for wrong dimensions")
@pytest.mark.parametrize("dataset_name", ["cifar10", "stl10"])
@pytest.mark.parametrize(
    "cli_args",
    [
        _DEFAULT_ARGS + " --dataset %(dataset_name)s" " --hidden_mlp 512" " --encoder resnet18" " --gpus 1",
    ],
)
@pytest.mark.skipif(**_MARK_REQUIRE_GPU)
def test_cli_run_ssl_cpc(cli_args, dataset_name):
    from pl_bolts.models.self_supervised.cpc.cpc_module import cli_main

    cli_args = cli_args % {"dataset_name": dataset_name}
    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    "cli_args",
    [
        _DEFAULT_ARGS + " --gpus 1",
    ],
)
@pytest.mark.skipif(**_MARK_REQUIRE_GPU)
def test_cli_run_ssl_moco(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.moco.moco2_module import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    "cli_args",
    [
        _DEFAULT_ARGS + " --online_ft" " --gpus 1" " --fp32",
    ],
)
@pytest.mark.skipif(**_MARK_REQUIRE_GPU)
def test_cli_run_ssl_simclr(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.simclr.simclr_module import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


# todo: seems to take too long
@pytest.mark.skip(reason="FIXME: seems to take too long")
@pytest.mark.parametrize(
    "cli_args",
    [
        _DEFAULT_ARGS + " --online_ft" " --gpus 1",
    ],
)
@pytest.mark.skipif(**_MARK_REQUIRE_GPU)
def test_cli_run_ssl_byol(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.byol.byol_module import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    "cli_args",
    [
        _DEFAULT_ARGS + " --dataset cifar10"
        " --arch resnet18"
        " --hidden_mlp 512"
        " --fp32"
        " --sinkhorn_iterations 1"
        " --nmb_prototypes 2"
        " --queue_length 0"
        " --gpus 1",
    ],
)
@pytest.mark.skipif(**_MARK_REQUIRE_GPU)
def test_cli_run_ssl_swav(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.swav.swav_module import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    "cli_args",
    [
        _DEFAULT_ARGS + " --dataset cifar10" " --online_ft" " --gpus 1" " --fp32",
    ],
)
@pytest.mark.skipif(**_MARK_REQUIRE_GPU)
def test_cli_run_ssl_simsiam(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.self_supervised.simsiam.simsiam_module import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()
