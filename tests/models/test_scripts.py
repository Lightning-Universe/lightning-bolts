from unittest import mock

import pytest

from tests import _MARK_REQUIRE_GPU, DATASETS_PATH


@pytest.mark.parametrize('dataset_name', ['mnist', 'cifar10'])
@pytest.mark.parametrize(
    'cli_args', [
        ' --dataset %(dataset_name)s'
        f' --data_dir {DATASETS_PATH}'
        ' --max_epochs 1'
        ' --batch_size 8'
        ' --limit_train_batches 2'
        ' --limit_val_batches 2',
    ]
)
def test_cli_run_basic_gan(cli_args, dataset_name):
    from pl_bolts.models.gans.basic.basic_gan_module import cli_main

    cli_args = cli_args % {'dataset_name': dataset_name}
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()


@pytest.mark.parametrize('cli_args', [f'--dataset mnist --data_dir {DATASETS_PATH} --fast_dev_run 1'])
def test_cli_run_dcgan(cli_args):
    from pl_bolts.models.gans.dcgan.dcgan_module import cli_main

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        cli_main()


@pytest.mark.parametrize(
    'cli_args', [
        f'--data_dir {DATASETS_PATH}'
        ' --max_epochs 1'
        ' --max_steps 2'
        ' --batch_size 4',
    ]
)
def test_cli_run_mnist(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.mnist_module import cli_main

    cli_args = cli_args.strip().split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    'cli_args', [
        f' --data_dir {DATASETS_PATH}'
        ' --max_epochs 1'
        ' --batch_size 4'
        ' --fast_dev_run 1'
        ' --num_workers 0',
    ]
)
def test_cli_run_basic_ae(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.autoencoders.basic_ae.basic_ae_module import cli_main

    cli_args = cli_args.strip().split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    'cli_args', [
        f' --data_dir {DATASETS_PATH}'
        ' --max_epochs 1'
        ' --batch_size 4'
        ' --num_workers 0',
    ]
)
def test_cli_run_basic_vae(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.autoencoders.basic_vae.basic_vae_module import cli_main

    cli_args = cli_args.strip().split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 2'])
def test_cli_run_lin_regression(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.regression.linear_regression import cli_main

    cli_args = cli_args.strip().split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1 --max_steps 2'])
def test_cli_run_log_regression(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.regression.logistic_regression import cli_main

    cli_args = cli_args.strip().split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.skipif(**_MARK_REQUIRE_GPU)
@pytest.mark.parametrize('cli_args', [f'--data_dir {DATASETS_PATH} --max_epochs 1 --max_steps 2 --gpus 1'])
def test_cli_run_vision_image_gpt(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.vision.image_gpt.igpt_module import cli_main

    cli_args = cli_args.strip().split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()
