from unittest import mock

import pytest


@pytest.mark.parametrize(
    "dataset_name", [
        pytest.param('mnist', id="mnist"),
        pytest.param('cifar10', id="cifar10")
    ]
)
def test_cli_basic_gan(dataset_name):
    from pl_bolts.models.gans.basic.basic_gan_module import cli_main

    cli_args = f"""
        --dataset {dataset_name}
        --max_epochs 1
        --limit_train_batches 3
        --limit_val_batches 3
        --batch_size 3
    """.strip().split()

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    "dataset_name", [
        pytest.param('cifar10', id="cifar10")
    ]
)
def test_cli_basic_vae(dataset_name):
    from pl_bolts.models.autoencoders.basic_vae.basic_vae_module import cli_main

    cli_args = f"""
        --dataset {dataset_name}
        --max_epochs 1
        --batch_size 3
        --fast_dev_run
        --gpus 0
    """.strip().split()

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    "dataset_name", [
        pytest.param('cifar10', id="cifar10")
    ]
)
def test_cli_basic_ae(dataset_name):
    from pl_bolts.models.autoencoders.basic_ae.basic_ae_module import cli_main

    cli_args = f"""
        --dataset {dataset_name}
        --max_epochs 1
        --batch_size 3
        --fast_dev_run
        --gpus 0
    """.strip().split()

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize('cli_args', ['--max_epochs 1'
                                      ' --limit_train_batches 3'
                                      ' --limit_val_batches 3'
                                      ' --batch_size 3'
                                      ' --encoder resnet18'])
def test_cli_cpc(cli_args):
    from pl_bolts.models.self_supervised.cpc.cpc_module import cli_main

    cli_args = cli_args.split(' ') if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()
