import importlib
from unittest import mock

import pytest
import torch
from pl_bolts.utils import _GYM_GREATER_EQUAL_0_20, _IS_WINDOWS, _JSONARGPARSE_GREATER_THAN_4_16_0

from tests import _MARK_REQUIRE_GPU, DATASETS_PATH

_DEFAULT_ARGS = f" --data_dir={DATASETS_PATH} --max_epochs=1 --max_steps=2 --batch_size=4"

_DEFAULT_ARGS_RL_PONG = (
    " --env=PongNoFrameskip-v4 --max_steps=10 --fast_dev_run=1 --warm_start_size=10 --n_steps=2 --batch_size=10"
)
_DEFAULT_ARGS_CARTPOLE = " --env=CartPole-v0 --max_steps=10 --fast_dev_run=1 --batch_size=10"
_DEFAULT_LIGHTNING_CLI_ARGS = (
    f" fit --data.data_dir={DATASETS_PATH} --data.batch_size=4 --trainer.max_epochs=1 --trainer.max_steps=2"
)
_ARG_GPUS = f" --gpus={int(torch.cuda.is_available())}"
_ARG_WORKERS_0 = " --num_workers=0"


@pytest.mark.parametrize(
    ("script_path", "cli_args"),
    [
        ("models.gans.basic.basic_gan_module", _DEFAULT_ARGS + " --dataset=mnist"),
        ("models.gans.basic.basic_gan_module", _DEFAULT_ARGS + " --dataset=cifar10"),
        ("models.gans.dcgan.dcgan_module", _DEFAULT_ARGS + " --dataset=mnist"),
        ("models.gans.srgan.srgan_module", _DEFAULT_ARGS + " --dataset=mnist --scale_factor=4"),
        ("models.gans.srgan.srresnet_module", _DEFAULT_ARGS + " --dataset=mnist --scale_factor=4"),
        ("models.mnist_module", _DEFAULT_ARGS),
        ("models.autoencoders.basic_ae.basic_ae_module", _DEFAULT_ARGS),
        ("models.autoencoders.basic_vae.basic_vae_module", _DEFAULT_ARGS),
        ("models.regression.linear_regression", " --max_epochs=1 --max_steps=2"),
        ("models.regression.logistic_regression", " --max_epochs=1 --max_steps=2"),
        pytest.param(
            "models.vision.image_gpt.igpt_module",
            _DEFAULT_ARGS + _ARG_GPUS,
            marks=pytest.mark.skipif(**_MARK_REQUIRE_GPU),
        ),
        pytest.param(
            "models.detection.retinanet.retinanet_module",
            _DEFAULT_LIGHTNING_CLI_ARGS + f" --trainer.gpus={int(torch.cuda.is_available())}",
            marks=[
                pytest.mark.skipif(**_MARK_REQUIRE_GPU),
                pytest.mark.skipif(not _JSONARGPARSE_GREATER_THAN_4_16_0, reason="Failing on CI, need to be fixed"),
            ],
        ),
        ("models.rl.dqn_model", _DEFAULT_ARGS_RL_PONG),
        ("models.rl.double_dqn_model", _DEFAULT_ARGS_RL_PONG),
        ("models.rl.dueling_dqn_model", _DEFAULT_ARGS_RL_PONG),
        ("models.rl.noisy_dqn_model", _DEFAULT_ARGS_RL_PONG),
        ("models.rl.per_dqn_model", _DEFAULT_ARGS_RL_PONG),
        ("models.rl.reinforce_model", _DEFAULT_ARGS_CARTPOLE),
        ("models.rl.vanilla_policy_gradient_model", _DEFAULT_ARGS_CARTPOLE),
        ("models.rl.advantage_actor_critic_model", _DEFAULT_ARGS_CARTPOLE),
        pytest.param(
            "models.rl.sac_model",
            "--env=Pendulum-v0 --max_steps=10 --fast_dev_run=1 --batch_size=10",
            marks=pytest.mark.skipif(
                _GYM_GREATER_EQUAL_0_20, reason="gym.error.DeprecatedEnv: Env Pendulum-v0 not found"
            ),
        ),
        pytest.param(  # fixme
            "models.self_supervised.amdim.amdim_module",
            _DEFAULT_ARGS + _ARG_WORKERS_0 + _ARG_GPUS,
            marks=[
                pytest.mark.skipif(**_MARK_REQUIRE_GPU),
                pytest.mark.xfail(reason="failing for GPU as some is on CPU other on GPU"),
            ],
        ),
        pytest.param(  # fixme
            "models.self_supervised.cpc.cpc_module",
            _DEFAULT_ARGS + _ARG_WORKERS_0 + _ARG_GPUS + " --dataset=cifar10 --hidden_mlp=512 --encoder=resnet18",
            marks=[
                pytest.mark.skipif(**_MARK_REQUIRE_GPU),
                pytest.mark.xfail(reason="failing for GPU as some is on CPU other on GPU"),
            ],
        ),
        pytest.param(  # fixme
            "models.self_supervised.cpc.cpc_module",
            _DEFAULT_ARGS + _ARG_WORKERS_0 + _ARG_GPUS + " --datase=stl10 --hidden_mlp=512 --encoder=resnet18",
            marks=[
                pytest.mark.skipif(**_MARK_REQUIRE_GPU),
                pytest.mark.xfail(reason="failing for GPU as some is on CPU other on GPU"),
            ],
        ),
        pytest.param(
            "models.self_supervised.moco.moco_module",
            _DEFAULT_LIGHTNING_CLI_ARGS + f" --data.num_workers=0 --trainer.gpus={int(torch.cuda.is_available())}",
            marks=pytest.mark.skipif(**_MARK_REQUIRE_GPU),
        ),
        pytest.param(
            "models.self_supervised.simclr.simclr_module",
            _DEFAULT_ARGS + _ARG_WORKERS_0 + _ARG_GPUS + " --online_ft --fp32",
            marks=pytest.mark.skipif(**_MARK_REQUIRE_GPU),
        ),
        pytest.param(
            "models.self_supervised.byol.byol_module",
            _DEFAULT_ARGS + _ARG_WORKERS_0 + _ARG_GPUS,
            marks=pytest.mark.skipif(**_MARK_REQUIRE_GPU),
        ),
        pytest.param(
            "models.self_supervised.swav.swav_module",
            _DEFAULT_ARGS + _ARG_WORKERS_0 + _ARG_GPUS + " --dataset=cifar10 --arch=resnet18 --hidden_mlp=512 --fp32"
            " --sinkhorn_iterations=1 --num_prototypes=2 --queue_length=0",
            marks=pytest.mark.skipif(**_MARK_REQUIRE_GPU),
        ),
        pytest.param(
            "models.self_supervised.simsiam.simsiam_module",
            _DEFAULT_ARGS + _ARG_WORKERS_0 + _ARG_GPUS,
            marks=pytest.mark.skipif(**_MARK_REQUIRE_GPU),
        ),
    ],
)
@pytest.mark.skipif(_IS_WINDOWS, reason="strange TimeOut or MemoryError")  # todo
def test_cli_run_(script_path, cli_args):
    py_module = importlib.import_module(f"pl_bolts.{script_path}")

    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args.strip().split()):
        py_module.cli_main()
