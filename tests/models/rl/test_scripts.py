from unittest import mock

import pytest


@pytest.mark.parametrize(
    "cli_args",
    [
        " --env PongNoFrameskip-v4"
        " --max_steps 10"
        " --fast_dev_run 1"
        " --warm_start_size 10"
        " --n_steps 2"
        " --batch_size 10",
    ],
)
def test_cli_run_rl_dqn(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.dqn_model import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    "cli_args",
    [
        " --env PongNoFrameskip-v4"
        " --max_steps 10"
        " --fast_dev_run 1"
        " --warm_start_size 10"
        " --n_steps 2"
        " --batch_size 10",
    ],
)
def test_cli_run_rl_double_dqn(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.double_dqn_model import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    "cli_args",
    [
        " --env PongNoFrameskip-v4"
        " --max_steps 10"
        " --fast_dev_run 1"
        " --warm_start_size 10"
        " --n_steps 2"
        " --batch_size 10",
    ],
)
def test_cli_run_rl_dueling_dqn(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.dueling_dqn_model import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    "cli_args",
    [
        " --env PongNoFrameskip-v4"
        " --max_steps 10"
        " --fast_dev_run 1"
        " --warm_start_size 10"
        " --n_steps 2"
        " --batch_size 10",
    ],
)
def test_cli_run_rl_noisy_dqn(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.noisy_dqn_model import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    "cli_args",
    [
        " --env PongNoFrameskip-v4"
        " --max_steps 10"
        " --fast_dev_run 1"
        " --warm_start_size 10"
        " --n_steps 2"
        " --batch_size 10",
    ],
)
def test_cli_run_rl_per_dqn(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.per_dqn_model import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    "cli_args",
    [
        " --env CartPole-v0" " --max_steps 10" " --fast_dev_run 1" " --batch_size 10",
    ],
)
def test_cli_run_rl_reinforce(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.reinforce_model import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    "cli_args",
    [
        " --env CartPole-v0" " --max_steps 10" " --fast_dev_run 1" " --batch_size 10",
    ],
)
def test_cli_run_rl_vanilla_policy_gradient(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.vanilla_policy_gradient_model import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    "cli_args",
    [
        " --env CartPole-v0" " --max_steps 10" " --fast_dev_run 1" " --batch_size 10",
    ],
)
def test_cli_run_rl_advantage_actor_critic(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.advantage_actor_critic_model import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()


@pytest.mark.parametrize(
    "cli_args",
    [
        " --env Pendulum-v0" " --max_steps 10" " --fast_dev_run 1" " --batch_size 10",
    ],
)
def test_cli_run_rl_soft_actor_critic(cli_args):
    """Test running CLI for an example with default params."""
    from pl_bolts.models.rl.sac_model import cli_main

    cli_args = cli_args.strip().split(" ") if cli_args else []
    with mock.patch("argparse._sys.argv", ["any.py"] + cli_args):
        cli_main()
