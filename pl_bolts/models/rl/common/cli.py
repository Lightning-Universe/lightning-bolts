"""Contains generic arguments used for all models."""

import argparse

from pl_bolts.utils.stability import under_review


@under_review()
def add_base_args(parent) -> argparse.ArgumentParser:
    """Adds arguments for DQN model.

    Note:
        These params are fine tuned for Pong env.

    Args:
        parent
    """
    arg_parser = argparse.ArgumentParser(parents=[parent])

    arg_parser.add_argument("--algo", type=str, default="dqn", help="algorithm to use for training")
    arg_parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    arg_parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")

    arg_parser.add_argument("--env", type=str, required=True, help="gym environment tag")
    arg_parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")

    arg_parser.add_argument("--episode_length", type=int, default=500, help="max length of an episode")
    arg_parser.add_argument("--max_episode_reward", type=int, default=18, help="max episode reward in the environment")
    arg_parser.add_argument(
        "--n_steps",
        type=int,
        default=4,
        help="how many steps to unroll for each update",
    )
    arg_parser.add_argument("--seed", type=int, default=123, help="seed for training run")
    arg_parser.add_argument("--epoch_len", type=int, default=1000, help="how many batches per epoch")
    arg_parser.add_argument("--num_envs", type=int, default=1, help="number of environments to run at once")
    arg_parser.add_argument(
        "--avg_reward_len", type=int, default=100, help="how many episodes to include in avg reward"
    )

    arg_parser.add_argument("--seed", type=int, default=123, help="seed for training run")
    return arg_parser
