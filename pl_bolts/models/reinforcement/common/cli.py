"""Contains generic arguments used for all models"""

import argparse


def add_base_args(parent) -> argparse.ArgumentParser:
    """
    Adds arguments for DQN model

    Note: these params are fine tuned for Pong env

    Args:
        parent
    """
    arg_parser = argparse.ArgumentParser(parents=[parent])

    arg_parser.add_argument(
        "--algo", type=str, default="dqn", help="algorithm to use for training"
    )
    arg_parser.add_argument(
        "--batch_size", type=int, default=32, help="size of the batches"
    )
    arg_parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    arg_parser.add_argument(
        "--env", type=str, default="PongNoFrameskip-v4", help="gym environment tag"
    )
    arg_parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    arg_parser.add_argument(
        "--episode_length", type=int, default=500, help="max length of an episode"
    )
    arg_parser.add_argument(
        "--max_episode_reward",
        type=int,
        default=18,
        help="max episode reward in the environment",
    )
    arg_parser.add_argument(
        "--max_steps", type=int, default=500000, help="max steps to train the agent"
    )
    arg_parser.add_argument(
        "--n_steps",
        type=int,
        default=4,
        help="how many steps to unroll for each update",
    )
    arg_parser.add_argument(
        "--gpus", type=int, default=1, help="number of gpus to use for training"
    )
    arg_parser.add_argument(
        "--seed", type=int, default=123, help="seed for training run"
    )
    arg_parser.add_argument(
        "--backend",
        type=str,
        default="dp",
        help="distributed backend to be used by lightning",
    )
    return arg_parser
