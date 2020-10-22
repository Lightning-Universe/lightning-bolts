"""
Dueling DQN
"""
import argparse

import pytorch_lightning as pl

from pl_bolts.models.rl.common.networks import DuelingCNN
from pl_bolts.models.rl.dqn_model import DQN


class DuelingDQN(DQN):
    """
    PyTorch Lightning implementation of `Dueling DQN <https://arxiv.org/abs/1511.06581>`_

    Paper authors: Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas

    Model implemented by:

        - `Donal Byrne <https://github.com/djbyrne>`

    Example:

        >>> from pl_bolts.models.rl.dueling_dqn_model import DuelingDQN
        ...
        >>> model = DuelingDQN("PongNoFrameskip-v4")

    Train::

        trainer = Trainer()
        trainer.fit(model)

    .. note:: Currently only supports CPU and single GPU training with `distributed_backend=dp`

    """

    def build_networks(self) -> None:
        """Initializes the Dueling DQN train and target networks"""
        self.net = DuelingCNN(self.obs_shape, self.n_actions)
        self.target_net = DuelingCNN(self.obs_shape, self.n_actions)


def cli_main():
    parser = argparse.ArgumentParser(add_help=False)

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = DuelingDQN.add_model_specific_args(parser)
    args = parser.parse_args()

    model = DuelingDQN(**args.__dict__)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == '__main__':
    cli_main()
