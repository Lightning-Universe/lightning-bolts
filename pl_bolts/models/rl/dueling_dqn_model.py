"""
Dueling DQN
"""
import argparse
import pytorch_lightning as pl

from pl_bolts.models.rl.common import cli
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

        Args:
            env: gym environment tag
            gpus: number of gpus being used
            eps_start: starting value of epsilon for the epsilon-greedy exploration
            eps_end: final value of epsilon for the epsilon-greedy exploration
            eps_last_frame: the final frame in for the decrease of epsilon. At this frame espilon = eps_end
            sync_rate: the number of iterations between syncing up the target network with the train network
            gamma: discount factor
            lr: learning rate
            batch_size: size of minibatch pulled from the DataLoader
            replay_size: total capacity of the replay buffer
            warm_start_size: how many random steps through the environment to be carried out at the start of
                training to fill the buffer with a starting point
            sample_len: the number of samples to pull from the dataset iterator and feed to the DataLoader

        .. note:: Currently only supports CPU and single GPU training with `distributed_backend=dp`

        """

    def build_networks(self) -> None:
        """Initializes the Dueling DQN train and target networks"""
        self.net = DuelingCNN(self.obs_shape, self.n_actions)
        self.target_net = DuelingCNN(self.obs_shape, self.n_actions)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = cli.add_base_args(parser)
    parser = DuelingDQN.add_model_specific_args(parser)
    args = parser.parse_args()

    model = DuelingDQN(**args.__dict__)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)
