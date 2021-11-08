"""Double DQN."""
import argparse
from collections import OrderedDict
from typing import Tuple

from pytorch_lightning import Trainer
from torch import Tensor

from pl_bolts.losses.rl import double_dqn_loss
from pl_bolts.models.rl.dqn_model import DQN


class DoubleDQN(DQN):
    """Double Deep Q-network (DDQN) PyTorch Lightning implementation of `Double DQN`_.

    Paper authors: Hado van Hasselt, Arthur Guez, David Silver

    Model implemented by:

        - `Donal Byrne <https://github.com/djbyrne>`

    Example:

        >>> from pl_bolts.models.rl.double_dqn_model import DoubleDQN
        ...
        >>> model = DoubleDQN("PongNoFrameskip-v4")

    Train::

        trainer = Trainer()
        trainer.fit(model)

    Note:
        This example is based on
        https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter08/03_dqn_double.py

    Note:
        Currently only supports CPU and single GPU training with `accelerator=dp`

    .. _`Double DQN`: https://arxiv.org/pdf/1509.06461.pdf
    """

    def training_step(self, batch: Tuple[Tensor, Tensor], _) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            _: batch number, not used

        Returns:
            Training loss and log metrics
        """

        # calculates training loss
        loss = double_dqn_loss(batch, self.net, self.target_net, self.gamma)

        if self._use_dp_or_ddp2(self.trainer):
            loss = loss.unsqueeze(0)

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict(
            {
                "total_reward": self.total_rewards[-1],
                "avg_reward": self.avg_rewards,
                "train_loss": loss,
                # "episodes": self.total_episode_steps,
            }
        )

        return OrderedDict(
            {
                "loss": loss,
                "avg_reward": self.avg_rewards,
            }
        )


def cli_main():
    parser = argparse.ArgumentParser(add_help=False)

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = DoubleDQN.add_model_specific_args(parser)
    args = parser.parse_args()

    model = DoubleDQN(**args.__dict__)

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == "__main__":
    cli_main()
