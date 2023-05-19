import math
from typing import Sequence, Union

import torch.nn as nn
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor


class BYOLMAWeightUpdate(Callback):
    """Weight update rule from Bootstrap Your Own Latent (BYOL).

    Updates the target_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.

    The PyTorch Lightning module being trained should have:

        - ``self.online_network``
        - ``self.target_network``

    .. note:: Automatically increases tau from ``initial_tau`` to 1.0 with every training step

    Args:
        initial_tau (float, optional): starting tau. Auto-updates with every training step

    Example::

        # model must have 2 attributes
        model = Model()
        model.online_network = ...
        model.target_network = ...

        trainer = Trainer(callbacks=[BYOLMAWeightUpdate()])
    """

    def __init__(self, initial_tau: float = 0.996) -> None:
        if not 0.0 <= initial_tau <= 1.0:
            raise ValueError(f"initial tau should be between 0 and 1 instead of {initial_tau}.")

        super().__init__()
        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        # get networks
        online_net = pl_module.online_network
        target_net = pl_module.target_network

        # update target network weights
        self.update_weights(online_net, target_net)

        # update tau after
        self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module: LightningModule, trainer: Trainer) -> None:
        """Update tau value for next update."""
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        self.current_tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * pl_module.global_step / max_steps) + 1) / 2

    def update_weights(self, online_net: Union[nn.Module, Tensor], target_net: Union[nn.Module, Tensor]) -> None:
        """Update target network parameters."""
        for online_p, target_p in zip(online_net.parameters(), target_net.parameters()):
            target_p.data = self.current_tau * target_p.data + (1.0 - self.current_tau) * online_p.data
