import math
from typing import Sequence, Union

from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torch.nn import Module


class BYOLMAWeightUpdate(Callback):
    """
    Weight update rule from BYOL.

    Your model should have:

        - ``self.online_network``
        - ``self.target_network``

    Updates the target_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.

    .. note:: Automatically increases tau from ``initial_tau`` to 1.0 with every training step

    Example::

        # model must have 2 attributes
        model = Model()
        model.online_network = ...
        model.target_network = ...

        trainer = Trainer(callbacks=[BYOLMAWeightUpdate()])
    """

    def __init__(self, initial_tau: float = 0.996):
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
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
        dataloader_idx: int,
    ) -> None:
        # get networks
        online_net = pl_module.online_network
        target_net = pl_module.target_network

        # update weights
        self.update_weights(online_net, target_net)

        # update tau after
        self.current_tau = self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module: LightningModule, trainer: Trainer) -> float:
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs  # type: ignore[attr-defined]
        tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * pl_module.global_step / max_steps) + 1) / 2
        return tau

    def update_weights(self, online_net: Union[Module, Tensor], target_net: Union[Module, Tensor]) -> None:
        # apply MA weight update
        for (name, online_p), (_, target_p) in zip(
            online_net.named_parameters(),  # type: ignore[union-attr]
            target_net.named_parameters()  # type: ignore[union-attr]
        ):
            target_p.data = self.current_tau * target_p.data + (1 - self.current_tau) * online_p.data
