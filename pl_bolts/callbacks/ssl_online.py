from typing import Optional, Sequence, Tuple, Union, Dict, Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics.functional import accuracy
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator

from pytorch_lightning.utilities import rank_zero_warn


class SSLOnlineEvaluator(Callback):  # pragma: no cover
    """Attaches a MLP for fine-tuning using the standard self-supervised protocol.

    Example::

        # your model must have 2 attributes
        model = Model()
        model.z_dim = ... # the representation dim
        model.num_classes = ... # the num of classes in the model

        online_eval = SSLOnlineEvaluator(
            z_dim=model.z_dim,
            num_classes=model.num_classes,
            dataset='imagenet'
        )

    """

    def __init__(
            self,
            drop_p: float = 0.2,
            hidden_dim: Optional[int] = None,
    ):
        """
        Args:
            dataset: if stl10, need to get the labeled batch
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
            z_dim: Representation dimension
            num_classes: Number of classes
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.drop_p = drop_p

        self.optimizer: Optional[Optimizer] = None
        self.online_evaluator: Optional[SSLEvaluator] = None
        self.z_dim: Optional[int] = None
        self.num_classes: Optional[int] = None
        self.dataset: Optional[str] = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        self.z_dim = pl_module.online_network.backbone.fc.in_features
        self.num_classes = trainer.datamodule.num_classes
        self.dataset = trainer.datamodule.name

        self.online_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim,
        )

        self.optimizer = torch.optim.Adam(self.online_evaluator.parameters(), lr=1e-4)

    def on_pretrain_routine_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.online_evaluator.to(pl_module.device)  # must move to device after setup, as during setup, pl_module is still on cpu

        if trainer.accelerator_connector.is_distributed:
            if trainer.accelerator_connector.use_ddp:
                from torch.nn.parallel import DistributedDataParallel as DDP
                self.online_evaluator = DDP(self.online_evaluator, find_unused_parameters=True, device_ids=[pl_module.device])
            elif trainer.accelerator_connector.use_dp:
                from torch.nn.parallel import DataParallel as DP
                self.online_evaluator = DP(self.online_evaluator, device_ids=[pl_module.device])
            else:
                rank_zero_warn("Does not support this type of distributed accelerator. The online evaluator will not sync.")

    @staticmethod
    def get_representations(pl_module: LightningModule, x: Tensor) -> Tensor:
        representations = pl_module(x)
        representations = representations.reshape(representations.size(0), -1)
        return representations

    def to_device(self, batch: Sequence, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor]:
        # get the labeled batch
        if self.dataset == 'stl10':
            labeled_batch = batch[1]
            batch = labeled_batch

        inputs, y = batch

        # last input is for online eval
        x = inputs[-1]
        x = x.to(device)
        y = y.to(device)

        return x, y

    def on_train_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: Sequence,
            batch: Sequence,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_logits = self.online_evaluator(representations)  # type: ignore[operator]
        mlp_loss = F.cross_entropy(mlp_logits, y)

        # print(f"mlp_loss:{mlp_loss}")
        # print(f"before step norm:{sum([torch.norm(p) for p in self.online_evaluator.parameters()])}")

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # print(f"after step norm:{sum([torch.norm(p) for p in self.online_evaluator.parameters()])}")

        # log metrics
        train_acc = accuracy(mlp_logits.softmax(-1), y)
        pl_module.log('online_train_acc', train_acc, on_step=True, on_epoch=False)
        pl_module.log('online_train_loss', mlp_loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: Sequence,
            batch: Sequence,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_logits = self.online_evaluator(representations)  # type: ignore[operator]
        mlp_loss = F.cross_entropy(mlp_logits, y)

        # log metrics
        val_acc = accuracy(mlp_logits.softmax(-1), y)
        pl_module.log('online_val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log('online_val_loss', mlp_loss, on_step=False, on_epoch=True, sync_dist=True)

    def on_save_checkpoint(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            checkpoint: Dict[str, Any]
    ) -> dict:
        return {
            'state_dict': self.online_evaluator.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }

    def on_load_checkpoint(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            callback_state: Dict[str, Any]
    ) -> None:
        self.online_evaluator.load_state_dict(callback_state['state_dict'])
        self.optimizer.load_state_dict(callback_state['optimizer_state'])
