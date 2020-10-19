from typing import Optional

import torch
from pytorch_lightning import Callback
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F

from pl_bolts.models.self_supervised.evaluator import SSLEvaluator


class SwavOnlineEvaluator(Callback):
    def __init__(
        self,
        drop_p: float = 0.2,
        hidden_dim: Optional[int] = None,
        z_dim: int = None,
        num_classes: int = None,
        dataset: str = 'stl10'
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.drop_p = drop_p
        self.optimizer = None

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.dataset = dataset

        self.loss = []
        self.acc = []

    def on_pretrain_routine_start(self, trainer, pl_module):
        pl_module.non_linear_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim,
        ).to(pl_module.device)

        self.optimizer = torch.optim.Adam(
            pl_module.non_linear_evaluator.parameters(), lr=1e-4
        )

    def get_representations(self, pl_module, x):
        representations = pl_module(x)
        representations = representations.reshape(representations.size(0), -1)
        return representations

    def to_device(self, batch, device):
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

    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)
        mlp_loss = F.cross_entropy(mlp_preds, y)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # log metrics
        acc = accuracy(mlp_preds, y)
        metrics = {"train_mlp_loss": mlp_loss, "train_mlp_acc": acc}
        pl_module.logger.log_metrics(metrics, step=trainer.global_step)

    def on_validation_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)
        mlp_loss = F.cross_entropy(mlp_preds, y)

        # log metrics
        acc = accuracy(mlp_preds, y)

        self.loss.append(mlp_loss.item())
        self.acc.append(acc.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {"val_mlp_loss": sum(self.loss) / len(self.loss), "val_mlp_acc": sum(self.acc) / len(self.acc)}
        pl_module.logger.log_metrics(metrics, step=trainer.current_epoch)

        # reset
        self.loss = []
        self.acc = []
