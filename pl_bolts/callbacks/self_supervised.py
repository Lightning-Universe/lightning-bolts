import math
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import Accuracy
from torch.nn import functional as F


class SSLOnlineEvaluator(pl.Callback):
    """
    Attaches a MLP for finetuning using the standard self-supervised protocol.

    Example::

        from pl_bolts.callbacks.self_supervised import SSLOnlineEvaluator

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

        self.train_acc = Accuracy(dist_sync_on_step=False)
        self.val_acc = Accuracy(compute_on_step=False)

    def on_pretrain_routine_start(self, trainer, pl_module):
        from pl_bolts.models.self_supervised.evaluator import SSLEvaluator

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
        acc = self.train_acc(mlp_preds, y)
        self.log('train_acc_step', acc, prog_bar=True)
        self.log('train_acc_epoch', self.train_acc)
        pl_module.log('train_mlp_loss', mlp_loss)

    def on_validation_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)
        mlp_loss = F.cross_entropy(mlp_preds, y)

        # log metrics
        self.val_acc(mlp_preds, y)
        pl_module.log('val_acc', self.val_acc)
        pl_module.log('val_mlp_loss', mlp_loss)


class BYOLMAWeightUpdate(pl.Callback):
    """
    Weight update rule from BYOL.

    Your model should have a:

        - self.online_network.
        - self.target_network.

    Updates the target_network params using an exponential moving average update rule weighted by tau.
    BYOL claims this keeps the online_network from collapsing.

    .. note:: Automatically increases tau from `initial_tau` to 1.0 with every training step

    Example::

        from pl_bolts.callbacks.self_supervised import BYOLMAWeightUpdate

        # model must have 2 attributes
        model = Model()
        model.online_network = ...
        model.target_network = ...

        trainer = Trainer(callbacks=[BYOLMAWeightUpdate()])
    """

    def __init__(self, initial_tau=0.996):
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
        super().__init__()
        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # get networks
        online_net = pl_module.online_network
        target_net = pl_module.target_network

        # update weights
        self.update_weights(online_net, target_net)

        # update tau after
        self.current_tau = self.update_tau(pl_module, trainer)

    def update_tau(self, pl_module, trainer):
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * pl_module.global_step / max_steps) + 1) / 2
        return tau

    def update_weights(self, online_net, target_net):
        # apply MA weight update
        for (name, online_p), (_, target_p) in zip(online_net.named_parameters(), target_net.named_parameters()):
            if 'weight' in name:
                target_p.data = self.current_tau * target_p.data + (1 - self.current_tau) * online_p.data
