import math

import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F


class SSLOnlineEvaluator(pl.Callback):  # pragma: no-cover

    def __init__(self, drop_p: float = 0.2, hidden_dim: int = 1024, z_dim: int = None, num_classes: int = None):
        """
        Attaches a MLP for finetuning using the standard self-supervised protocol.

        Example::

            from pl_bolts.callbacks.self_supervised import SSLOnlineEvaluator

            # your model must have 2 attributes
            model = Model()
            model.z_dim = ... # the representation dim
            model.num_classes = ... # the num of classes in the model

        Args:
            drop_p: (0.2) dropout probability
            hidden_dim: (1024) the hidden dimension for the finetune MLP
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop_p = drop_p
        self.optimizer = None
        self.z_dim = z_dim
        self.num_classes = num_classes

    def on_pretrain_routine_start(self, trainer, pl_module):
        from pl_bolts.models.self_supervised.evaluator import SSLEvaluator

        # attach the evaluator to the module

        if hasattr(pl_module, 'z_dim'):
            self.z_dim = pl_module.z_dim
        if hasattr(pl_module, 'num_classes'):
            self.num_classes = pl_module.num_classes

        pl_module.non_linear_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim
        ).to(pl_module.device)

        self.optimizer = torch.optim.SGD(pl_module.non_linear_evaluator.parameters(), lr=1e-3)

    def get_representations(self, pl_module, x):
        """
        Override this to customize for the particular model
        Args:
            pl_module:
            x:
        """
        if len(x) == 2 and isinstance(x, list):
            x = x[0]

        representations = pl_module(x)
        representations = representations.reshape(representations.size(0), -1)
        return representations

    def to_device(self, batch, device):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        return x, y

    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)
        mlp_loss = F.cross_entropy(mlp_preds, y)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # log metrics
        if trainer.datamodule is not None:
            acc = accuracy(mlp_preds, y, num_classes=trainer.datamodule.num_classes)
        else:
            acc = accuracy(mlp_preds, y)

        metrics = {'ft_callback_mlp_loss': mlp_loss, 'ft_callback_mlp_acc': acc}
        pl_module.logger.log_metrics(metrics, step=trainer.global_step)


class BYOLMAWeightUpdate(pl.Callback):

    def __init__(self, initial_tau=0.996):
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

        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
        super().__init__()
        self.initial_tau = initial_tau
        self.current_tau = initial_tau

    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
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
