import torch
import torch.nn.functional as F
from pytorch_lightning import Callback
from pytorch_lightning.metrics.functional import accuracy
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator


class SSLOnlineEvaluator(Callback):

    def __init__(self, drop_p: float = 0.2, hidden_dim: int = 1024):
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

    def on_pretrain_routine_start(self, trainer, pl_module):
        # attach the evaluator to the module
        z_dim = pl_module.z_dim
        num_classes = pl_module.num_classes
        pl_module.non_linear_evaluator = SSLEvaluator(
            n_input=z_dim,
            n_classes=num_classes,
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

    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        x, y = batch
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
        acc = accuracy(mlp_preds, y)
        metrics = {'ft_callback_mlp_loss': mlp_loss, 'ft_callback_mlp_acc': acc}
        pl_module.logger.log_metrics(metrics)
