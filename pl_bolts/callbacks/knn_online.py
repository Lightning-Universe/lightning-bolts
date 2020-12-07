from typing import Optional

import torch
from pytorch_lightning import Callback

from sklearn.neighbors import KNeighborsClassifier

class KNNOnlineEvaluator(Callback):  # pragma: no-cover
    """
    Evaluates self-supervised K nearest neighbors.

    Example::

        # your model must have 1 attribute
        model = Model()
        model.num_classes = ... # the num of classes in the model

        online_eval = KNNOnlineEvaluator(
            num_classes=model.num_classes,
            dataset='imagenet'
        )

    """
    def __init__(
        self,
        dataset: str,
        num_classes: int = None,
    ):
        """
        Args:
            dataset: if stl10, need to get the labeled batch
            num_classes: Number of classes
        """
        super().__init__()

        self.num_classes = num_classes
        self.dataset = dataset

    def on_pretrain_routine_start(self, trainer, pl_module):
        pl_module.knn_evaluator = KNeighborsClassifier(n_neighbors=self.num_classes)


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

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # knn fit
        pl_module.knn_evaluator.fit(representations, y)
        train_acc = pl_module.knn_evaluator.score(representations, y)

        # log metrics
        pl_module.log('online_knn_train_acc', train_acc, on_step=True, on_epoch=False)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module, x)

        representations = representations.detach()

        # train knn
        val_acc = pl_module.knn_evaluator.score(representations, y)
        
        # log metrics
        pl_module.log('online_knn_val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True)
