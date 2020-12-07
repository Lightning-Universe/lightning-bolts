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


    def get_representations(self, pl_module, x):
        representations = pl_module(x)
        representations = representations.reshape(representations.size(0), -1)
        return representations

    def get_all_representations(self, pl_module, dataloader):
        all_representations = None
        ys = None

        for batch in dataloader:
            x, y = self.to_device(batch, pl_module.device)

            with torch.no_grad():
                representations = self.get_representations(pl_module, x)
            
            if all_representations is None:
                all_representations = representations.detach()
            else:
                all_representations = torch.cat([all_representations,representations])

            if ys is None:
                ys = y
            else:
                ys = torch.cat([ys,y])

        return all_representations.numpy(), ys.numpy()

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

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.knn_evaluator = KNeighborsClassifier(n_neighbors=self.num_classes)

        train_dataloader = pl_module.train_dataloader()
        representations, y = self.get_all_representations(pl_module, train_dataloader)

        # knn fit
        pl_module.knn_evaluator.fit(representations, y)
        train_acc = pl_module.knn_evaluator.score(representations, y)

        # log metrics

        val_dataloader = pl_module.val_dataloader()
        representations, y = self.get_all_representations(pl_module, val_dataloader)

        # knn val acc
        val_acc = pl_module.knn_evaluator.score(representations, y)
        
        # log metrics
        pl_module.log('online_knn_train_acc', train_acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log('online_knn_val_acc', val_acc, on_step=False, on_epoch=True, sync_dist=True)
