from typing import Optional, Tuple, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.accelerators import Accelerator
from torch import Tensor
from torch.nn import functional as F

from pl_bolts.utils.stability import under_review


@under_review()
class KNNOnlineEvaluator(Callback):
    """Weighted KNN online evaluator for self-supervised learning.
    The weighted KNN classifier matches sec 3.4 of https://arxiv.org/pdf/1805.01978.pdf.
    The implementation follows:
        1. https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
        2. https://github.com/leftthomas/SimCLR
        3. https://github.com/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
    Example::

        # your datamodule must have 2 attributes
        dm = DataModule()
        dm.num_classes = ... # the num of classes in the datamodule
        dm.name = ... # name of the datamodule (e.g. ImageNet, STL10, CIFAR10)

        online_eval = KNNOnlineEvaluator(
            k=100,
            temperature=0.1
        )
    """

    def __init__(self, k: int = 200, temperature: float = 0.07) -> None:
        """
        Args:
            k: k for k nearest neighbor
            temperature: temperature. See tau in section 3.4 of https://arxiv.org/pdf/1805.01978.pdf.
        """
        self.num_classes: Optional[int] = None
        self.dataset: Optional[int] = None
        self.k = k
        self.temperature = temperature

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        self.num_classes = trainer.datamodule.num_classes
        self.dataset = trainer.datamodule.name

    def predict(self, query_feature: Tensor, feature_bank: Tensor, target_bank: Tensor) -> Tensor:
        """
        Args:
            query_feature: (B, D) a batch of B query vectors with dim=D
            feature_bank: (N, D) the bank of N known vectors with dim=D
            target_bank: (N, ) the bank of N known vectors' labels

        Returns:
            (B, ) the predicted labels of B query vectors
        """

        B = query_feature.shape[0]

        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = query_feature @ feature_bank.T
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=self.k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(target_bank.expand(B, -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / self.temperature).exp()

        # counts for each class
        one_hot_label = torch.zeros(B * self.k, self.num_classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(B, -1, self.num_classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels

    def to_device(self, batch: Tensor, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor]:
        # get the labeled batch
        if self.dataset == "stl10":
            labeled_batch = batch[1]
            batch = labeled_batch

        inputs, y = batch

        # last input is for online eval
        x = inputs[-1]
        x = x.to(device)
        y = y.to(device)

        return x, y

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        assert not trainer.model.training

        # Skip Sanity Check as train_dataloader is not initialized during Sanity Check
        if trainer.train_dataloader is None:
            return

        total_top1, total_num, feature_bank, target_bank = 0.0, 0, [], []

        # go through train data to generate feature bank
        for batch in trainer.train_dataloader:
            x, target = self.to_device(batch, pl_module.device)
            feature = pl_module(x).flatten(start_dim=1)
            feature = F.normalize(feature, dim=1)

            feature_bank.append(feature)
            target_bank.append(target)

        # [N, D]
        feature_bank = torch.cat(feature_bank, dim=0)
        # [N]
        target_bank = torch.cat(target_bank, dim=0)

        # switch fo PL compatibility reasons
        accel = (
            trainer.accelerator_connector
            if hasattr(trainer, "accelerator_connector")
            else trainer._accelerator_connector
        )
        # gather representations from other gpus
        if accel.is_distributed:
            feature_bank = concat_all_gather(feature_bank, trainer.accelerator)
            target_bank = concat_all_gather(target_bank, trainer.accelerator)

        # go through val data to predict the label by weighted knn search
        for val_dataloader in trainer.val_dataloaders:
            for batch in val_dataloader:
                x, target = self.to_device(batch, pl_module.device)
                feature = pl_module(x).flatten(start_dim=1)
                feature = F.normalize(feature, dim=1)

                pred_labels = self.predict(feature, feature_bank, target_bank)

                total_num += x.shape[0]
                total_top1 += (pred_labels[:, 0] == target).float().sum().item()

        pl_module.log("online_knn_val_acc", total_top1 / total_num, on_step=False, on_epoch=True, sync_dist=True)


@under_review()
def concat_all_gather(tensor: Tensor, accelerator: Accelerator) -> Tensor:
    return accelerator.all_gather(tensor).view(-1, *tensor.shape[1:])
