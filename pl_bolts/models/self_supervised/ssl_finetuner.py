from typing import List, Optional

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torchmetrics import Accuracy

from pl_bolts.models.self_supervised import SSLEvaluator


class SSLFineTuner(pl.LightningModule):
    """
    Finetunes a self-supervised learning backbone using the standard evaluation protocol of a singler layer MLP
    with 1024 units

    Example::

        from pl_bolts.utils.self_supervised import SSLFineTuner
        from pl_bolts.models.self_supervised import CPC_v2
        from pl_bolts.datamodules import CIFAR10DataModule
        from pl_bolts.models.self_supervised.cpc.transforms import CPCEvalTransformsCIFAR10,
                                                                    CPCTrainTransformsCIFAR10

        # pretrained model
        backbone = CPC_v2.load_from_checkpoint(PATH, strict=False)

        # dataset + transforms
        dm = CIFAR10DataModule(data_dir='.')
        dm.train_transforms = CPCTrainTransformsCIFAR10()
        dm.val_transforms = CPCEvalTransformsCIFAR10()

        # finetuner
        finetuner = SSLFineTuner(backbone, in_features=backbone.z_dim, num_classes=backbone.num_classes)

        # train
        trainer = pl.Trainer()
        trainer.fit(finetuner, dm)

        # test
        trainer.test(datamodule=dm)
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        in_features: int = 2048,
        num_classes: int = 1000,
        epochs: int = 100,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-6,
        nesterov: bool = False,
        scheduler_type: str = 'cosine',
        decay_epochs: List = [60, 80],
        gamma: float = 0.1,
        final_lr: float = 0.
    ):
        """
        Args:
            backbone: a pretrained model
            in_features: feature dim of backbone outputs
            num_classes: classes of the dataset
            hidden_dim: dim of the MLP (1024 default used in self-supervised literature)
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.nesterov = nesterov
        self.weight_decay = weight_decay

        self.scheduler_type = scheduler_type
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.final_lr = final_lr

        self.backbone = backbone
        self.linear_layer = SSLEvaluator(n_input=in_features, n_classes=num_classes, p=dropout, n_hidden=hidden_dim)

        # metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        self.test_acc = Accuracy(compute_on_step=False)

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        acc = self.train_acc(logits, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc_step', acc, prog_bar=True)
        self.log('train_acc_epoch', self.train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.val_acc(logits, y)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc)

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.test_acc(logits, y)

        self.log('test_loss', loss, sync_dist=True)
        self.log('test_acc', self.test_acc)

        return loss

    def shared_step(self, batch):
        x, y = batch

        with torch.no_grad():
            feats = self.backbone(x)

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        loss = F.cross_entropy(logits, y)

        return loss, logits, y

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.linear_layer.parameters(),
            lr=self.learning_rate,
            nesterov=self.nesterov,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.decay_epochs, gamma=self.gamma)
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                self.epochs,
                eta_min=self.final_lr  # total epochs to run
            )

        return [optimizer], [scheduler]
