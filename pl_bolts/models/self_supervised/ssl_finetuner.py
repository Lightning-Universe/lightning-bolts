import math
from typing import List, Optional

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics import Accuracy

from pl_bolts.models.self_supervised import SSLEvaluator


class SSLFineTuner(LightningModule):
    """Finetunes a self-supervised learning backbone using the standard evaluation protocol of a singler layer MLP
    with 1024 units.

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
        dropout: float = 0.0,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-6,
        nesterov: bool = False,
        scheduler_type: str = "cosine",
        decay_epochs: List = [60, 80],
        gamma: float = 0.1,
        final_lr: float = 0.0,
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

        # relic

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)

        acc = self.train_acc(logits.softmax(-1), y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc_step", acc, prog_bar=True)
        self.log("train_acc_epoch", self.train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.val_acc(logits.softmax(-1), y)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc)

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.test_acc(logits.softmax(-1), y)

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", self.test_acc)

        return loss

    def shared_step(self, batch, image_list=False):
        if image_list:
            x_list, y = batch
            loss_list, logits_list = [], []
            for x in x_list:
                with torch.no_grad():
                    feats = self.backbone(x)
                feats = feats.view(feats.size(0), -1)
                logits = self.linear_layer(feats)
                logits_list.append(logits)
                loss_list.append(F.cross_entropy(logits, y))

            return loss_list, logits_list, y
        else:
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
                optimizer, self.epochs, eta_min=self.final_lr  # total epochs to run
            )

        return [optimizer], [scheduler]


class RelicDALearner(LightningModule):
    """Finetunes a self-supervised learning backbone using the standard evaluation protocol of a singler layer MLP
    with 1024 units.

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
        dropout: float = 0.0,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-6,
        nesterov: bool = False,
        scheduler_type: str = "cosine",
        decay_epochs: List = [60, 80],
        gamma: float = 0.1,
        final_lr: float = 0.0,
        alfa: float = 0.1,
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
        for params in self.backbone.parameters():
            params.requires_grad = False
        self.data_augmentation = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        # self.data_augmentation = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=1, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 16, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 3, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(3),
        #     nn.ReLU(),
        # )
        # self.data_augmentation = MLP_Augmentation()
        # print(self.backbone)
        print(self.data_augmentation)
        
        # relic params
        self.alfa = alfa

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        # print(batch[0][0].shape)  # torch.Size([256, 3, 32, 32])
        loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss)
        return loss

    def forward(self, x):
        x = self.data_augmentation(x)
        return self.backbone(x)

    def shared_step(self, batch):
        img_list, y = batch
        i = self.data_augmentation(img_list[-1])
        z_list = [self.backbone(i)]  # img_list[-1] is the original image.
        for img in img_list[:-1]:
            z_list.append(self.backbone(img))
        return self.relic_loss(z_list, self.alfa)

    def relic_loss(self, z_list, alfa=0.1):

        _nt_xent_loss, _relic_loss, p_do_list = 0, 0, []
        batch_size, device = z_list[0].shape[0], z_list[0].device
        mask = torch.ones([batch_size, batch_size], device=device) - torch.eye(batch_size, device=device)

        for i in range(len(z_list) - 1):
            for j in range(i + 1, len(z_list)):
                _loss, p_do = self.nt_xent_loss(z_list[i], z_list[j])
                _nt_xent_loss += _loss
                p_do_list.append(p_do)

        for i in range(len(p_do_list) - 1):
            for j in range(i + 1, len(p_do_list)):
                do1_log = p_do_list[i].log() * mask
                do2 = p_do_list[j] * mask
                _relic_loss += nn.KLDivLoss()(do1_log, do2)

        self.log('_nt_xent_loss', _nt_xent_loss)
        self.log('_relic_loss', _relic_loss)
        loss = _nt_xent_loss + alfa * _relic_loss

        return loss

    def nt_xent_loss(self, out_1, out_2, temperature=0.5, eps=1e-6):
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        out_1_dist = out_1
        out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        # import ipdb; ipdb.set_trace()
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss, sim[out_1.shape[0] :, : out_1.shape[0]]  # sim[] is for use_relic_loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.data_augmentation.parameters(),
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
                optimizer, self.epochs, eta_min=self.final_lr  # total epochs to run
            )

        return [optimizer], [scheduler]


class MLP_Augmentation(LightningModule):
    def __init__(self, input_size=3 * 32 * 32, output_size=3 * 32 * 32, num_layer=3, hidden_dim=[128, 32, 128]):
        super().__init__()
        assert num_layer == len(hidden_dim)
        mlp_list = []
        zView()
        for idx, dim in enumerate(hidden_dim):
            if idx == 0:
                mlp_list.append(nn.Linear(input_size, dim))
            else:
                mlp_list.append(nn.Linear(pre_dim, dim))
            mlp_list.append(nn.ReLU())
            if idx == num_layer - 1:
                mlp_list.append(nn.Linear(dim, output_size))
            pre_dim = dim
        self.model = nn.Sequential(*mlp_list)

    def forward(self, x):
        return self.model(x)


class View(LightningModule):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


if __name__ == "__main__":
    # test = MLP_Augmentation()
    input = [256, 3, 32, 32]
    padding = 0
    dilation = 1
    kernel_size = 1
    In_w = input[2]

    h = 1 + (In_w + 2 * padding - dilation * (kernel_size - 1) - 1) / 1
    print(h)

    """
    256, 3, 32, 32  => 256, 16, 32, 32 (padding=1, dilation=1, kernel_size=3, strade(1))
    256, 16, 32, 32 => 256, 3, 32, 32
    """
