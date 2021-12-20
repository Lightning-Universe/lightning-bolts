from typing import Any, Optional

import torch
from pytorch_lightning import LightningModule

from pl_bolts.models.detection.retinanet import create_retinanet_backbone
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.models.detection.retinanet import RetinaNet as torchvision_RetinaNet
    from torchvision.models.detection.retinanet import RetinaNetHead, retinanet_resnet50_fpn
    from torchvision.ops import box_iou
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class RetinaNet(LightningModule):
    """PyTorch Lightning implementation of RetinaNet.

    Paper: `Focal Loss for Dense Object Detection <https://arxiv.org/abs/1708.02002>`_.

    Paper authors: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár

    Model implemented by:
        - `Aditya Oke <https://github.com/oke-aditya>`

    During training, the model expects both the input tensors, as well as targets (list of dictionary), containing:
        - boxes (`FloatTensor[N, 4]`): the ground truth boxes in `[x1, y1, x2, y2]` format.
        - labels (`Int64Tensor[N]`): the class label for each ground truh box

    CLI command::

        # PascalVOC using LightningCLI
        python retinanet_module.py --trainer.gpus 1 --model.pretrained True
    """

    def __init__(
        self,
        learning_rate: float = 0.0001,
        num_classes: int = 91,
        backbone: Optional[str] = None,
        fpn: bool = True,
        pretrained: bool = False,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        **kwargs: Any,
    ):
        """
        Args:
            learning_rate: the learning rate
            num_classes: number of detection classes (including background)
            backbone: Pretained backbone CNN architecture.
            fpn: If True, creates a Feature Pyramind Network on top of Resnet based CNNs.
            pretrained: if true, returns a model pre-trained on COCO train2017
            pretrained_backbone: if true, returns a model with backbone pre-trained on Imagenet
            trainable_backbone_layers: number of trainable resnet layers starting from final block
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.backbone = backbone
        if backbone is None:
            self.model = retinanet_resnet50_fpn(pretrained=pretrained, **kwargs)

            self.model.head = RetinaNetHead(
                in_channels=self.model.backbone.out_channels,
                num_anchors=self.model.head.classification_head.num_anchors,
                num_classes=num_classes,
                **kwargs,
            )

        else:
            backbone_model = create_retinanet_backbone(
                self.backbone, fpn, pretrained_backbone, trainable_backbone_layers, **kwargs
            )
            self.model = torchvision_RetinaNet(backbone_model, num_classes=num_classes, **kwargs)

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch, batch_idx):

        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        preds = self.model(images)
        iou = torch.stack([self._evaluate_iou(p, t) for p, t in zip(preds, targets)]).mean()
        self.log("val_iou", iou, prog_bar=True)
        return {"val_iou": iou}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        self.log("val_avg_iou", avg_iou)

    def _evaluate_iou(self, preds, targets):
        """Evaluate intersection over union (IOU) for target from dataset and output prediction from model."""
        # no box detected, 0 IOU
        if preds["boxes"].shape[0] == 0:
            return torch.tensor(0.0, device=preds["boxes"].device)
        return box_iou(preds["boxes"], targets["boxes"]).diag().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.005,
        )


def cli_main():
    from pytorch_lightning.utilities.cli import LightningCLI

    from pl_bolts.datamodules import VOCDetectionDataModule

    LightningCLI(RetinaNet, VOCDetectionDataModule, seed_everything_default=42)


if __name__ == "__main__":
    cli_main()
