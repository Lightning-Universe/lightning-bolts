from argparse import ArgumentParser
from typing import Any, Optional

import pytorch_lightning as pl
import torch

from pl_bolts.metrics.object_detection import _evaluate_iou
from pl_bolts.models.detection.retinanet import create_retinanet_backbone
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.models.detection.retinanet import RetinaNet as torchvision_RetinaNet
    from torchvision.models.detection.retinanet import RetinaNetHead, retinanet_resnet50_fpn
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class RetinaNet(pl.LightningModule):
    """PyTorch Lightning implementation of Retina Net `Focal Loss for Dense Object Detection.

    <https://arxiv.org/abs/1708.02002>`_.

    Paper authors: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár

    Model implemented by:
        - `Aditya Oke <https://github.com/oke-aditya>`

    During training, the model expects both the input tensors, as well as targets (list of dictionary), containing:
        - boxes (`FloatTensor[N, 4]`): the ground truth boxes in `[x1, y1, x2, y2]` format.
        - labels (`Int64Tensor[N]`): the class label for each ground truh box

    CLI command::

        # PascalVOC
        python retinanet_module.py --gpus 1 --pretrained True
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

        self.save_hyperparameters()

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

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        outs = self.model(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        self.log("val_iou", iou, prog_bar=True)

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        logs = {"val_iou": avg_iou}
        return {"avg_val_iou": avg_iou, "log": logs}

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.005,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--num_classes", type=int, default=91)
        parser.add_argument("--backbone", type=str, default=None)
        parser.add_argument("--fpn", type=bool, default=True)
        parser.add_argument("--pretrained", type=bool, default=False)
        parser.add_argument("--pretrained_backbone", type=bool, default=True)
        parser.add_argument("--trainable_backbone_layers", type=int, default=3)
        return parser


def cli_main():
    from pl_bolts.datamodules import VOCDetectionDataModule

    pl.seed_everything(42)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=1)
    parser = RetinaNet.add_model_specific_args(parser)

    args = parser.parse_args()

    datamodule = VOCDetectionDataModule.from_argparse_args(args)
    args.num_classes = datamodule.num_classes

    model = RetinaNet(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    cli_main()
