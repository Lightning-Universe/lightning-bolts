from argparse import ArgumentParser
from typing import Any, Optional

import pytorch_lightning as pl
import torch

from pl_bolts.utils.warnings import warn_missing_pkg

try:
    from torchvision.models.detection.faster_rcnn import FasterRCNN as torchvision_FasterRCNN
    from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor
    from torchvision.ops import box_iou

    from pl_bolts.models.detection.faster_rcnn import create_fasterrcnn_backbone
except ModuleNotFoundError:
    warn_missing_pkg("torchvision")  # pragma: no-cover


def _evaluate_iou(target, pred):
    """
    Evaluate intersection over union (IOU) for target from dataset and output prediction
    from model
    """
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()


class FasterRCNN(pl.LightningModule):
    """
    PyTorch Lightning implementation of `Faster R-CNN: Towards Real-Time Object Detection with
    Region Proposal Networks <https://arxiv.org/abs/1506.01497>`_.

    Paper authors: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun

    Model implemented by:
        - `Teddy Koker <https://github.com/teddykoker>`

    During training, the model expects both the input tensors, as well as targets (list of dictionary), containing:
        - boxes (`FloatTensor[N, 4]`): the ground truth boxes in `[x1, y1, x2, y2]` format.
        - labels (`Int64Tensor[N]`): the class label for each ground truh box

    CLI command::

        # PascalVOC
        python faster_rcnn.py --gpus 1 --pretrained True
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
            self.model = fasterrcnn_resnet50_fpn(
                pretrained=pretrained,
                pretrained_backbone=pretrained_backbone,
                trainable_backbone_layers=trainable_backbone_layers,
            )

            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, self.num_classes
            )

        else:
            backbone_model = create_fasterrcnn_backbone(
                self.backbone,
                fpn,
                pretrained_backbone,
                trainable_backbone_layers,
                **kwargs,
            )
            self.model = torchvision_FasterRCNN(
                backbone_model, num_classes=num_classes, **kwargs
            )

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch, batch_idx):

        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        outs = self.model(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou}

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


def run_cli():
    from pl_bolts.datamodules import VOCDetectionDataModule

    pl.seed_everything(42)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--batch_size", type=int, default=1)
    parser = FasterRCNN.add_model_specific_args(parser)

    args = parser.parse_args()

    datamodule = VOCDetectionDataModule.from_argparse_args(args)
    args.num_classes = datamodule.num_classes

    model = FasterRCNN(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    run_cli()
