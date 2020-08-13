import torch
from torch import nn

from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn
import pytorch_lightning as pl

from pytorch_lightning.metrics import IoU
from argparse import ArgumentParser


class FasterRCNN(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.001,
        num_classes: int = 91,
        pretrained: bool = False,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        replace_head: bool = True,
        **kwargs,
    ):
        """
        PyTorch Lightning implementation of `Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks https://arxiv.org/abs/1506.01497`_
        Paper authors: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun

        Model implemented by:
            - `Teddy Koker <https://github.com/teddykoker>`

        TODO
        Args:
            datamodule: The datamodule

        """
        super().__init__()

        model = fasterrcnn_resnet50_fpn(
            num_classes=num_classes,
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=trainable_backbone_layers,
        )

        if replace_head:
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            head = faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            model.roi_heads.box_predictor = head
        else:
            assert num_classes == 91, "replace_head must be true to change num_classes"

        self.model = model
        self.metric = IoU()

    def forward(self, x):
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
        iou = torch.stack([self.metric(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        logs = {"val_iou": avg_iou}
        return {"avg_val_iou": avg_iou, "log": logs}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--num_classes", type=int, default=91)
        parser.add_argument("--pretrained", type=bool, default=False)
        parser.add_argument("--pretrained_backbone", type=bool, default=True)
        parser.add_argument("--trainable_backbone_layers", type=int, default=3)
        parser.add_argument("--replace_head", type=bool, default=True)
        return parser
