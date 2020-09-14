from argparse import ArgumentParser
from warnings import warn

import pytorch_lightning as pl
import torch

try:
    from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn
    from torchvision.ops import box_iou
except ImportError:
    warn('You want to use `torchvision` which is not installed yet,'  # pragma: no-cover
         ' install it with `pip install torchvision`.')


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
    def __init__(
        self,
        learning_rate: float = 0.0001,
        num_classes: int = 91,
        pretrained: bool = False,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        replace_head: bool = True,
        **kwargs,
    ):
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

        Args:
            learning_rate: the learning rate
            num_classes: number of detection classes (including background)
            pretrained: if true, returns a model pre-trained on COCO train2017
            pretrained_backbone: if true, returns a model with backbone pre-trained on Imagenet
            trainable_backbone_layers: number of trainable resnet layers starting from final block
        """
        super().__init__()

        model = fasterrcnn_resnet50_fpn(
            # num_classes=num_classes,
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
        self.learning_rate = learning_rate

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
        parser.add_argument("--pretrained", type=bool, default=False)
        parser.add_argument("--pretrained_backbone", type=bool, default=True)
        parser.add_argument("--trainable_backbone_layers", type=int, default=3)
        parser.add_argument("--replace_head", type=bool, default=True)

        parser.add_argument("--data_dir", type=str, default=".")
        parser.add_argument("--batch_size", type=int, default=1)
        return parser


def run_cli():
    from pl_bolts.datamodules import VOCDetectionDataModule

    pl.seed_everything(42)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = FasterRCNN.add_model_specific_args(parser)

    args = parser.parse_args()

    datamodule = VOCDetectionDataModule.from_argparse_args(args)
    args.num_classes = datamodule.num_classes

    model = FasterRCNN(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    run_cli()
