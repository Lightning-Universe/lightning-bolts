from argparse import ArgumentParser
from warnings import warn

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pl_bolts.models.detection.detr import detr_loss


class Detr(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        num_classes: int = 91,
        num_queries: int = 5,
        pretrained: bool = False,
        backbone: str = 'detr_resnet50',
        **kwargs,
    ):

        """
            PyTorch Lightning implementation of `Detr: End-to-End Object Detection with Transformers
            Paper authors: Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier,
                           Alexander Kirillov, and Sergey Zagoruyko
            Model implemented by:
                - `Aditya Oke <https://github.com/oke-aditya>`
            During training, the model expects both the input tensors, as well as targets (list of dictionary),
            containing:
                - boxes (`FloatTensor[N, 4]`): the ground truth boxes in `[x1, y1, x2, y2]` format.
                - labels (`Int64Tensor[N]`): the class label for each ground truh box
            CLI command::
                # PascalVOC
                python faster_rcnn.py --gpus 1 --pretrained True
            Args:
                learning_rate: the learning rate
                num_classes: number of detection classes (including background)
                num_queries: number of queries to the transformer module.
                pretrained: if true, returns a model pre-trained on COCO train2017
                backbone:  'detr_resnet50' or 'detr_resnet101' backbone as passed.
                trainable_backbone_layers: number of trainable resnet layers starting from final block
        """

        super().__init__()

        model = torch.hub.load('facebookresearch/detr', backbone, pretrained=pretrained)
        in_features = model.class_embed.in_features
        model.class_embed = nn.Linear(in_features=in_features, out_features=num_classes)
        model.num_queries = num_queries

        self.model = model
        self.learning_rate = learning_rate

        matcher = detr_loss.HungarianMatcher()
        weight_dict = {"loss_ce": 1, "loss_bbox": 1, "loss_giou": 1}
        losses = ['labels', 'boxes', 'cardinality']
        self.criterion = detr_loss.SetCriterion(num_classes - 1, matcher, weight_dict, eos_coef=0.5, losses=losses)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images)

        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        return {"loss": loss, "log": loss_dict}

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
        parser.add_argument("--num_queries", type=int, default=5)
        parser.add_argument("--pretrained", type=bool, default=True)
        parser.add_argument("--pretrained_backbone", type=str, default="resnet50")

        parser.add_argument("--data_dir", type=str, default=".")
        parser.add_argument("--batch_size", type=int, default=1)
        return parser


def run_cli():
    # from pl_bolts.datamodules import VOCDetectionDataModule

    pl.seed_everything(42)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Detr.add_model_specific_args(parser)

    args = parser.parse_args()

    # datamodule = VOCDetectionDataModule.from_argparse_args(args)
    # args.num_classes = datamodule.num_classes

    # model = Detr(**vars(args))
    # trainer = pl.Trainer.from_argparse_args(args)
    # trainer.fit(model, datamodule)


if __name__ == "__main__":
    run_cli()
