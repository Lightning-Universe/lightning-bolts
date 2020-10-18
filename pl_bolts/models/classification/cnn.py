from argparse import ArgumentParser
from warnings import warn
import pytorch_lightning as pl
import torch
from pl_bolts.models.classification.cnn_backbones import create_torchvision_backbone
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning.metrics.functional import accuracy


__all__ = ["CNN"]


class CNN(pl.LightningModule):
    """
    Creates a CNN which can be fine-tuned.
    """

    def __init__(self,
                 pretrained_backbone: str,
                 learning_rate: float = 0.0001,
                 num_classes: int = 100,
                 pretrained: bool = True,
                 **kwargs,):

        super().__init__()
        self.num_classes = num_classes
        self.bottom, self.out_channels = create_torchvision_backbone(pretrained_backbone,
                                                                     num_classes, pretrained=pretrained)
        self.top = nn.Linear(self.out_channels, self.num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.bottom(x)
        x = self.top(x.view(-1, self.out_channels))
        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)
        train_loss = F.cross_entropy(outputs, targets, reduction='sum')
        # Possible we can compute top-1 and top-5 accuracy here.
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)
        val_loss = F.cross_entropy(outputs, targets, reduction='sum')
        # Possible we can compute top-1 and top-5 accuracy here.
        return {"loss": val_loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--pretrained_backbone", type=str, default="resnet50")
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--num_classes", type=int, default=100)
        parser.add_argument("--pretrained", type=bool, default=False)
        parser.add_argument("--data_dir", type=str, default=".")
        parser.add_argument("--batch_size", type=int, default=1)
        return parser


def run_cli():
    from pl_bolts.datamodules import CIFAR10DataModule
    pl.seed_everything(42)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CNN.add_model_specific_args(parser)
    args = parser.parse_args()

    datamodule = CIFAR10DataModule.from_argparse_args(args)
    args.num_classes = datamodule.num_classes

    model = CNN(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    run_cli()
