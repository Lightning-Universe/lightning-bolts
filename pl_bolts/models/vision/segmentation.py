from argparse import ArgumentParser
from typing import Any, Dict, Optional

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import Tensor
from torch.nn import functional as F

from pl_bolts.models.vision.unet import UNet


class SemSegment(LightningModule):
    """Basic model for semantic segmentation. Uses UNet architecture by default.

    The default parameters in this model are for the KITTI dataset. Note, if you'd like to use this model as is,
    you will first need to download the KITTI dataset yourself. You can download the dataset `here.
    <http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015>`_

    Implemented by:

        - `Annika Brundyn <https://github.com/annikabrundyn>`_

    Example::

        from pl_bolts.models.vision import SemSegment

        model = SemSegment(num_classes=19)
        dm = KittiDataModule(data_dir='/path/to/kitti/')

        Trainer().fit(model, datamodule=dm)

    Example CLI::

        # KITTI
        python segmentation.py --data_dir /path/to/kitti/ --accelerator=gpu
    """

    def __init__(
        self,
        num_classes: int = 19,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
        ignore_index: Optional[int] = 250,
        lr: float = 0.01,
        **kwargs: Any
    ):
        """
        Args:
            num_classes: number of output classes (default 19)
            num_layers: number of layers in each side of U-net (default 5)
            features_start: number of features in first layer (default 64)
            bilinear: whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.
            ignore_index: target value to be ignored in cross_entropy (default 250)
            lr: learning rate (default 0.01)
        """

        super().__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        if ignore_index is None:
            # set ignore_index to default value of F.cross_entropy if it is None.
            self.ignore_index = -100
        else:
            self.ignore_index = ignore_index
        self.lr = lr

        self.net = UNet(
            num_classes=num_classes,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def training_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Any]:
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=self.ignore_index)
        log_dict = {"train_loss": loss_val}
        return {"loss": loss_val, "log": log_dict, "progress_bar": log_dict}

    def validation_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Any]:
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=self.ignore_index)
        return {"val_loss": loss_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        log_dict = {"val_loss": loss_val}
        return {"log": log_dict, "val_loss": log_dict["val_loss"], "progress_bar": log_dict}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    @staticmethod
    def add_model_specific_args(parent_parser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
        parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument(
            "--bilinear", action="store_true", default=False, help="whether to use bilinear interpolation or transposed"
        )

        return parser


def cli_main():
    from pl_bolts.datamodules import KittiDataModule

    seed_everything(1234)

    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)
    # model args
    parser = SemSegment.add_model_specific_args(parser)
    # datamodule args
    parser = KittiDataModule.add_argparse_args(parser)

    args = parser.parse_args()

    # data
    dm = KittiDataModule(args.data_dir).from_argparse_args(args)

    # model
    model = SemSegment(**args.__dict__)

    # train
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()
