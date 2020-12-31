from argparse import ArgumentParser

from torch.utils.data.dataset import random_split
from pl_bolts.datasets.stl10_sr_dataset import SRSTL10Dataset
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from pl_bolts.callbacks import SRImageLoggerCallback
from pl_bolts.datamodules.sr_datamodule import SRDataModule
from pl_bolts.datasets.mnist_dataset import SRMNISTDataset
from pl_bolts.datasets.sr_celeba_dataset import SRCelebADataset
from pl_bolts.models.gans.srgan.components import SRGANGenerator


class SRResNet(pl.LightningModule):
    """
    SRResNet implementation from the paper `Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
    Network <https://arxiv.org/pdf/1609.04802.pdf>`_. A pretrained model is used as the generator for SRGAN.

    Example::

        from pl_bolts.models.gan import SRResNet

        m = SRResNet()
        Trainer(gpus=1).fit(m)

    Example CLI::

        # STL10_SR_DataModule
        python ssresnetmodule.py --gpus 1
    """

    def __init__(
        self,
        image_channels: int = 3,
        feature_maps: int = 64,
        num_res_blocks: int = 16,
        scale_factor: int = 4,
        learning_rate: float = 1e-4,
        **kwargs,
    ) -> None:
        """
        Args:
            image_channels: Number of channels of the images from the dataset
            feature_maps: Number of feature maps to use
            learning_rate: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()

        assert scale_factor in [2, 4]
        num_ps_blocks = scale_factor // 2

        self.srresnet = SRGANGenerator(image_channels, feature_maps, num_res_blocks, num_ps_blocks)

    def configure_optimizers(self):
        return torch.optim.Adam(self.srresnet.parameters(), lr=self.hparams.learning_rate)

    def forward(self, lr_image: torch.Tensor) -> torch.Tensor:
        """
        Creates a high resolution image given a low resolution image

        Example::

            srresnet = SRResNet.load_from_checkpoint(PATH)
            hr_image = srresnet(lr_image)
        """
        return self.srresnet(lr_image)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("loss/train", loss, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("loss/val", loss, sync_dist=True)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("loss/test", loss, sync_dist=True)
        return loss

    def _loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hr_image, lr_image = batch
        fake = self(lr_image)
        loss = F.mse_loss(hr_image, fake)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--feature_maps", default=64, type=int)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--num_res_blocks", default=16, type=int)
        return parser


def cli_main(args=None):
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--dataset", default="mnist", type=str, choices=["celeba", "mnist", "stl10"])
    parser.add_argument("--data_dir", default="./", type=str)
    parser.add_argument("--log_interval", default=1000, type=int)
    parser.add_argument("--scale_factor", default=4, type=int)
    parser.add_argument("--save_model_checkpoint", dest="save_model_checkpoint", action="store_true")

    script_args, _ = parser.parse_known_args(args)

    if script_args.dataset == "celeba":
        hr_image_size = 128
        lr_image_size = hr_image_size // script_args.scale_factor
        image_channels = 3
        dataset_cls = SRCelebADataset

        dataset_train = dataset_cls(
            hr_image_size, lr_image_size, image_channels, root=script_args.data_dir, split="train"
        )
        dataset_val = dataset_cls(
            hr_image_size, lr_image_size, image_channels, root=script_args.data_dir, split="valid"
        )
        dataset_test = dataset_cls(
            hr_image_size, lr_image_size, image_channels, root=script_args.data_dir, split="test"
        )

    elif script_args.dataset == "mnist":
        hr_image_size = 28
        lr_image_size = hr_image_size // script_args.scale_factor
        image_channels = 1
        dataset_cls = SRMNISTDataset

        dataset_dev = dataset_cls(hr_image_size, lr_image_size, image_channels, root=script_args.data_dir, train=True)
        dataset_train, dataset_val = random_split(dataset_dev, lengths=[55_000, 5_000])
        dataset_test = dataset_cls(hr_image_size, lr_image_size, image_channels, root=script_args.data_dir, train=False)

    elif script_args.dataset == "stl10":
        hr_image_size = 96
        lr_image_size = hr_image_size // script_args.scale_factor
        image_channels = 3
        dataset_cls = SRSTL10Dataset

        dataset_dev = dataset_cls(
            hr_image_size, lr_image_size, image_channels, root=script_args.data_dir, split="train"
        )
        dataset_train, dataset_val = random_split(dataset_dev, lengths=[4_500, 500])
        dataset_test = dataset_cls(
            hr_image_size, lr_image_size, image_channels, root=script_args.data_dir, split="test"
        )

    parser = SRDataModule.add_argparse_args(parser)
    parser = SRResNet.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    dm = SRDataModule(dataset_train, dataset_val, dataset_test)
    model = SRResNet(**vars(args), image_channels=image_channels)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[SRImageLoggerCallback(log_interval=args.log_interval, scale_factor=args.scale_factor)],
        logger=pl.loggers.TensorBoardLogger(
            save_dir="lightning_logs",
            name="srresnet",
            version=f"{args.dataset}-scale_factor={args.scale_factor}",
            default_hp_metric=False,
        ),
    )
    trainer.fit(model, dm)

    if args.save_model_checkpoint:
        torch.save(model.srresnet, f"model_checkpoints/srresnet-{args.dataset}-scale_factor={args.scale_factor}.pt")


if __name__ == "__main__":
    cli_main()
