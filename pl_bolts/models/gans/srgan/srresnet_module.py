"""Adapted from: https://github.com/https-deeplearning-ai/GANs-Public."""
from argparse import ArgumentParser
from typing import Any, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from pl_bolts.callbacks import SRImageLoggerCallback
from pl_bolts.datamodules import TVTDataModule
from pl_bolts.datasets.utils import prepare_sr_datasets
from pl_bolts.models.gans.srgan.components import SRGANGenerator


class SRResNet(pl.LightningModule):
    """SRResNet implementation from the paper `Photo-Realistic Single Image Super-Resolution Using a Generative
    Adversarial Network <https://arxiv.org/abs/1609.04802>`__. A pretrained SRResNet model is used as the generator
    for SRGAN.

    Code adapted from `https-deeplearning-ai/GANs-Public <https://github.com/https-deeplearning-ai/GANs-Public>`_ to
    Lightning by:

        - `Christoph Clement <https://github.com/chris-clem>`_

    Example::

        from pl_bolts.models.gan import SRResNet

        m = SRResNet()
        Trainer(gpus=1).fit(m)

    Example CLI::

        # CelebA dataset, scale_factor 4
        python srresnet_module.py --dataset=celeba --scale_factor=4 --gpus=1

        # MNIST dataset, scale_factor 4
        python srresnet_module.py --dataset=mnist --scale_factor=4 --gpus=1

        # STL10 dataset, scale_factor 4
        python srresnet_module.py --dataset=stl10 --scale_factor=4 --gpus=1
    """

    def __init__(
        self,
        image_channels: int = 3,
        feature_maps: int = 64,
        num_res_blocks: int = 16,
        scale_factor: int = 4,
        learning_rate: float = 1e-4,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            image_channels: Number of channels of the images from the dataset
            feature_maps: Number of feature maps to use
            num_res_blocks: Number of res blocks to use in the generator
            scale_factor: Scale factor for the images (either 2 or 4)
            learning_rate: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()

        assert scale_factor in [2, 4]
        num_ps_blocks = scale_factor // 2
        self.srresnet = SRGANGenerator(image_channels, feature_maps, num_res_blocks, num_ps_blocks)

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.srresnet.parameters(), lr=self.hparams.learning_rate)

    def forward(self, lr_image: torch.Tensor) -> torch.Tensor:
        """Creates a high resolution image given a low resolution image.

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

    parser = TVTDataModule.add_argparse_args(parser)
    parser = SRResNet.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    datasets = prepare_sr_datasets(args.dataset, args.scale_factor, args.data_dir)
    dm = TVTDataModule(*datasets, **vars(args))

    model = SRResNet(**vars(args), image_channels=dm.dataset_train.dataset.image_channels)

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
