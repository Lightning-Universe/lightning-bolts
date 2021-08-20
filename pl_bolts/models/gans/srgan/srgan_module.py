"""Adapted from: https://github.com/https-deeplearning-ai/GANs-Public."""
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Tuple
from warnings import warn

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from pl_bolts.callbacks import SRImageLoggerCallback
from pl_bolts.datamodules import TVTDataModule
from pl_bolts.datasets.utils import prepare_sr_datasets
from pl_bolts.models.gans.srgan.components import SRGANDiscriminator, SRGANGenerator, VGG19FeatureExtractor


class SRGAN(pl.LightningModule):
    """SRGAN implementation from the paper `Photo-Realistic Single Image Super-Resolution Using a Generative
    Adversarial Network <https://arxiv.org/abs/1609.04802>`__. It uses a pretrained SRResNet model as the generator
    if available.

    Code adapted from `https-deeplearning-ai/GANs-Public <https://github.com/https-deeplearning-ai/GANs-Public>`_ to
    Lightning by:

        - `Christoph Clement <https://github.com/chris-clem>`_

    You can pretrain a SRResNet model with :code:`srresnet_module.py`.

    Example::

        from pl_bolts.models.gan import SRGAN

        m = SRGAN()
        Trainer(gpus=1).fit(m)

    Example CLI::

        # CelebA dataset, scale_factor 4
        python srgan_module.py --dataset=celeba --scale_factor=4 --gpus=1

        # MNIST dataset, scale_factor 4
        python srgan_module.py --dataset=mnist --scale_factor=4 --gpus=1

        # STL10 dataset, scale_factor 4
        python srgan_module.py --dataset=stl10 --scale_factor=4 --gpus=1
    """

    def __init__(
        self,
        image_channels: int = 3,
        feature_maps_gen: int = 64,
        feature_maps_disc: int = 64,
        num_res_blocks: int = 16,
        scale_factor: int = 4,
        generator_checkpoint: Optional[str] = None,
        learning_rate: float = 1e-4,
        scheduler_step: int = 100,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            image_channels: Number of channels of the images from the dataset
            feature_maps_gen: Number of feature maps to use for the generator
            feature_maps_disc: Number of feature maps to use for the discriminator
            num_res_blocks: Number of res blocks to use in the generator
            scale_factor: Scale factor for the images (either 2 or 4)
            generator_checkpoint: Generator checkpoint created with SRResNet module
            learning_rate: Learning rate
            scheduler_step: Number of epochs after which the learning rate gets decayed
        """
        super().__init__()
        self.save_hyperparameters()

        if generator_checkpoint:
            self.generator = torch.load(generator_checkpoint)
        else:
            assert scale_factor in [2, 4]
            num_ps_blocks = scale_factor // 2
            self.generator = SRGANGenerator(image_channels, feature_maps_gen, num_res_blocks, num_ps_blocks)

        self.discriminator = SRGANDiscriminator(image_channels, feature_maps_disc)
        self.vgg_feature_extractor = VGG19FeatureExtractor(image_channels)

    def configure_optimizers(self) -> Tuple[List[torch.optim.Adam], List[torch.optim.lr_scheduler.MultiStepLR]]:
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.learning_rate)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.learning_rate)

        sched_disc = torch.optim.lr_scheduler.MultiStepLR(opt_disc, milestones=[self.hparams.scheduler_step], gamma=0.1)
        sched_gen = torch.optim.lr_scheduler.MultiStepLR(opt_gen, milestones=[self.hparams.scheduler_step], gamma=0.1)
        return [opt_disc, opt_gen], [sched_disc, sched_gen]

    def forward(self, lr_image: torch.Tensor) -> torch.Tensor:
        """Generates a high resolution image given a low resolution image.

        Example::

            srgan = SRGAN.load_from_checkpoint(PATH)
            hr_image = srgan(lr_image)
        """
        return self.generator(lr_image)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ) -> torch.Tensor:
        hr_image, lr_image = batch

        # Train discriminator
        result = None
        if optimizer_idx == 0:
            result = self._disc_step(hr_image, lr_image)

        # Train generator
        if optimizer_idx == 1:
            result = self._gen_step(hr_image, lr_image)

        return result

    def _disc_step(self, hr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        disc_loss = self._disc_loss(hr_image, lr_image)
        self.log("loss/disc", disc_loss, on_step=True, on_epoch=True)
        return disc_loss

    def _gen_step(self, hr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        gen_loss = self._gen_loss(hr_image, lr_image)
        self.log("loss/gen", gen_loss, on_step=True, on_epoch=True)
        return gen_loss

    def _disc_loss(self, hr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        real_pred = self.discriminator(hr_image)
        real_loss = self._adv_loss(real_pred, ones=True)

        _, fake_pred = self._fake_pred(lr_image)
        fake_loss = self._adv_loss(fake_pred, ones=False)

        disc_loss = 0.5 * (real_loss + fake_loss)

        return disc_loss

    def _gen_loss(self, hr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        fake, fake_pred = self._fake_pred(lr_image)

        perceptual_loss = self._perceptual_loss(hr_image, fake)
        adv_loss = self._adv_loss(fake_pred, ones=True)
        content_loss = self._content_loss(hr_image, fake)

        gen_loss = 0.006 * perceptual_loss + 0.001 * adv_loss + content_loss

        return gen_loss

    def _fake_pred(self, lr_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fake = self(lr_image)
        fake_pred = self.discriminator(fake)
        return fake, fake_pred

    @staticmethod
    def _adv_loss(pred: torch.Tensor, ones: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if ones else torch.zeros_like(pred)
        adv_loss = F.binary_cross_entropy_with_logits(pred, target)
        return adv_loss

    def _perceptual_loss(self, hr_image: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        real_features = self.vgg_feature_extractor(hr_image)
        fake_features = self.vgg_feature_extractor(fake)
        perceptual_loss = self._content_loss(real_features, fake_features)
        return perceptual_loss

    @staticmethod
    def _content_loss(hr_image: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(hr_image, fake)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--feature_maps_gen", default=64, type=int)
        parser.add_argument("--feature_maps_disc", default=64, type=int)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--scheduler_step", default=100, type=float)
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
    parser = SRGAN.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args(args)

    datasets = prepare_sr_datasets(args.dataset, args.scale_factor, args.data_dir)
    dm = TVTDataModule(*datasets, **vars(args))

    generator_checkpoint = Path(f"model_checkpoints/srresnet-{args.dataset}-scale_factor={args.scale_factor}.pt")
    if not generator_checkpoint.exists():
        warn(
            "No generator checkpoint found. Training generator from scratch. \
            Use srresnet_module.py to pretrain the generator."
        )
        generator_checkpoint = None

    model = SRGAN(
        **vars(args), image_channels=dm.dataset_test.image_channels, generator_checkpoint=generator_checkpoint
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[SRImageLoggerCallback(log_interval=args.log_interval, scale_factor=args.scale_factor)],
        logger=pl.loggers.TensorBoardLogger(
            save_dir="lightning_logs",
            name="srgan",
            version=f"{args.dataset}-scale_factor={args.scale_factor}",
            default_hp_metric=False,
        ),
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    cli_main()
