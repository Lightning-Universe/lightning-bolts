from argparse import ArgumentParser
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from pl_bolts.callbacks import SRImageLoggerCallback
from pl_bolts.datamodules import STL10_SR_DataModule
from pl_bolts.models.gans.srgan.components import SRGANDiscriminator, SRGANGenerator, VGG19FeatureExtractor


class SRGAN(pl.LightningModule):
    """
    SRGAN implementation from the paper `Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
    Network <https://arxiv.org/pdf/1609.04802.pdf>`_. It uses a pretrained SRResNet model as the generator.

    Example::

        from pl_bolts.models.gan import SRGAN

        m = SRGAN()
        Trainer(gpus=2).fit(m)

    Example CLI::

        # STL10 dataset
        python  srgan_module.py --gpus 1
    """

    def __init__(
        self,
        image_channels: int = 3,
        feature_maps_gen: int = 64,
        feature_maps_disc: int = 64,
        generator_checkpoint: Optional[str] = None,
        learning_rate: float = 0.0002,
        scheduler_step: int = 100,
        **kwargs
    ) -> None:
        """
        Args:
            image_channels: Number of channels of the images from the dataset
            feature_maps_gen: Number of feature maps to use for the generator
            feature_maps_disc: Number of feature maps to use for the discriminator
            generator_checkpoint: Generator checkpoint created with SRResNet module
            learning_rate: Learning rate
            scheduler_step: Number of epochs after which the learning rate gets decayed
        """
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.generator_checkpoint:
            self.generator = torch.load(self.hparams.generator_checkpoint)
        else:
            self.generator = SRGANGenerator(self.hparams.image_channels, self.hparams.feature_maps_gen)
        self.discriminator = SRGANDiscriminator(self.hparams.image_channels, self.hparams.feature_maps_disc)
        self.vgg_feature_extractor = VGG19FeatureExtractor()

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr)

        sched_disc = torch.optim.lr_scheduler.MultiStepLR(opt_disc, milestones=[self.hparams.scheduler_step], gamma=0.1)
        sched_gen = torch.optim.lr_scheduler.MultiStepLR(opt_gen, milestones=[self.hparams.scheduler_step], gamma=0.1)
        return [opt_disc, opt_gen], [sched_disc, sched_gen]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates a high resolution image given a low resolution image

        Example::

            gan = SRGAN.load_from_checkpoint(PATH)
            hr_image = gan(lr_image)
        """
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
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
        disc_loss = self._get_disc_loss(hr_image, lr_image)
        self.log("loss/disc", disc_loss, on_step=True, on_epoch=True, prog_bar=True)
        return disc_loss

    def _gen_step(self, hr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        gen_loss = self._get_gen_loss(hr_image, lr_image)
        self.log("loss/gen", gen_loss, on_step=True, on_epoch=True, prog_bar=True)
        return gen_loss

    def _get_disc_loss(self, hr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        real_pred = self.discriminator(hr_image)
        real_loss = self._get_adv_loss(real_pred, ones=True)

        _, fake_pred = self._get_fake_pred(lr_image)
        fake_loss = self._get_adv_loss(fake_pred, ones=False)

        disc_loss = 0.5 * (real_loss + fake_loss)

        return disc_loss

    def _get_gen_loss(self, hr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        fake, fake_pred = self._get_fake_pred(lr_image)
        adv_loss = self._get_adv_loss(fake_pred, ones=True)

        content_loss = self._get_content_loss(hr_image, fake)

        gen_loss = content_loss + 1e-3 * adv_loss

        return gen_loss

    def _get_fake_pred(self, lr_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fake = self(lr_image)
        fake_pred = self.discriminator(fake)
        return fake, fake_pred

    @staticmethod
    def _get_adv_loss(pred: torch.Tensor, ones: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if ones else torch.zeros_like(pred)
        adv_loss = F.binary_cross_entropy_with_logits(pred, target)
        return adv_loss

    def _get_content_loss(self, hr_image: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        real_features = self.vgg_feature_extractor(hr_image)
        fake_features = self.vgg_feature_extractor(fake)
        content_loss = F.mse_loss(real_features, fake_features)
        return content_loss

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--image_channels", default=3, type=int)
        parser.add_argument("--feature_maps_gen", default=64, type=int)
        parser.add_argument("--feature_maps_disc", default=64, type=int)
        parser.add_argument("--generator_checkpoint", default=None, type=str)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--scheduler_step", default=100, type=int)
        return parser


def cli_main(args=None):
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = STL10_SR_DataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SRGAN.add_model_specific_args(parser)
    args = parser.parse_args(args)

    model = SRGAN(**vars(args), scheduler_step=args.max_epochs // 2)
    dm = STL10_SR_DataModule.from_argparse_args(args)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[SRImageLoggerCallback()])
    trainer.fit(model, dm)

    torch.save(model.generator, "srgenerator.pt")
    torch.save(model.discriminator, "srdiscriminator.pt")


if __name__ == "__main__":
    cli_main()
