from argparse import ArgumentParser
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Callback
from torchvision.utils import make_grid

from pl_bolts.models.gans.srgan.components import SRGANDiscriminator, SRGANGenerator, VGG19FeatureExtractor
from pl_bolts.utils.warnings import warn_missing_pkg

try:
    from torchvision import transforms as transform_lib
except ModuleNotFoundError:
    warn_missing_pkg('torchvision')  # pragma: no-cover
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True


class SRGAN(pl.LightningModule):
    def __init__(
        self,
        beta1: float = 0.5,
        beta2: float = 0.999,
        image_channels: int = 1,
        feature_maps_gen: int = 64,
        feature_maps_disc: int = 64,
        latent_dim: int = 100,
        learning_rate: float = 0.0002,
        **kwargs
    ):
        """
        Args:
            beta1: Beta1 value for Adam optimizer
            beta2: Beta2 value for Adam optimizer
            image_channels: Number of channels of the images from the dataset
            feature_maps_gen: Number of feature maps to use for the generator
            feature_maps_disc: Number of feature maps to use for the discriminator
            learning_rate: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()

        self.generator = SRGANGenerator(self.hparams.image_channels, self.hparams.feature_maps_gen)
        self.discriminator = SRGANDiscriminator(self.hparams.image_channels, self.hparams.feature_maps_gen)
        self.vgg_feature_extractor = VGG19FeatureExtractor()

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        betas = (self.hparams.beta1, self.hparams.beta2)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        return [opt_disc, opt_gen], []

    def forward(self, x):
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
        self.log("loss/disc", disc_loss, on_epoch=True, prog_bar=True)
        return disc_loss

    def _gen_step(self, hr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        gen_loss = self._get_gen_loss(hr_image, lr_image)
        self.log("loss/gen", gen_loss, on_epoch=True, prog_bar=True)
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
        parser.add_argument("--beta1", default=0.5, type=float)
        parser.add_argument("--beta2", default=0.999, type=float)
        parser.add_argument("--image_channels", default=3, type=int)
        parser.add_argument("--feature_maps_gen", default=64, type=int)
        parser.add_argument("--feature_maps_disc", default=64, type=int)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        return parser


class TensorboardImageLoggerCallback(Callback):
    def __init__(self, log_interval: int = 100, num_samples: int = 3):
        super().__init__()
        self.log_interval = log_interval
        self.num_samples = num_samples

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        global_step = pl_module.global_step
        if global_step % self.log_interval == 0:
            hr_image, lr_image = batch
            hr_image, lr_image = hr_image.to(pl_module.device), lr_image.to(pl_module.device)
            lr_image_scaled = F.interpolate(lr_image, scale_factor=4)
            hr_fake = pl_module(lr_image)

            hr_image_grid = make_grid(hr_image[:self.num_samples], nrow=1, normalize=True)
            lr_image_grid = make_grid(lr_image_scaled[:self.num_samples], nrow=1, normalize=True)
            hr_fake_grid = make_grid(hr_fake[:self.num_samples], nrow=1, normalize=True)

            grid = torch.cat((lr_image_grid, hr_image_grid, hr_fake_grid), -1)
            title = f"images_step_{global_step}"
            trainer.logger.experiment.add_image(title, grid, global_step=global_step)


def cli_main(args=None):
    from pl_bolts.datamodules.stl10_sr_datamodule import STL10_SR_DataModule

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser = STL10_SR_DataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SRGAN.add_model_specific_args(parser)
    args = parser.parse_args(args)

    model = SRGAN(**vars(args))
    dm = STL10_SR_DataModule(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[TensorboardImageLoggerCallback()])
    trainer.fit(model, dm)

    return dm, model, trainer


if __name__ == "__main__":
    dm, model, trainer = cli_main()
