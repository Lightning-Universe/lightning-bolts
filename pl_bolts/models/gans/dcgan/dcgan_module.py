from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch import nn

from pl_bolts.callbacks import LatentDimInterpolator, TensorboardGenerativeModelImageSampler
from pl_bolts.utils.warnings import warn_missing_pkg

try:
    from torchvision import transforms as transform_lib
except ModuleNotFoundError:
    warn_missing_pkg('torchvision')  # pragma: no-cover
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True
from pl_bolts.models.gans.dcgan.components import DCGANDiscriminator, DCGANGenerator


class DCGAN(pl.LightningModule):
    def __init__(
        self,
        beta1: float = 0.5,
        beta2: float = 0.999,
        feature_maps_gen: int = 64,
        feature_maps_disc: int = 64,
        image_channels: int = 1,
        latent_dim: int = 100,
        learning_rate: float = 0.0002,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.generator = self._get_generator()
        self.discriminator = self._get_discriminator()

        self.criterion = nn.BCEWithLogitsLoss()

    def _get_generator(self):
        generator = DCGANGenerator(self.hparams.latent_dim, self.hparams.feature_maps_gen, self.hparams.image_channels)
        generator.apply(self._weights_init)
        return generator

    def _get_discriminator(self):
        discriminator = DCGANDiscriminator(self.hparams.feature_maps_disc, self.hparams.image_channels)
        discriminator.apply(self._weights_init)
        return discriminator

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        betas = (self.hparams.beta1, self.hparams.beta2)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        return [opt_disc, opt_gen], []

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        noise = noise.view(*noise.shape, 1, 1)
        return self.generator(noise)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch

        # Train discriminator
        result = None
        if optimizer_idx == 0:
            result = self._disc_step(real)

        # Train generator
        if optimizer_idx == 1:
            result = self._gen_step(real)

        return result

    def _disc_step(self, real: torch.Tensor) -> torch.Tensor:
        disc_loss = self._get_disc_loss(real)
        self.log("loss/disc", disc_loss, on_epoch=True, prog_bar=True)
        return disc_loss

    def _gen_step(self, real: torch.Tensor) -> torch.Tensor:
        gen_loss = self._get_gen_loss(real)
        self.log("loss/gen", gen_loss, on_epoch=True, prog_bar=True)
        return gen_loss

    def _get_disc_loss(self, real: torch.Tensor) -> torch.Tensor:
        # Train with real
        real_pred = self.discriminator(real)
        real_gt = torch.ones_like(real_pred)
        real_loss = self.criterion(real_pred, real_gt)

        # Train with fake
        fake_pred = self._get_fake_pred(real)
        fake_gt = torch.zeros_like(fake_pred)
        fake_loss = self.criterion(fake_pred, fake_gt)

        disc_loss = real_loss + fake_loss

        return disc_loss

    def _get_gen_loss(self, real: torch.Tensor) -> torch.Tensor:
        # Train with fake
        fake_pred = self._get_fake_pred(real)
        fake_gt = torch.ones_like(fake_pred)
        gen_loss = self.criterion(fake_pred, fake_gt)

        return gen_loss

    def _get_fake_pred(self, real: torch.Tensor) -> torch.Tensor:
        batch_size = self._get_batch_size(real)
        noise = self._get_noise(batch_size, self.hparams.latent_dim)
        fake = self(noise)
        fake_pred = self.discriminator(fake)

        return fake_pred

    @staticmethod
    def _get_batch_size(real: torch.Tensor) -> int:
        batch_size = len(real)
        return batch_size

    def _get_noise(self, n_samples: int, latent_dim: int) -> torch.Tensor:
        noise = torch.randn(n_samples, latent_dim, device=self.device)
        return noise

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--beta1", default=0.5, type=float)
        parser.add_argument("--beta2", default=0.999, type=float)
        parser.add_argument("--feature_maps_gen", default=64, type=int)
        parser.add_argument("--feature_maps_disc", default=64, type=int)
        parser.add_argument("--image_channels", default=1, type=int)
        parser.add_argument("--latent_dim", default=100, type=int)
        parser.add_argument("--learning_rate", default=0.0002, type=float)
        return parser


def cli_main(args=None):
    from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule

    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist, cifar10")
    parser.add_argument("--image_size", default=64, type=int)
    script_args, _ = parser.parse_known_args(args)

    if script_args.dataset == "mnist":
        dm_cls = MNISTDataModule
    elif script_args.dataset == "cifar10":
        dm_cls = CIFAR10DataModule

    parser = dm_cls.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DCGAN.add_model_specific_args(parser)
    args = parser.parse_args(args)

    transforms = transform_lib.Compose([transform_lib.Resize(args.image_size), transform_lib.ToTensor()])
    dm = dm_cls.from_argparse_args(args)
    dm.train_transforms = transforms
    dm.val_transforms = transforms
    dm.test_transforms = transforms

    model = DCGAN(**vars(args))
    callbacks = [TensorboardGenerativeModelImageSampler(), LatentDimInterpolator(interpolate_epoch_interval=5)]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model, dm)
    return dm, model, trainer


if __name__ == "__main__":
    dm, model, trainer = cli_main()
