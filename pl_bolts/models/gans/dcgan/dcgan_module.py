from argparse import ArgumentParser
from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from pl_bolts.callbacks import LatentDimInterpolator, TensorboardGenerativeModelImageSampler
from pl_bolts.models.gans.dcgan.components import DCGANDiscriminator, DCGANGenerator
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import LSUN, MNIST
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class DCGAN(pl.LightningModule):
    """
    DCGAN implementation.

    Example::

        from pl_bolts.models.gans import DCGAN

        m = DCGAN()
        Trainer(gpus=2).fit(m)

    Example CLI::

        # mnist
        python dcgan_module.py --gpus 1

        # cifar10
        python dcgan_module.py --gpus 1 --dataset cifar10 --image_channels 3
    """

    def __init__(
        self,
        beta1: float = 0.5,
        feature_maps_gen: int = 64,
        feature_maps_disc: int = 64,
        image_channels: int = 1,
        latent_dim: int = 100,
        learning_rate: float = 0.0002,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            beta1: Beta1 value for Adam optimizer
            feature_maps_gen: Number of feature maps to use for the generator
            feature_maps_disc: Number of feature maps to use for the discriminator
            image_channels: Number of channels of the images from the dataset
            latent_dim: Dimension of the latent space
            learning_rate: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()

        self.generator = self._get_generator()
        self.discriminator = self._get_discriminator()

        self.criterion = nn.BCELoss()

    def _get_generator(self) -> nn.Module:
        generator = DCGANGenerator(self.hparams.latent_dim, self.hparams.feature_maps_gen, self.hparams.image_channels)
        generator.apply(self._weights_init)
        return generator

    def _get_discriminator(self) -> nn.Module:
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
        betas = (self.hparams.beta1, 0.999)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        return [opt_disc, opt_gen], []

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generates an image given input noise

        Example::

            noise = torch.rand(batch_size, latent_dim)
            gan = GAN.load_from_checkpoint(PATH)
            img = gan(noise)
        """
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
        self.log("loss/disc", disc_loss, on_epoch=True)
        return disc_loss

    def _gen_step(self, real: torch.Tensor) -> torch.Tensor:
        gen_loss = self._get_gen_loss(real)
        self.log("loss/gen", gen_loss, on_epoch=True)
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
        batch_size = len(real)
        noise = self._get_noise(batch_size, self.hparams.latent_dim)
        fake = self(noise)
        fake_pred = self.discriminator(fake)

        return fake_pred

    def _get_noise(self, n_samples: int, latent_dim: int) -> torch.Tensor:
        return torch.randn(n_samples, latent_dim, device=self.device)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--beta1", default=0.5, type=float)
        parser.add_argument("--feature_maps_gen", default=64, type=int)
        parser.add_argument("--feature_maps_disc", default=64, type=int)
        parser.add_argument("--latent_dim", default=100, type=int)
        parser.add_argument("--learning_rate", default=0.0002, type=float)
        return parser


def cli_main(args=None):
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--dataset", default="mnist", type=str, choices=["lsun", "mnist"])
    parser.add_argument("--data_dir", default="./", type=str)
    parser.add_argument("--image_size", default=64, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    script_args, _ = parser.parse_known_args(args)

    if script_args.dataset == "lsun":
        transforms = transform_lib.Compose([
            transform_lib.Resize(script_args.image_size),
            transform_lib.CenterCrop(script_args.image_size),
            transform_lib.ToTensor(),
            transform_lib.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = LSUN(root=script_args.data_dir, classes=["bedroom_train"], transform=transforms)
        image_channels = 3
    elif script_args.dataset == "mnist":
        transforms = transform_lib.Compose([
            transform_lib.Resize(script_args.image_size),
            transform_lib.ToTensor(),
            transform_lib.Normalize((0.5, ), (0.5, )),
        ])
        dataset = MNIST(root=script_args.data_dir, download=True, transform=transforms)
        image_channels = 1

    dataloader = DataLoader(
        dataset, batch_size=script_args.batch_size, shuffle=True, num_workers=script_args.num_workers
    )

    parser = DCGAN.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    model = DCGAN(**vars(args), image_channels=image_channels)
    callbacks = [
        TensorboardGenerativeModelImageSampler(num_samples=5),
        LatentDimInterpolator(interpolate_epoch_interval=5),
    ]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    cli_main()
