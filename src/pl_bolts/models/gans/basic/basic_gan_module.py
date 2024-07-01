from argparse import ArgumentParser

import torch
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from torch.nn import functional as F  # noqa: N812

from pl_bolts.models.gans.basic.components import Discriminator, Generator


class GAN(LightningModule):
    """Vanilla GAN implementation.

    Example::

        from pl_bolts.models.gans import GAN

        m = GAN()
        Trainer(gpus=2).fit(m)

    Example CLI::

        # mnist
        python basic_gan_module.py --gpus 1

        # imagenet
        python  basic_gan_module.py --gpus 1 --dataset 'imagenet2012'
        --data_dir /path/to/imagenet/folder/ --meta_dir ~/path/to/meta/bin/folder
        --batch_size 256 --learning_rate 0.0001

    """

    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        latent_dim: int = 32,
        learning_rate: float = 0.0002,
        **kwargs
    ) -> None:
        """
        Args:
            input_channels: number of channels of an image
            input_height: image height
            input_width: image width
            latent_dim: emb dim for encoder
            learning_rate: the learning rate
        """
        super().__init__()

        # makes self.hparams under the hood and saves to ckpt
        self.save_hyperparameters()
        self.img_dim = (input_channels, input_height, input_width)

        # networks
        self.generator = self.init_generator(self.img_dim)
        self.discriminator = self.init_discriminator(self.img_dim)

    def init_generator(self, img_dim):
        return Generator(latent_dim=self.hparams.latent_dim, img_shape=img_dim)

    def init_discriminator(self, img_dim):
        return Discriminator(img_shape=img_dim)

    def forward(self, z):
        """Generates an image given input noise z.

        Example::

            z = torch.rand(batch_size, latent_dim)
            gan = GAN.load_from_checkpoint(PATH)
            img = gan(z)

        """
        return self.generator(z)

    def generator_loss(self, x):
        # sample noise
        z = torch.randn(x.shape[0], self.hparams.latent_dim, device=self.device)
        y = torch.ones(x.size(0), 1, device=self.device)

        # generate images
        generated_imgs = self(z)

        # ground truth result (ie: all real)
        return F.binary_cross_entropy(self.discriminator(generated_imgs), y)

    def discriminator_loss(self, x):
        # train discriminator on real
        b = x.size(0)
        x_real = x
        y_real = torch.ones(b, 1, device=self.device)

        # calculate real score
        real_loss = F.binary_cross_entropy(self.discriminator(x_real), y_real)

        # train discriminator on fake
        z = torch.randn(b, self.hparams.latent_dim, device=self.device)
        x_fake = self(z)
        y_fake = torch.zeros(b, 1, device=self.device)

        # calculate fake score
        fake_loss = F.binary_cross_entropy(self.discriminator(x_fake), y_fake)

        # gradient backprop & optimize ONLY D's parameters
        return real_loss + fake_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, _ = batch
        # train generator
        if optimizer_idx == 0:
            return self.generator_step(x)
        # train discriminator
        if optimizer_idx == 1:
            return self.discriminator_step(x)
        return None

    def generator_step(self, x):
        g_loss = self.generator_loss(x)
        # log to prog bar on each step AND for the full epoch use the generator loss for checkpointing
        self.log("g_loss", g_loss, on_epoch=True, prog_bar=True)
        return g_loss

    def discriminator_step(self, x):
        # Measure discriminator's ability to classify real from generated samples
        d_loss = self.discriminator_loss(x)

        # log to prog bar on each step AND for the full epoch
        self.log("d_loss", d_loss, on_epoch=True, prog_bar=True)
        return d_loss

    def configure_optimizers(self):
        lr = self.hparams.learning_rate

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.0002, help="adam: learning rate")
        parser.add_argument(
            "--adam_b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient"
        )
        parser.add_argument(
            "--adam_b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient"
        )
        parser.add_argument("--latent_dim", type=int, default=100, help="generator embedding dim")
        return parser


def cli_main(args=None):
    from pl_bolts.callbacks import (
        LatentDimInterpolator,
        TensorboardGenerativeModelImageSampler,
    )
    from pl_bolts.datamodules import (
        CIFAR10DataModule,
        ImagenetDataModule,
        MNISTDataModule,
        STL10DataModule,
    )

    seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--dataset", default="mnist", type=str, help="mnist, cifar10, stl10, imagenet")
    script_args, _ = parser.parse_known_args(args)

    if script_args.dataset == "mnist":
        dm_cls = MNISTDataModule
    elif script_args.dataset == "cifar10":
        dm_cls = CIFAR10DataModule
    elif script_args.dataset == "stl10":
        dm_cls = STL10DataModule
    elif script_args.dataset == "imagenet":
        dm_cls = ImagenetDataModule

    parser = dm_cls.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser = GAN.add_model_specific_args(parser)
    args = parser.parse_args(args)

    dm = dm_cls.from_argparse_args(args)
    model = GAN(*dm.dims, **vars(args))
    callbacks = [
        TensorboardGenerativeModelImageSampler(),
        LatentDimInterpolator(interpolate_epoch_interval=5),
        TQDMProgressBar(refresh_rate=20),
    ]
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model, datamodule=dm)
    return dm, model, trainer


if __name__ == "__main__":
    dm, model, trainer = cli_main()
