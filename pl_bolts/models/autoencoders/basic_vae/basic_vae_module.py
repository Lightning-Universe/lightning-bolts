import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch import distributions
from torch.nn import functional as F

from pl_bolts.datamodules import (BinaryMNISTDataModule, CIFAR10DataModule,
                                  ImagenetDataModule, MNISTDataModule,
                                  STL10DataModule)
from pl_bolts.models.autoencoders.basic_vae.components import Decoder, Encoder
from pl_bolts.utils import shaping
from pl_bolts.utils.pretrained_weights import load_pretrained


class VAE(pl.LightningModule):
    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        learning_rate: float = 0.001,
        pretrained: str = None,
        **kwargs
    ):
        """
        Standard VAE with Gaussian Prior and approx posterior.

        Model is available pretrained on different datasets:

        Example::

            # not pretrained
            vae = VAE()

            # pretrained on imagenet
            vae = VAE(pretrained='imagenet')

            # pretrained on cifar10
            vae = VAE(pretrained='cifar10')

        Args:

            hidden_dim: encoder and decoder hidden dims
            latent_dim: latenet code dim
            input_channels: num of channels of the input image.
            input_width: image input width
            input_height: image input height
            batch_size: the batch size
            learning_rate" the learning rate
            data_dir: the directory to store data
            datamodule: The Lightning DataModule
            pretrained: Load weights pretrained on a dataset
        """
        super().__init__()
        self.save_hyperparameters()
        self.img_dim = (input_channels, input_height, input_width)

        # init actual model
        self.__init_system()

        if pretrained:
            self.load_pretrained(pretrained)

    def __init_system(self):
        self.encoder = self.init_encoder()
        self.decoder = self.init_decoder()

    def load_pretrained(self, pretrained):
        available_weights = {"imagenet2012"}

        if pretrained in available_weights:
            weights_name = f"vae-{pretrained}"
            load_pretrained(self, weights_name)

    def init_encoder(self):
        encoder = Encoder(
            self.hparams.hidden_dim,
            self.hparams.latent_dim,
            self.hparams.input_channels,
            self.hparams.input_width,
            self.hparams.input_height,
        )
        return encoder

    def init_decoder(self):
        decoder = Decoder(
            self.hparams.hidden_dim,
            self.hparams.latent_dim,
            self.hparams.input_width,
            self.hparams.input_height,
            self.hparams.input_channels,
        )
        return decoder

    def get_prior(self, z_mu, z_std):
        # Prior ~ Normal(0,1)
        P = distributions.normal.Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(z_std))
        return P

    def get_approx_posterior(self, z_mu, z_std):
        # Approx Posterior ~ Normal(mu, sigma)
        Q = distributions.normal.Normal(loc=z_mu, scale=z_std)
        return Q

    def elbo_loss(self, x, P, Q, num_samples):
        z = Q.rsample()

        # ----------------------
        # KL divergence loss (using monte carlo sampling)
        # ----------------------
        log_qz = Q.log_prob(z)
        log_pz = P.log_prob(z)

        # (batch, num_samples, z_dim) -> (batch, num_samples)
        kl_div = (log_qz - log_pz).sum(dim=2)

        # we used monte carlo sampling to estimate. average across samples
        # kl_div = kl_div.mean(-1)

        # ----------------------
        # Reconstruction loss
        # ----------------------
        z = z.view(-1, z.size(-1)).contiguous()
        pxz = self.decoder(z)

        pxz = pxz.view(-1, num_samples, pxz.size(-1))
        x = shaping.tile(x.unsqueeze(1), 1, num_samples)

        pxz = torch.sigmoid(pxz)
        recon_loss = F.binary_cross_entropy(pxz, x, reduction="none")

        # sum across dimensions because sum of log probabilities of iid univariate gaussians is the same as
        # multivariate gaussian
        recon_loss = recon_loss.sum(dim=-1)

        # we used monte carlo sampling to estimate. average across samples
        # recon_loss = recon_loss.mean(-1)

        # ELBO = reconstruction + KL
        loss = recon_loss + kl_div

        # average over batch
        loss = loss.mean()
        recon_loss = recon_loss.mean()
        kl_div = kl_div.mean()

        return loss, recon_loss, kl_div, pxz

    def forward(self, z):
        return self.decoder(z)

    def _run_step(self, batch):
        x, _ = batch
        z_mu, z_log_var = self.encoder(x)

        # we're estimating the KL divergence using sampling
        num_samples = 32

        # expand dims to sample all at once
        # (batch, z_dim) -> (batch, num_samples, z_dim)
        z_mu = z_mu.unsqueeze(1)
        z_mu = shaping.tile(z_mu, 1, num_samples)

        # (batch, z_dim) -> (batch, num_samples, z_dim)
        z_log_var = z_log_var.unsqueeze(1)
        z_log_var = shaping.tile(z_log_var, 1, num_samples)

        # convert to std
        z_std = torch.exp(z_log_var / 2)

        P = self.get_prior(z_mu, z_std)
        Q = self.get_approx_posterior(z_mu, z_std)

        x = x.view(x.size(0), -1)

        loss, recon_loss, kl_div, pxz = self.elbo_loss(x, P, Q, num_samples)

        return loss, recon_loss, kl_div, pxz

    def training_step(self, batch, batch_idx):
        loss, recon_loss, kl_div, pxz = self._run_step(batch)
        result = pl.TrainResult(loss)
        result.log_dict({"train_elbo_loss": loss, "train_recon_loss": recon_loss, "train_kl_loss": kl_div})
        return result

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, kl_div, pxz = self._run_step(batch)
        result = pl.EvalResult(loss, checkpoint_on=loss)
        result.log_dict(
            {"val_loss": loss, "val_recon_loss": recon_loss, "val_kl_div": kl_div}
        )
        return result

    def test_step(self, batch, batch_idx):
        loss, recon_loss, kl_div, pxz = self._run_step(batch)
        result = pl.EvalResult(loss)
        result.log_dict(
            {"test_loss": loss, "test_recon_loss": recon_loss, "test_kl_div": kl_div}
        )
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--hidden_dim",
            type=int,
            default=128,
            help="itermediate layers dimension before embedding for default encoder/decoder",
        )
        parser.add_argument("--latent_dim", type=int, default=4, help="dimension of latent variables z")
        parser.add_argument("--pretrained", type=str, default=None)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parser


def cli_main(args=None):
    from pl_bolts.callbacks import LatentDimInterpolator, TensorboardGenerativeModelImageSampler

    # cli_main()
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
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VAE.add_model_specific_args(parser)
    args = parser.parse_args(args)

    dm = dm_cls.from_argparse_args(args)
    model = VAE(*dm.size(), **vars(args))
    callbacks = [TensorboardGenerativeModelImageSampler(), LatentDimInterpolator(interpolate_epoch_interval=5)]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, progress_bar_refresh_rate=20)
    trainer.fit(model, dm)
    return dm, model, trainer


if __name__ == "__main__":
    dm, model, trainer = cli_main()
