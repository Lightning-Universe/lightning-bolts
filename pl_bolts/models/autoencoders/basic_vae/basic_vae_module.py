import os
from argparse import ArgumentParser

import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from pl_bolts.datamodules import (BinaryMNISTDataModule, CIFAR10DataModule,
                                  ImagenetDataModule, MNISTDataModule,
                                  STL10DataModule)
from pl_bolts.models.autoencoders.basic_vae.components import resnet18_encoder, resnet18_decoder
from pl_bolts.models.autoencoders.basic_vae.components import resnet50_encoder, resnet50_decoder
from pl_bolts.utils.pretrained_weights import load_pretrained

pretrained_urls = {
    'cifar10': 'abc'
}

"""
# TODO: pretrained url
# TODO: correct enc, dec for maxpool and conv
# correct params for class
# run cifar10
# run imagenet
"""

class VAE(pl.LightningModule):
    def __init__(
        self,
        enc_type='resnet18',
        conv1=False,
        maxpool1=False,
        enc_out_dim=512,
        kl_coeff=0.1,
        latent_dim=256,
        lr=1e-4
    ):
        """
        Standard VAE with Gaussian Prior and approx posterior.

        Model is available pretrained on different datasets:

        Example::

            # not pretrained
            vae = VAE()

            # pretrained on imagenet
            vae = VAE.from_pretrained('resnet50-imagenet')

            # pretrained on cifar10
            vae = VAE.from_pretrained('resnet18-cifar10')

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

        super(VAE, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim

        valid_encoders = {
            'resnet18': {'enc': resnet18_encoder, 'dec': resnet18_decoder},
            'resnet50': {'enc': resnet50_encoder, 'dec': resnet50_decoder},
        }

        if enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(conv1, maxpool1)
            self.decoder = resnet18_decoder(latent_dim=self.latent_dim, conv1, maxpool1)
        else:
            self.encoder = valid_encoders[enc_type]['enc'](conv1, maxpool1)
            self.decoder = valid_encoders[enc_type]['dec'](latent_dim=self.latent_dim, conv1, maxpool1)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

    def from_pretrained(checkpoint_name):
        pass

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch

        z, x_hat, p, q = self.forward(x)

        reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.sum(dim=-1)

        loss = kl + reconstruction_loss

        logs = {
            "recon_loss": reconstruction_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        result = pl.TrainResult(minimize=loss)
        result.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return result

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({f"val_{k}": v for k, v in logs.items()})
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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
    parser.add_argument("--dataset", default="cifar10", type=str, help="cifar10, stl10, imagenet")
    script_args, _ = parser.parse_known_args(args)

    if script_args.dataset == "cifar10":
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
