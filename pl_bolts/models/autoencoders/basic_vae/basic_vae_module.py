import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import distributions
from torch.nn import functional as F

from pl_bolts.datamodules import MNISTDataLoaders
from pl_bolts.models.autoencoders.basic_vae.components import Encoder, Decoder


class VAE(LightningModule):

    def __init__(
            self,
            hidden_dim=128,
            latent_dim=32,
            input_width=28,
            input_height=28,
            batch_size=32,
            learning_rate=0.001,
            data_dir='',
            datamodule=None,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        if datamodule is None:
            self.datamodule = MNISTDataLoaders(save_path=data_dir)

        self.encoder = self.init_encoder(self.hparams.hidden_dim, self.hparams.latent_dim,
                                         self.hparams.input_width, self.hparams.input_height)
        self.decoder = self.init_decoder(self.hparams.hidden_dim, self.hparams.latent_dim,
                                         self.hparams.input_width, self.hparams.input_height)

    def init_encoder(self, hidden_dim, latent_dim, input_width, input_height):
        encoder = Encoder(hidden_dim, latent_dim, input_width, input_height, input_channels)
        return encoder

    def init_decoder(self, hidden_dim, latent_dim, input_width, input_height):
        decoder = Decoder(hidden_dim, latent_dim, input_width, input_height)
        return decoder

    def get_prior(self, z_mu, z_std):
        # Prior ~ Normal(0,1)
        P = distributions.normal.Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(z_std))
        return P

    def get_approx_posterior(self, z_mu, z_std):
        # Approx Posterior ~ Normal(mu, sigma)
        Q = distributions.normal.Normal(loc=z_mu, scale=z_std)
        return Q

    def elbo_loss(self, x, P, Q):
        # Reconstruction loss
        z = Q.rsample()
        pxz = self(z)
        recon_loss = F.binary_cross_entropy(pxz, x, reduction='none')

        # sum across dimensions because sum of log probabilities of iid univariate gaussians is the same as
        # multivariate gaussian
        recon_loss = recon_loss.sum(dim=-1)

        # KL divergence loss
        log_qz = Q.log_prob(z)
        log_pz = P.log_prob(z)
        kl_div = (log_qz - log_pz).sum(dim=1)

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
        z_std = torch.exp(z_log_var / 2)

        P = self.get_prior(z_mu, z_std)
        Q = self.get_approx_posterior(z_mu, z_std)

        x = x.view(x.size(0), -1)

        loss, recon_loss, kl_div, pxz = self.elbo_loss(x, P, Q)

        return loss, recon_loss, kl_div, pxz

    def training_step(self, batch, batch_idx):
        loss, recon_loss, kl_div, pxz = self._run_step(batch)

        tensorboard_logs = {
            'train_elbo_loss': loss,
            'train_recon_loss': recon_loss,
            'train_kl_loss': kl_div
        }

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, recon_loss, kl_div, pxz = self._run_step(batch)

        return {
            'val_loss': loss,
            'val_recon_loss': recon_loss,
            'val_kl_div': kl_div,
            'pxz': pxz
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        recon_loss = torch.stack([x['val_recon_loss'] for x in outputs]).mean()
        kl_loss = torch.stack([x['val_kl_div'] for x in outputs]).mean()

        tensorboard_logs = {'val_elbo_loss': avg_loss,
                            'val_recon_loss': recon_loss,
                            'val_kl_loss': kl_loss}

        return {
            'avg_val_loss': avg_loss,
            'log': tensorboard_logs
        }

    def test_step(self, batch, batch_idx):
        loss, recon_loss, kl_div, pxz = self._run_step(batch)

        return {
            'test_loss': loss,
            'test_recon_loss': recon_loss,
            'test_kl_div': kl_div,
            'pxz': pxz
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        recon_loss = torch.stack([x['test_recon_loss'] for x in outputs]).mean()
        kl_loss = torch.stack([x['test_kl_div'] for x in outputs]).mean()

        tensorboard_logs = {'test_elbo_loss': avg_loss,
                            'test_recon_loss': recon_loss,
                            'test_kl_loss': kl_loss}

        return {
            'avg_test_loss': avg_loss,
            'log': tensorboard_logs
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def prepare_data(self):
        self.datamodule.prepare_data()

    def train_dataloader(self):
        return self.datamodule.train_dataloader(self.hparams.batch_size)

    def val_dataloader(self):
        return self.datamodule.val_dataloader(self.hparams.batch_size)

    def test_dataloader(self):
        return self.datamodule.test_dataloader(self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128,
                            help='itermediate layers dimension before embedding for default encoder/decoder')
        parser.add_argument('--latent_dim', type=int, default=32,
                            help='dimension of latent variables z')
        parser.add_argument('--input_width', type=int, default=28,
                            help='input image width - 28 for MNIST (must be even)')
        parser.add_argument('--input_height', type=int, default=28,
                            help='input image height - 28 for MNIST (must be even)')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--data_dir', type=str, default='')
        return parser


if __name__ == '__main__':
    from pl_bolts.datamodules import ImagenetDataModule
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = VAE.add_model_specific_args(parser)
    args = parser.parse_args()

    vae = VAE(**vars(args), datamodule=ImagenetDataModule)
    trainer = Trainer()
    trainer.fit(vae)
