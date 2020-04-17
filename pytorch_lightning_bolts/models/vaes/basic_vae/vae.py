import os

from argparse import ArgumentParser
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import distributions
from pytorch_lightning_bolts.models.vaes.basic_vae.models import Encoder, Decoder

import pytorch_lightning as pl


class VAE(pl.LightningModule):

    def __init__(self, hparams, encoder=None, decoder=None, prior='gaussian', approx_posterior='gaussian'):
        super().__init__(self)
        self.hparams = hparams

        if encoder is None:
            self.encoder = Encoder(hidden_dim=hparams.hidden_dim, latent_dim=hparams.latent_dim,
                                   input_width=hparams.input_width, input_height=hparams.input_height)
        else:
            self.encoder = encoder

        if decoder is None:
            self.decoder = Decoder(hidden_dim=hparams.hidden_dim, latent_dim=hparams.latent_dim)
        else:
            self.decoder = decoder

        self.prior = prior
        self.approx_posterior = approx_posterior

    def get_distribution(self, name, **params):
        if name == 'gaussian':
            return distributions.normal.Normal(**params)

    def forward(self, z):
        return self.decoder(z)

    def _run_step(self, batch, batch_idx):
        x, _ = batch
        mu, log_var = self.encoder(x)
        std = torch.exp(log_var/2)

        # Prior ~ Normal(0,1)
        P = self.get_distribution(self.prior, loc=torch.zeros_like(mu), scale=torch.ones_like(std))

        # Approx Posterior ~ Normal(mu, sigma)
        Q = self.get_distribution(self.approx_posterior, loc=mu, scale=std)

        z = Q.rsample()
        pxz = self(z)

        # sum across dimensions because sum of log probabilities of iid univariate gaussians is the same as
        # multivariate gaussian
        x = x.view(x.size(0), -1)
        reconstruction_loss = F.binary_cross_entropy(pxz, x, reduction='none')
        reconstruction_loss = reconstruction_loss.sum(dim=-1)

        log_qz = Q.log_prob(z)
        log_pz = P.log_prob(z)
        kl_divergence = (log_qz - log_pz).sum(dim=1)

        # ELBO = reconstruction + KL
        loss = reconstruction_loss + kl_divergence

        # average loss over batch
        loss = loss.mean()

        return loss, reconstruction_loss, kl_divergence, pxz

    def training_step(self, batch, batch_idx):
        loss, reconstruction_loss, kl_divergence, pxz = self._run_step(batch, batch_idx)

        tensorboard_logs = {
            'train_elbo_loss': loss,
            'train_recon_loss': reconstruction_loss.mean(),
            'train_kl_loss': kl_divergence.mean()
        }

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, reconstruction_loss, kl_divergence, pxz = self._run_step(batch, batch_idx)

        return {'val_loss': loss, 'reconstruction_loss': reconstruction_loss, 'kl_divergence': kl_divergence, 'pxz': pxz}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        recon_loss = torch.stack([x['reconstruction_loss'] for x in outputs]).mean()
        kl_loss = torch.stack([x['kl_divergence'] for x in outputs]).mean()

        tensorboard_logs = {'val_elbo_loss': avg_loss,
                            'val_recon_loss': recon_loss,
                            'val_kl_loss': kl_loss}

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        loss, reconstruction_loss, kl_divergence, pxz = self._run_step(batch, batch_idx)

        return {'test_loss': loss, 'reconstruction_loss': reconstruction_loss, 'kl_divergence': kl_divergence,
                'pxz': pxz}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        recon_loss = torch.stack([x['reconstruction_loss'] for x in outputs]).mean()
        kl_loss = torch.stack([x['kl_divergence'] for x in outputs]).mean()

        tensorboard_logs = {'test_elbo_loss': avg_loss,
                            'test_recon_loss': recon_loss,
                            'test_kl_loss': kl_loss}

        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def prepare_data(self):
        self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    def train_dataloader(self):
        loader = DataLoader(self.mnist_train, batch_size=32)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.mnist_test, batch_size=32)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.mnist_test, batch_size=32)
        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128, help='dimension of itermediate layers before embedding')
        parser.add_argument('--latent_dim', type=int, default=32, help='dimension of latent variables z')
        parser.add_argument('--input_width', type=int, default=28, help='input image width - 28 for MNIST')
        parser.add_argument('--input_height', type=int, default=28, help='input image height - 28 for MNIST')
        return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = VAE.add_model_specific_args(parser)
    args = parser.parse_args()

    vae = VAE(args)
    pl.Trainer(fast_dev_run=True).fit(vae)
