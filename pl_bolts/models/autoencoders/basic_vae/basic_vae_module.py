import os
from argparse import ArgumentParser

import torch
import torchvision
from pytorch_lightning import LightningModule, Trainer
from torch import distributions
from torch.nn import functional as F

import pl_bolts
from pl_bolts.datamodules import MNISTDataModule
from pl_bolts.models.autoencoders.basic_vae.components import Encoder, Decoder
from pl_bolts.utils.pretrained_weights import load_pretrained


class VAE(LightningModule):

    def __init__(
            self,
            hidden_dim: int = 128,
            latent_dim: int = 32,
            input_channels: int = 3,
            input_width: int = 224,
            input_height: int = 224,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            data_dir: str = '.',
            datamodule: pl_bolts.datamodules.LightningDataModule = None,
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

        self.datamodule = datamodule
        self.__set_pretrained_dims(pretrained)

        # use mnist as the default module
        self.__set_default_datamodule(data_dir)

        # init actual model
        self.__init_system()

        if pretrained:
            self.load_pretrained(pretrained)

    def __init_system(self):
        self.encoder = self.init_encoder()
        self.decoder = self.init_decoder()

    def __set_pretrained_dims(self, pretrained):
        if pretrained == 'imagenet2012':
            self.datamodule = ImagenetDataModule(data_dir=self.hparams.data_dir)
            (self.hparams.input_channels, self.hparams.input_height, self.hparams.input_width) = self.datamodule.size()

    def __set_default_datamodule(self, data_dir):
        if self.datamodule is None:
            self.datamodule = MNISTDataModule(data_dir=data_dir)
            (self.hparams.input_channels, self.hparams.input_height, self.hparams.input_width) = self.datamodule.size()

    def load_pretrained(self, pretrained):
        available_weights = {'imagenet2012'}

        if pretrained in available_weights:
            weights_name = f'vae-{pretrained}'
            load_pretrained(self, weights_name)

    def init_encoder(self):
        encoder = Encoder(
            self.hparams.hidden_dim,
            self.hparams.latent_dim,
            self.hparams.input_channels,
            self.hparams.input_width,
            self.hparams.input_height
        )
        return encoder

    def init_decoder(self):
        decoder = Decoder(
            self.hparams.hidden_dim,
            self.hparams.latent_dim,
            self.hparams.input_width,
            self.hparams.input_height,
            self.hparams.input_channels
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

    def elbo_loss(self, x, P, Q):
        # Reconstruction loss
        z = Q.rsample()
        pxz = self(z)
        pxz = torch.tanh(pxz)
        recon_loss = F.mse_loss(pxz, x, reduction='none')

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
            'val_loss': avg_loss,
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
            'test_loss': avg_loss,
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
        parser.add_argument('--input_width', type=int, default=224,
                            help='input width (used Imagenet downsampled size)')
        parser.add_argument('--input_height', type=int, default=224,
                            help='input width (used Imagenet downsampled size)')
        parser.add_argument('--input_channels', type=int, default=3,
                            help='number of input channels')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--pretrained', type=str, default=None)
        parser.add_argument('--data_dir', type=str, default=os.getcwd())

        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser


if __name__ == '__main__':
    from pl_bolts.datamodules import ImagenetDataModule
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='mnist', type=str)

    parser = Trainer.add_argparse_args(parser)
    parser = VAE.add_model_specific_args(parser)
    parser = ImagenetDataModule.add_argparse_args(parser)
    parser = MNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    #
    # if args.dataset == 'imagenet' or args.pretrained:
    #     datamodule = ImagenetDataModule.from_argparse_args(args)
    #     args.image_width = datamodule.size()[1]
    #     args.image_height = datamodule.size()[2]
    #     args.input_channels = datamodule.size()[0]
    #
    # elif args.dataset == 'mnist':
    #     datamodule = MNISTDataModule.from_argparse_args(args)
    #     args.image_width = datamodule.size()[1]
    #     args.image_height = datamodule.size()[2]
    #     args.input_channels = datamodule.size()[0]

    vae = VAE(**vars(args))
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(vae)
