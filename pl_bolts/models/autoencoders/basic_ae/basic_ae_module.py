import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F

from pl_bolts.datamodules import MNISTDataLoaders
from pl_bolts.models.autoencoders.basic_ae.components import AEEncoder
from pl_bolts.models.autoencoders.basic_vae.components import Decoder


class BasicAE(LightningModule):

    def __init__(
            self,
            hparams=None,
    ):
        super().__init__()
        # attach hparams to log hparams to the loggers (like tensorboard)
        self.__check_hparams(hparams)
        self.hparams = hparams

        self.dataloaders = MNISTDataLoaders(save_path=os.getcwd())

        self.encoder = self.init_encoder(self.hidden_dim, self.latent_dim,
                                         self.input_width, self.input_height)
        self.decoder = self.init_decoder(self.hidden_dim, self.latent_dim,
                                         self.input_width, self.input_height)

    def __check_hparams(self, hparams):
        self.hidden_dim = hparams.hidden_dim if hasattr(hparams, 'hidden_dim') else 128
        self.latent_dim = hparams.latent_dim if hasattr(hparams, 'latent_dim') else 32
        self.input_width = hparams.input_width if hasattr(hparams, 'input_width') else 28
        self.input_height = hparams.input_height if hasattr(hparams, 'input_height') else 28
        self.batch_size = hparams.batch_size if hasattr(hparams, 'batch_size') else 32

    def init_encoder(self, hidden_dim, latent_dim, input_width, input_height):
        encoder = AEEncoder(hidden_dim, latent_dim, input_width, input_height)
        return encoder

    def init_decoder(self, hidden_dim, latent_dim, input_width, input_height):
        decoder = Decoder(hidden_dim, latent_dim, input_width, input_height)
        return decoder

    def forward(self, z):
        return self.decoder(z)

    def _run_step(self, batch):
        x, _ = batch
        z = self.encoder(x)
        x_hat = self(z)

        loss = F.mse_loss(x.view(x.size(0), -1), x_hat)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._run_step(batch)

        tensorboard_logs = {
            'mse_loss': loss,
        }

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._run_step(batch)

        return {
            'val_loss': loss,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'mse_loss': avg_loss}

        return {
            'avg_val_loss': avg_loss,
            'log': tensorboard_logs
        }

    def test_step(self, batch, batch_idx):
        loss = self._run_step(batch)

        return {
            'test_loss': loss,
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        tensorboard_logs = {'mse_loss': avg_loss}

        return {
            'avg_test_loss': avg_loss,
            'log': tensorboard_logs
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def prepare_data(self):
        self.dataloaders.prepare_data()

    def train_dataloader(self):
        return self.dataloaders.train_dataloader(self.batch_size)

    def val_dataloader(self):
        return self.dataloaders.val_dataloader(self.batch_size)

    def test_dataloader(self):
        return self.dataloaders.test_dataloader(self.batch_size)

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
        return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = BasicAE.add_model_specific_args(parser)
    args = parser.parse_args()

    ae = BasicAE(args)
    trainer = Trainer()
    trainer.fit(ae)
