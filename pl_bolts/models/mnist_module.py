from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
    from torchvision.datasets import MNIST
else:  # pragma: no cover
    warn_missing_pkg('torchvision')


class LitMNIST(LightningModule):

    def __init__(self, hidden_dim=128, learning_rate=1e-3, batch_size=32, num_workers=4, data_dir='', **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `torchvision` which is not installed yet.')

        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

        self.mnist_train = None
        self.mnist_val = None

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def prepare_data(self):
        MNIST(self.hparams.data_dir, train=True, download=True, transform=transforms.ToTensor())

    def train_dataloader(self):
        dataset = MNIST(self.hparams.data_dir, train=True, download=False, transform=transforms.ToTensor())
        mnist_train, _ = random_split(dataset, [55000, 5000])
        loader = DataLoader(mnist_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        return loader

    def val_dataloader(self):
        dataset = MNIST(self.hparams.data_dir, train=True, download=False, transform=transforms.ToTensor())
        _, mnist_val = random_split(dataset, [55000, 5000])
        loader = DataLoader(mnist_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        return loader

    def test_dataloader(self):
        test_dataset = MNIST(self.hparams.data_dir, train=False, download=True, transform=transforms.ToTensor())
        loader = DataLoader(test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--data_dir', type=str, default='')
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def cli_main():
    # args
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = LitMNIST.add_model_specific_args(parser)
    args = parser.parse_args()

    # model
    model = LitMNIST(**vars(args))

    # training
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


if __name__ == '__main__':  # pragma: no cover
    cli_main()
