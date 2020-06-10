import os
from argparse import ArgumentParser

import torch
from torch import nn
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split


class LinearRegression(LightningModule):

    def __init__(self, input_dim=None, bias=True, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        if input_dim is not None:
            self.linear = nn.Linear(self.hparams.input_dim, out_features=1, bias=bias)
        else:
            self.linear = None

    def forward(self, x):
        # infer feature dimension at run time
        if self.hparams.input_dim is None:
            self.hparams.input_dim = x.size()[-1]
            self.linear = nn.Linear(self.hparams.input_dim, out_features=1, bias=self.hparams.bias)
            self.linear.to(self.device)

        x = self.linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        tensorboard_logs = {'train_mse_loss': loss}
        progress_bar_metrics = tensorboard_logs
        return {
            'loss': loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'val_loss': F.mse_loss(y_hat, y)}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_mse_loss': val_loss}
        progress_bar_metrics = tensorboard_logs
        return {
            'val_loss': val_loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'test_loss': F.mse_loss(y_hat, y)}

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_mse_loss': test_loss}
        progress_bar_metrics = tensorboard_logs
        return {
            'test_loss': test_loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--input_dim', type=int, default=None)
        parser.add_argument('--bias', default='store_true')
        return parser


if __name__ == '__main__':  # pragma: no cover
    # create dataset
    from sklearn.datasets import make_regression
    X, y = make_regression()
    dataset = SKLearnDataset(X, y)

    # args
    parser = ArgumentParser()
    parser = LinearRegression.add_model_specific_args(parser)
    args = parser.parse_args()

    # model
    model = LinearRegression(**vars(args))

    batch =(torch.randn(5, 7), torch.rand(5))
    model.training_step(batch, 3)
    trainer = Trainer()
    trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())