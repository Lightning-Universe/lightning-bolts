import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer

from pl_bolts.datamodules.sklearn_dataloaders import SklearnDataLoaders


class LinearRegression(pl.LightningModule):

    def __init__(self, input_dim: int, bias=True, learning_rate=0.05, optimizer:Optimizer = 'Adam',**kwargs):
        """
        Linear regression model implementing
        $$min_{W} ||(Wx + b) - y ||_2^2 $$

        Args:
            input_dim: number of dimensions of the input (1+)
            bias: If false, will not use $$+b$$
            learning_rate: learning_rate for the optimizer
            optimizer: the optimizer to use (default='adam')

        """
        super().__init__()
        self.save_hyperparameters()

        self.linear = nn.Linear(in_features=self.hparams.input_dim, out_features=1, bias=bias).double()

    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat

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
        optimizer_class = getattr(torch.optim, self.hparams.optimizer)
        return optimizer_class(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--input_dim', type=int, default=None)
        parser.add_argument('--bias', default='store_true')
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--optimizer', type=str, default='Adam')
        return parser


if __name__ == '__main__':  # pragma: no cover
    from argparse import ArgumentParser
    pl.seed_everything(1234)

    # create dataset
    from sklearn.datasets import load_boston
    X, y = load_boston(return_X_y=True) #these are numpy arrays
    loaders = SklearnDataLoaders(X, y)

    # args
    parser = ArgumentParser()
    parser = LinearRegression.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # model
    model = LinearRegression(**vars(args))

    # train
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, loaders.train_dataloader(args.batch_size), loaders.val_dataloader(args.batch_size))