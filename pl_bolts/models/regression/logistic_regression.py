import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.optim.optimizer import Optimizer

from pl_bolts.datamodules.sklearn_dataloaders import SklearnDataLoaders


class LogisticRegression(pl.LightningModule):

    def __init__(self, input_dim: int, num_classes: int, bias=True, learning_rate=0.05, optimizer:Optimizer = 'Adam',**kwargs):
        """
        Logistic regression model

        Args:
            input_dim: number of dimensions of the input (at least 1)
            num_classes: number of class labels (binary: 2, multi-class: >2)
            bias: specifies if a constant or intercept should be fitted (equivalent to fit_intercept in sklearn)
            learning_rate: learning_rate for the optimizer
            optimizer: the optimizer to use (default='Adam')

        """
        super().__init__()
        self.save_hyperparameters()

        self.linear = nn.Linear(in_features=self.hparams.input_dim, out_features=self.hparams.num_classes, bias=bias).double()

    def forward(self, x):
        y_hat = self.linear(x)
        #y_hat = torch.log_softmax(self.linear(x))  #returns the logits
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        #loss = F.nll_loss(y_hat, y) #logits, labels
        tensorboard_logs = {'train_ce_loss': loss}
        progress_bar_metrics = tensorboard_logs
        return {
            'loss': loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_ce_loss': val_loss}
        progress_bar_metrics = tensorboard_logs
        return {
            'val_loss': val_loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_ce_loss': test_loss}
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
        parser.add_argument('--num_classes', type=int, default=None)
        parser.add_argument('--bias', default='store_true')
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--optimizer', type=str, default='Adam')
        return parser


if __name__ == '__main__':  # pragma: no cover
    from argparse import ArgumentParser
    pl.seed_everything(1234)

    # create dataset
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True) #these are numpy arrays
    loaders = SklearnDataLoaders(X, y)

    # args
    parser = ArgumentParser()
    parser = LogisticRegression.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # model
    model = LogisticRegression(**vars(args))

    # train
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, loaders.train_dataloader(args.batch_size), loaders.val_dataloader(args.batch_size))
