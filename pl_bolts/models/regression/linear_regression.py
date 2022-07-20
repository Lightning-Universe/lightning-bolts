from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Type

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from pl_bolts.utils.stability import under_review


@under_review()
class LinearRegression(LightningModule):
    """
    Linear regression model implementing - with optional L1/L2 regularization
    $$min_{W} ||(Wx + b) - y ||_2^2 $$
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        bias: bool = True,
        learning_rate: float = 1e-4,
        optimizer: Type[Optimizer] = Adam,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            input_dim: number of dimensions of the input (1+)
            output_dim: number of dimensions of the output (default: ``1``)
            bias: If false, will not use $+b$
            learning_rate: learning_rate for the optimizer
            optimizer: the optimizer to use (default: ``Adam``)
            l1_strength: L1 regularization strength (default: ``0.0``)
            l2_strength: L2 regularization strength (default: ``0.0``)
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer

        self.linear = nn.Linear(in_features=self.hparams.input_dim, out_features=self.hparams.output_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        y_hat = self.linear(x)
        return y_hat

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch

        # flatten any input
        x = x.view(x.size(0), -1)

        y_hat = self(x)

        loss = F.mse_loss(y_hat, y, reduction="sum")

        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = self.linear.weight.abs().sum()
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = self.linear.weight.pow(2).sum()
            loss += self.hparams.l2_strength * l2_reg

        loss /= x.size(0)

        tensorboard_logs = {"train_mse_loss": loss}
        progress_bar_metrics = tensorboard_logs
        return {"loss": loss, "log": tensorboard_logs, "progress_bar": progress_bar_metrics}

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        return {"val_loss": F.mse_loss(y_hat, y)}

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_mse_loss": val_loss}
        progress_bar_metrics = tensorboard_logs
        return {"val_loss": val_loss, "log": tensorboard_logs, "progress_bar": progress_bar_metrics}

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        y_hat = self(x)
        return {"test_loss": F.mse_loss(y_hat, y)}

    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        tensorboard_logs = {"test_mse_loss": test_loss}
        progress_bar_metrics = tensorboard_logs
        return {"test_loss": test_loss, "log": tensorboard_logs, "progress_bar": progress_bar_metrics}

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--input_dim", type=int, default=None)
        parser.add_argument("--output_dim", type=int, default=1)
        parser.add_argument("--bias", default="store_true")
        parser.add_argument("--batch_size", type=int, default=16)
        return parser


@under_review()
def cli_main() -> None:
    from pl_bolts.datamodules.sklearn_datamodule import SklearnDataModule
    from pl_bolts.utils import _SKLEARN_AVAILABLE

    seed_everything(1234)

    # create dataset
    if _SKLEARN_AVAILABLE:
        from sklearn.datasets import load_diabetes
    else:  # pragma: no cover
        raise ModuleNotFoundError(
            "You want to use `sklearn` which is not installed yet, install it with `pip install sklearn`."
        )

    # args
    parser = ArgumentParser()
    parser = LinearRegression.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # model
    model = LinearRegression(input_dim=10, l1_strength=1, l2_strength=1)
    # model = LinearRegression(**vars(args))

    # data
    X, y = load_diabetes(return_X_y=True)  # these are numpy arrays
    loaders = SklearnDataModule(X, y, batch_size=args.batch_size)

    # train
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloaders=loaders.train_dataloader(), val_dataloaders=loaders.val_dataloader())


if __name__ == "__main__":
    cli_main()
