"""An implemen§ation of Logistic Regression in PyTorch-Lightning."""

from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple, Type

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import functional

from pl_bolts.utils.stability import under_review


class LogisticRegression(LightningModule):
    """Logistic Regression Model."""

    criterion: nn.CrossEntropyLoss
    linear: nn.Linear

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        bias: bool = True,
        learning_rate: float = 1e-4,
        optimizer: Type[Optimizer] = Adam,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Logistic Regression.

        Args:
            input_dim: Number of dimensions of the input (at least `1`).
            num_classes: Number of class labels (binary: `2`, multi-class: > `2`).
            bias: Specifies if a constant or intercept should be fitted (equivalent to `fit_intercept` in `sklearn`).
            learning_rate: Learning rate for the optimizer.
            optimizer: Model optimizer to use.
            l1_strength: L1 regularization strength.
            l2_strength: L2 regularization strength.

        Attributes:
            linear: Linear layer.
            criterion: Cross-Entropy loss function.
            optimizer: Model optimizer to use.

        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.linear = nn.Linear(
            in_features=self.hparams.input_dim, out_features=self.hparams.num_classes, bias=self.hparams.bias
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.

        """
        return self.linear(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """Training step for the model.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.

        Returns:
            Loss tensor.

        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """Validation step for the model.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.

        Returns:
            Loss tensor.

        """
        return self._shared_step(batch, "val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """Test step for the model.

        Args:
            batch: Batch of data.
            batch_idx: Batch index.

        Returns:
            Loss tensor.

        """
        return self._shared_step(batch, "test")

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """Validation epoch end for the model.

        Args:
            outputs: List of outputs from the validation step.

        Returns:
            Loss tensor.

        """
        return self._shared_epoch_end(outputs, "val")

    def test_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """Test epoch end for the model.

        Args:
            outputs: List of outputs from the test step.

        Returns:
            Loss tensor.

        """
        return self._shared_epoch_end(outputs, "test")

    def configure_optimizers(self) -> Optimizer:
        """Configure the optimizer for the model.

        Returns:
            Optimizer.

        """
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

    def _prepare_batch(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        x, y = batch
        x = x.view(x.size(0), -1)
        return self.linear(x), torch.tensor(y, dtype=torch.long)

    def _shared_step(self, batch: Tuple[Tensor, Tensor], stage: str) -> Dict[str, Tensor]:
        x, y = self._prepare_batch(batch)
        loss = self.criterion(x, y)

        if stage == "train":
            loss = self._regularization(loss)
            loss /= x.size(0)
            metrics = {"loss": loss}
            self.log_dict(metrics, on_step=True)
            return metrics

        acc = self._calculate_accuracy(x, y)
        return self._log_metrics(acc, loss, stage, on_step=True)

    def _shared_epoch_end(self, outputs: List[Dict[str, Tensor]], stage: str) -> Dict[str, Tensor]:
        acc = torch.stack([x[f"{stage}_acc"] for x in outputs]).mean()
        loss = torch.stack([x[f"{stage}_loss"] for x in outputs]).mean()
        return self._log_metrics(acc, loss, stage, on_epoch=True)

    def _log_metrics(self, acc: Tensor, loss: Tensor, stage: str, **kwargs: bool) -> Dict[str, Tensor]:
        metrics = {f"{stage}_loss": loss, f"{stage}_acc": acc}
        self.log_dict(metrics, **kwargs)
        return metrics

    def _calculate_accuracy(self, x: Tensor, y: Tensor) -> Tensor:
        _, y_hat = torch.max(x, dim=-1)
        return functional.accuracy(
            y_hat, y, task="multiclass", average="weighted", num_classes=self.hparams.num_classes, top_k=1
        )

    def _regularization(self, loss: Tensor) -> Tensor:
        if self.hparams.l1_strength > 0:
            l1_reg = self.linear.weight.abs().sum()
            loss += self.hparams.l1_strength * l1_reg

        if self.hparams.l2_strength > 0:
            l2_reg = self.linear.weight.pow(2).sum()
            loss += self.hparams.l2_strength * l2_reg
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds model specific arguments to the parser.

        Args:
            parent_parser: Parent parser to which the arguments will be added.

        Returns:
            ArgumentParser with the added arguments.

        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--input_dim", type=int, default=None)
        parser.add_argument("--num_classes", type=int, default=None)
        parser.add_argument("--bias", default="store_true")
        parser.add_argument("--batch_size", type=int, default=16)
        return parser


@under_review()
def cli_main() -> None:
    from pl_bolts.datamodules.sklearn_datamodule import SklearnDataModule
    from pl_bolts.utils import _SKLEARN_AVAILABLE

    seed_everything(1234)

    # Example: Iris dataset in Sklearn (4 features, 3 class labels)
    if _SKLEARN_AVAILABLE:
        from sklearn.datasets import load_iris
    else:  # pragma: no cover
        raise ModuleNotFoundError(
            "You want to use `sklearn` which is not installed yet, install it with `pip install sklearn`."
        )

    # args
    parser = ArgumentParser()
    parser = LogisticRegression.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # model
    # model = LogisticRegression(**vars(args))
    model = LogisticRegression(input_dim=4, num_classes=3, l1_strength=0.01, learning_rate=0.01)

    # data
    x, y = load_iris(return_X_y=True)
    loaders = SklearnDataModule(x, y, batch_size=args.batch_size, num_workers=0)

    # train
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloaders=loaders.train_dataloader(), val_dataloaders=loaders.val_dataloader())


if __name__ == "__main__":
    cli_main()
