from argparse import ArgumentParser
from typing import Any

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor
from torch.nn import functional as F

from pl_bolts.utils import _TORCHVISION_AVAILABLE


class LitMNIST(LightningModule):
    """PyTorch Lightning implementation of a two-layer MNIST classification module.

    Args:
        hidden_dim (int, optional): dimension of hidden layer (default: ``128``).
        learning_rate (float, optional): optimizer learning rate (default: ``1e-3``).

    Example::

        datamodule = MNISTDataModule()

        model = LitMNIST()

        trainer = Trainer()
        trainer.fit(model, datamodule=datamodule)
    """

    def __init__(self, hidden_dim: int = 128, learning_rate: float = 1e-3, **kwargs: Any) -> None:

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x: Tensor) -> Tensor:
        out = x.view(x.size(0), -1)
        out = torch.relu(self.l1(out))
        out = torch.relu(self.l2(out))
        return out

    def shared_step(self, batch: Any, batch_idx: int, step: str) -> Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        if step == "train":
            self.log("train_loss", loss)
        elif step == "val":
            self.log("val_loss", loss)
        elif step == "test":
            self.log("test_loss", loss)
        else:
            raise ValueError(f"Step {step} is not recognized. Must be 'train', 'val', or 'test'.")

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self.shared_step(batch, batch_idx, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self.shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parser


def cli_main():
    from pl_bolts.datamodules import MNISTDataModule

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = LitMNIST.add_model_specific_args(parser)
    parser = MNISTDataModule.add_dataset_specific_args(parser)

    args = parser.parse_args()

    # Initialize MNISTDatamodule
    datamodule = MNISTDataModule.from_argparse_args(args)

    # Initialize LitMNIST model
    model = LitMNIST(**vars(args))

    # Train LitMNIST model
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":  # pragma: no cover
    cli_main()
