import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn

from pl_bolts.models.vision.image_gpt.gpt2 import GPT2


def _shape_input(x):
    """shape batch of images for input into GPT2 model."""
    x = x.view(x.shape[0], -1)  # flatten images into sequences
    x = x.transpose(0, 1).contiguous()  # to shape [seq len, batch]
    return x


class ImageGPT(LightningModule):
    """
    **Paper**: `Generative Pretraining from Pixels
    <https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf>`_
    [original paper `code <https://github.com/openai/image-gpt>`_].

    **Paper by:** Mark Che, Alec Radford, Rewon Child, Jeff Wu, Heewoo Jun,
    Prafulla Dhariwal, David Luan, Ilya Sutskever

    **Implementation contributed by**:

        - `Teddy Koker <https://github.com/teddykoker>`_

    **Original repo with results and more implementation details**:

        - `https://github.com/teddykoker/image-gpt <https://github.com/teddykoker/image-gpt>`_

    **Example Results (Photo credits: Teddy Koker)**:

    .. image:: https://raw.githubusercontent.com/teddykoker/image-gpt/master/figures/mnist.png
        :width: 250
        :alt: credit-Teddy-Koker

    .. image:: https://raw.githubusercontent.com/teddykoker/image-gpt/master/figures/fmnist.png
        :width: 250
        :alt: credit-Teddy-Koker

    **Default arguments:**

    .. list-table:: Argument Defaults
        :widths: 50 25 25
        :header-rows: 1

        * - Argument
          - Default
          - iGPT-S (`Chen et al. <https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf>`_)
        * - `--embed_dim`
          - 16
          - 512
        * - `--heads`
          - 2
          - 8
        * - `--layers`
          - 8
          - 24
        * - `--pixels`
          - 28
          - 32
        * - `--vocab_size`
          - 16
          - 512
        * - `--num_classes`
          - 10
          - 10
        * - `--batch_size`
          - 64
          - 128
        * - `--learning_rate`
          - 0.01
          - 0.01
        * - `--steps`
          - 25000
          - 1000000

    Example::

        import pytorch_lightning as pl
        from pl_bolts.models.vision import ImageGPT

        dm = MNISTDataModule('.')
        model = ImageGPT(dm)

        pl.Trainer(gpu=4).fit(model)

    As script:

    .. code-block:: bash

        cd pl_bolts/models/vision/image_gpt
        python igpt_module.py --learning_rate 1e-2 --batch_size 32 --gpus 4
    """

    def __init__(
        self,
        embed_dim: int = 16,
        heads: int = 2,
        layers: int = 2,
        pixels: int = 28,
        vocab_size: int = 16,
        num_classes: int = 10,
        classify: bool = False,
        batch_size: int = 64,
        learning_rate: float = 1e-2,
        steps: int = 25_000,
        data_dir: str = ".",
        num_workers: int = 8,
        **kwargs,
    ):
        """
        Args:
            embed_dim: the embedding dim
            heads: number of attention heads
            layers: number of layers
            pixels: number of input pixels
            vocab_size: vocab size
            num_classes: number of classes in the input
            classify: true if should classify
            batch_size: the batch size
            learning_rate: learning rate
            steps: number of steps for cosine annealing
            data_dir: where to store data
            num_workers: num_data workers
        """
        super().__init__()
        self.save_hyperparameters()

        # default to MNIST if no datamodule given
        # if datamodule is None:
        #     datamodule = FashionMNISTDataModule(
        #         self.hparams.data_dir, num_workers=self.hparams.num_workers
        #     )
        #     self.hparams.pixels = datamodule.size(1)
        #     self.hparams.num_classes = datamodule.num_classes

        self.gpt = GPT2(
            embed_dim=self.hparams.embed_dim,
            heads=self.hparams.heads,
            layers=self.hparams.layers,
            num_positions=self.hparams.pixels * self.hparams.pixels,
            vocab_size=self.hparams.vocab_size,
            num_classes=self.hparams.num_classes,
        )

        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.gpt.parameters(), lr=self.hparams.learning_rate)

        # paper states cosine annealing is only used for pretraining
        if self.hparams.classify:
            return optim

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.hparams.steps)
        return [optim], [sched]

    def forward(self, x, classify=False):
        x = _shape_input(x)

        # TODO(teddykoker): this is a hack to quantize images into `vocab_size` bins.
        # This only works with 1 channel images; something like KNN needs to be used
        # for RGB. Assumes data is in [0.0, 1.0].
        x = torch.round(x * (self.hparams.vocab_size - 1)).long()

        return self.gpt(x, classify)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.hparams.classify:
            clf_logits = self(x, classify=True)
            loss = self.criterion(clf_logits, y)
        else:
            logits = self(x)
            loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1).long())

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        result = {}
        if self.hparams.classify:
            clf_logits = self(x, classify=True)
            loss = self.criterion(clf_logits, y)
            _, preds = torch.max(clf_logits, 1)
            correct = preds == y
            result.update({"val_loss": loss, "correct": correct})
        else:
            logits = self(x)
            logits = logits.view(-1, logits.size(-1))
            loss = self.criterion(logits, x.view(-1).long())
            result.update({"val_loss": loss})

        return result

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        logs = {"val_loss": avg_loss}
        if self.hparams.classify:
            correct = torch.cat([x["correct"] for x in outs])
            logs["val_acc"] = correct.sum().float() / correct.shape[0]
        return {"val_loss": avg_loss, "log": logs}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outs):
        result = self.validation_epoch_end(outs)

        # replace valid stats with test stats becuase we are reusing function
        result["log"]["test_loss"] = result["log"].pop("val_loss")
        result["test_loss"] = result.pop("val_loss")
        if self.hparams.classify:
            result["log"]["test_acc"] = result["log"].pop("val_acc")
        return result

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embed_dim", type=int, default=16)
        parser.add_argument("--dataset", type=str, default="fashion_mnist")
        parser.add_argument("--data_dir", type=str, default=os.getcwd())
        parser.add_argument("--heads", type=int, default=2)
        parser.add_argument("--layers", type=int, default=8)
        parser.add_argument("--vocab_size", type=int, default=16)
        parser.add_argument("--classify", action="store_true", default=False)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--learning_rate", type=float, default=1e-2)
        parser.add_argument("--steps", type=int, default=25_000)
        return parser


def cli_main():
    from pl_bolts.datamodules import FashionMNISTDataModule, ImagenetDataModule

    parser = ArgumentParser()

    # trainer args
    parser = Trainer.add_argparse_args(parser)

    # model args
    parser = ImageGPT.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.dataset == "fashion_mnist":
        datamodule = FashionMNISTDataModule.from_argparse_args(args)

    elif args.dataset == "imagenet128":
        datamodule = ImagenetDataModule.from_argparse_args(args)

    model = ImageGPT(**args.__dict__)

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    cli_main()
