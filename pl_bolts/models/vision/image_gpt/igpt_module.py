import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import pytorch_lightning as pl
import numpy as np
from pl_bolts.datamodules import MNISTDataModule
from pl_bolts.models.vision.image_gpt.gpt2 import GPT2
from pl_bolts.datamodules import LightningDataModule


class ImageGPT(pl.LightningModule):
    def __init__(self,
                 datamodule: LightningDataModule = None,
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
                 **kwargs
                 ):
        super(ImageGPT, self).__init__()
        self.save_hyperparameters()

        # default to MNIST if no datamodule given
        if datamodule is None:
            datamodule = MNISTDataModule(self.hparams.data_dir, num_workers=self.hparams.num_workers)
        self.datamodule = datamodule
        num_pixels = self.datamodule.size(1)

        self.gpt = GPT2(
            embed_dim=self.hparams.embed_dim,
            heads=self.hparams.heads,
            layers=self.hparams.layers,
            num_positions=num_pixels * num_pixels,
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

    def forward(self, x):
        return self.gpt(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = _shape_input(x)

        if self.hparams.classify:
            clf_logits = self.gpt(x, classify=True)
            loss = self.criterion(clf_logits, y)
        else:
            logits = self.gpt(x)
            loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = _shape_input(x)

        if self.hparams.classify:
            clf_logits = self.gpt(x, classify=True)
            loss = self.criterion(clf_logits, y)
            _, preds = torch.max(clf_logits, 1)
            correct = preds == y
            return {"val_loss": loss, "correct": correct}
        else:
            logits = self.gpt(x)
            loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))
            return {"val_loss": loss}

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

    def setup(self, stage: str):
        ds = lambda x, y: TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

        train_x = np.load(self.hparams.train_x)
        train_y = np.load(self.hparams.train_y)
        test_x = np.load(self.hparams.test_x)
        test_y = np.load(self.hparams.test_y)

        train_ds = ds(train_x, train_y)
        train_size = int(0.9 * len(train_ds))
        self.train_ds, self.valid_ds = random_split(
            train_ds, [train_size, len(train_ds) - train_size]
        )

        self.test_ds = ds(test_x, test_y)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds, batch_size=self.hparams.batch_size, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.hparams.batch_size, num_workers=4
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embed_dim", type=int, default=16)
        parser.add_argument("--heads", type=int, default=2)
        parser.add_argument("--layers", type=int, default=8)
        parser.add_argument("--num_pixels", type=int, default=28)
        parser.add_argument("--vocab_size", type=int, default=16)
        parser.add_argument("--num_classes", type=int, default=10)
        parser.add_argument("--classify", action="store_true", default=False)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--learning_rate", type=float, default=1e-2)
        parser.add_argument("--steps", type=int, default=25_000)
        return parser


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)

    # model args
    parser = ImageGPT.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        datamodule = CIFAR10DataModule.from_argparse_args(args)
        datamodule.train_transforms = Moco2TrainCIFAR10Transforms()
        datamodule.val_transforms = Moco2EvalCIFAR10Transforms()

    elif args.dataset == 'stl10':
        datamodule = STL10DataModule.from_argparse_args(args)
        datamodule.train_dataloader = datamodule.train_dataloader_mixed
        datamodule.val_dataloader = datamodule.val_dataloader_mixed
        datamodule.train_transforms = Moco2TrainSTL10Transforms()
        datamodule.val_transforms = Moco2EvalSTL10Transforms()

    elif args.dataset == 'imagenet2012':
        datamodule = SSLImagenetDataModule.from_argparse_args(args)
        datamodule.train_transforms = Moco2TrainImagenetTransforms()
        datamodule.val_transforms = Moco2EvalImagenetTransforms()

    model = MocoV2(**args.__dict__, datamodule=datamodule)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)
