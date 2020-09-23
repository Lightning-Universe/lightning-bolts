import os
import random
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pl_examples.models.unet import UNet


class SemSegment(pl.LightningModule):
    """
    Basic Semantic Segmentation Module.
    By default it uses a UNet architecture which can easily be substituted.
    The default loss function is CrossEntropyLoss and it has been specified for the KITTI dataset.
    The default optimizer is Adam with Cosine Annealing learning rate scheduler.

    Args:
        data_dir:
        batch_size:
        lr:
        num_layers:
        features_start:
        bilinear:
    """
    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 lr: float,
                 num_layers: int,
                 features_start: int,
                 bilinear: bool, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = lr
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear

        self.net = UNet(num_classes=19, num_layers=self.num_layers,
                        features_start=self.features_start, bilinear=self.bilinear)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.35675976, 0.37380189, 0.3764753],
                                 std=[0.32064945, 0.32098866, 0.32325324])
        ])
        self.trainset = KITTI(self.data_dir, split='train', transform=self.transform)
        self.validset = KITTI(self.data_dir, split='valid', transform=self.transform)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        log_dict = {'train_loss': loss_val}
        return {'loss': loss_val, 'log': log_dict, 'progress_bar': log_dict}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        return {'val_loss': loss_val}

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x['val_loss'] for x in outputs]).mean()
        log_dict = {'val_loss': loss_val}
        return {'log': log_dict, 'val_loss': log_dict['val_loss'], 'progress_bar': log_dict}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser()
        parser.add_argument("--data_dir", type=str, help="path where dataset is stored")
        parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
        parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument("--bilinear", action='store_true', default=False,
                            help="whether to use bilinear interpolation or transposed")


def cli_main(hparams: Namespace):

    from pl_bolts.datamodules import KittiDataModule
    pl.seed_everything(1234)

    # data
    loaders = KittiDataModule()

    # args
    parser = ArgumentParser()
    parser = SemSegment.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # model
    model = SemSegment(**vars(hparams))

    # train
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, loaders.train_dataloader(args.batch_size), loaders.val_dataloader(args.batch_size))


if __name__ == '__main__':
    cli_main()

