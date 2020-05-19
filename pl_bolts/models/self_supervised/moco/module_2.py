import pytorch_lightning as pl
import torch
from pl_bolts.models.self_supervised.moco.moco_transforms import \
    Moco2Imagenet128Transforms, Moco2CIFAR10Transforms, Moco2STL10Transforms
from torch import nn

from pl_bolts.datamodules import CIFAR10DataLoaders, STL10DataLoaders
from pl_bolts.datamodules.ssl_imagenet_dataloaders import SSLImagenetDataLoaders


class Moco(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        pass

    def get_dataset(self, name):
        if name == 'cifar10':
            return CIFAR10DataLoaders(self.hparams.data_dir)
        elif name == 'stl10':
            return STL10DataLoaders(self.hparams.data_dir)
        elif name == 'imagenet128':
            return SSLImagenetDataLoaders(self.hparams.data_dir)
        else:
            raise FileNotFoundError(f'the {name} dataset is not supported. Subclass \'get_dataset to provide'
                                    f'your own \'')

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), self.hparams.lr,
                                    momentum=self.hparams.momentum,
                                    weight_decay=self.hparams.weight_decay)
        return optimizer

    # TODO: implement training

    def train_dataloader(self):
        if self.hparams.dataset == 'cifar10':
            train_transform = Moco2CIFAR10Transforms().train_transform

        elif self.hparams.dataset == 'stl10':
            stl10_transform = Moco2STL10Transforms()
            train_transform = stl10_transform.train_transform

        elif self.hparams.dataset == 'imagenet128':
            train_transform = Moco2Imagenet128Transforms()
            train_transform = train_transform.train_transform

        loader = self.dataset.train_dataloader(self.hparams.batch_size, transforms=train_transform)
        return loader

    def val_dataloader(self):
        if self.hparams.dataset == 'cifar10':
            train_transform = Moco2CIFAR10Transforms().train_transform

        elif self.hparams.dataset == 'stl10':
            train_transform = Moco2STL10Transforms().train_transform

        elif self.hparams.dataset == 'imagenet128':
            train_transform = Moco2Imagenet128Transforms().train_transform

        loader = self.dataset.train_dataloader(self.hparams.batch_size, transforms=train_transform)
        return loader

    def training_step(self, batch, batch_idx):
        criterion = nn.CrossEntropyLoss()
