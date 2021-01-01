from argparse import ArgumentParser, Namespace
from typing import List, Optional, Sequence, Text, Tuple

import pytorch_lightning as pl
from torch.utils.data.dataset import random_split

from pl_bolts.datamodules import SRDataModule
from pl_bolts.datasets.mnist_dataset import SRMNISTDataset
from pl_bolts.datasets.sr_celeba_dataset import SRCelebADataset
from pl_bolts.datasets.sr_stl10_dataset import SRSTL10Dataset


def parse_args(args: Optional[Sequence[Text]], pl_module_cls: pl.LightningModule) -> Tuple[Namespace, int, List]:

    parser = ArgumentParser()
    parser.add_argument("--dataset", default="mnist", type=str, choices=["celeba", "mnist", "stl10"])
    parser.add_argument("--data_dir", default="./", type=str)
    parser.add_argument("--log_interval", default=1000, type=int)
    parser.add_argument("--scale_factor", default=4, type=int)
    parser.add_argument("--save_model_checkpoint", dest="save_model_checkpoint", action="store_true")

    parser = SRDataModule.add_argparse_args(parser)
    parser = pl_module_cls.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    return args


def prepare_datasets(dataset, scale_factor, data_dir):
    assert dataset in ["celeba", "mnist", "stl10"]

    if dataset == "celeba":
        dataset_cls = SRCelebADataset
        dataset_train = dataset_cls(scale_factor, root=data_dir, split="train")
        dataset_val = dataset_cls(scale_factor, root=data_dir, split="valid")
        dataset_test = dataset_cls(scale_factor, root=data_dir, split="test")

    elif dataset == "mnist":
        dataset_cls = SRMNISTDataset
        dataset_dev = dataset_cls(scale_factor, root=data_dir, train=True)
        dataset_train, dataset_val = random_split(dataset_dev, lengths=[55_000, 5_000])
        dataset_test = dataset_cls(scale_factor, root=data_dir, train=False)

    elif dataset == "stl10":
        dataset_cls = SRSTL10Dataset
        dataset_dev = dataset_cls(scale_factor, root=data_dir, split="train")
        dataset_train, dataset_val = random_split(dataset_dev, lengths=[4_500, 500])
        dataset_test = dataset_cls(scale_factor, root=data_dir, split="test")

    return (dataset_train, dataset_val, dataset_test)
